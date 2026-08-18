"""``RadialSphereEnv`` — an OpenAI-Gym-compatible wrapper around RoboVerse.

A sphere with N telescoping bars (Fibonacci-distributed on the unit sphere)
must roll along a sinusoidal path on the floor.  The agent commands a normalised
extension per bar; physics runs on the RoboVerse MuJoCo handler.

This module wires together the modular components::

    config  → mjcf (robot)  → ScenarioCfg → handler
            → ActionModel, ObservationModel, RewardModel

It also provides ``GymCompatWrapper`` / ``make_compat_env`` (classic 4-tuple API).
Policies live in the root agent scripts (``random_agent.py`` / ``heuristic_agent.py``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (PrimitiveCubeCfg, PrimitiveCylinderCfg,
                                     PrimitiveSphereCfg)
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.scene import SceneCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.setup_util import get_handler

from ._gym import gym
from .action import ActionModel
from .config import load_config
from .mjcf import build_robot_mjcf, rolling_radius
from .observation import ObservationModel
from .render import Renderer
from .reward import RewardModel
from .scenario import generate_scenario
from .snapshot import STORAGE_DIR

ROBOT_NAME = "radial_sphere"
GOAL_NAME = "goal"
GOAL_PAD_NAME = "goal_pad"
CAMERA_NAME = "chase"
PILLAR_HEIGHT = 0.8
PILLAR_PARK = 60.0     # unused pillars wait far outside the arena


def build_arena_scene_mjcf() -> str:
    """Build MJCF scene XML with custom skybox, dark slate tile floor, and studio lighting."""
    return """<mujoco model="sci_fi_arena">
    <visual>
        <headlight diffuse="0.48 0.50 0.55" specular="0.1 0.1 0.1" ambient="0.32 0.35 0.40"/>
        <quality shadowsize="4096" offsamples="8"/>
        <map shadowclip="0.1" shadowscale="0.8"/>
    </visual>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.13 0.16 0.22" rgb2="0.04 0.05 0.07" width="512" height="512"/>
        <texture name="texfloor" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.11 0.13 0.16" rgb2="0.13 0.15 0.19" mark="edge" markrgb="0.20 0.24 0.30"/>
        <material name="matfloor" texture="texfloor" texrepeat="35 35" texuniform="true"
                  reflectance="0.03" specular="0.2" shininess="0.4" roughness="0.5"/>
    </asset>
    <worldbody>
        <light directional="true" diffuse="0.85 0.88 0.92" specular="0.25 0.25 0.25"
               pos="3.0 2.0 20" dir="-0.08 -0.06 -1" castshadow="true"/>
        <light directional="true" diffuse="0.30 0.35 0.45" specular="0.05 0.05 0.05"
               pos="-10 -10 15" dir="0.4 0.4 -1" castshadow="false"/>
        <geom name="ground" type="plane" pos="0 0 0" size="60 60 0.001"
              condim="3" conaffinity="15" material="matfloor"/>
    </worldbody>
</mujoco>
"""


class RadialSphereEnv(gym.Env):
    """Gym/Gymnasium env: roll a telescoping sphere along a sinusoidal path.

    Single environment (``num_envs=1`` under the hood) to match the classic
    gym contract; observations and actions are per-env (no leading batch dim).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config=None, *, scenario=None, max_steps=None, output_dir=None,
                 render_mode="rgb_array", seed=None, enable_camera=None, randomize=None):
        super().__init__()
        # config may be a namespace, a path to a config.yaml, or None (default).
        self.cfg = config if (config is not None and not isinstance(config, (str, Path))) \
            else load_config(config)
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        # Camera is optional: rendering the chase camera inside get_states()
        # dominates step time, so RL training runs with it disabled.
        self.enable_camera = bool(enable_camera if enable_camera is not None
                                  else getattr(self.cfg.camera, "enabled", True))
        # Resample the goal every reset (goal kind only) so a policy cannot
        # memorise a single target.
        self._randomize = bool(randomize if randomize is not None
                               else getattr(self.cfg.scenario, "randomize", False))

        # Where the generated robot MJCF is written. Defaults under storage_local
        # so nothing lands outside it; agents pass their run dir.
        self.output_dir = Path(output_dir) if output_dir is not None else STORAGE_DIR / "_assets"

        self.n_bars = int(self.cfg.robot.n_bars)
        self.max_extend = float(self.cfg.robot.max_extend)
        self.base_ext = 0.15 * self.max_extend          # resting / spawn extension
        self.action_repeat = int(self.cfg.env.action_repeat)
        self.max_steps = int(max_steps if max_steps is not None else self.cfg.env.max_steps)
        self.slide_names = [f"slide_{k}" for k in range(self.n_bars)]

        # Scenario = the task (path navigation / goal finding). Default from config.
        self.scenario = scenario if scenario is not None \
            else generate_scenario(getattr(self.cfg.scenario, "kind", "path"), self.cfg)
        self.path_pts = np.asarray(self.scenario.path_pts, dtype=np.float32)
        self.marker_pts = np.asarray(self.scenario.markers, dtype=np.float32).reshape(-1, 2)
        self.path_length = float(self.scenario.path_length)
        log.info(f"Scenario: {self.scenario.kind!r} ({self.scenario.name}) "
                 f"goal={np.round(self.scenario.goal, 2).tolist()}")

        # modular components
        self.action_model = ActionModel(self.cfg)
        self.obs_model = ObservationModel(self.cfg, self.path_pts, self.path_length)
        self.reward_model = RewardModel(self.cfg)
        default_cam = getattr(getattr(self.cfg, "camera", None), "view", CAMERA_NAME)
        if default_cam in ("iso_fixed",):
            default_cam = "isometric"
        self.renderer = Renderer(self.cfg, camera_name=default_cam)
        self.action_space = self.action_model.space()
        self.observation_space = self.obs_model.space()

        # Cameras: multi-camera setup supporting chase, bird, bird_fixed, isometric.
        # Computed from the scenario; _track_camera translates follow cameras with the ball.
        self._setup_camera_pose()

        # build the simulator (robot MJCF + scenario + handler)
        self.dirs_body = self._build_handler()
        self._cam_ids = {}
        self._cam_id = None
        self._core_bid = None

        self._state = None          # last TensorState from the handler
        self._prev_dist = 0.0
        self.step_count = 0

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _setup_camera_pose(self) -> None:
        """Compute camera poses for chase, bird, bird_fixed, and isometric views."""
        cam = self.cfg.camera
        view = getattr(cam, "view", "chase")
        spawn = np.asarray(self.scenario.spawn_xy, dtype=float)
        # Aim along the path's initial tangent, not spawn → goal
        k = min(5, len(self.path_pts) - 1)
        d = np.asarray(self.path_pts[k], dtype=float) - spawn
        n = float(np.linalg.norm(d))
        d = d / n if n > 1e-6 else np.array([1.0, 0.0])   # travel direction (xy)

        # Chase view
        dist, h_chase = float(cam.distance), float(cam.height)
        ahead, lh = float(cam.look_ahead), float(cam.look_height)
        chase_pos0 = np.array([spawn[0] - d[0] * dist, spawn[1] - d[1] * dist, h_chase])
        chase_lookat0 = np.array([spawn[0] + d[0] * ahead, spawn[1] + d[1] * ahead, lh])
        chase_offset = chase_pos0 - np.array([spawn[0], spawn[1], 0.0])

        # Arena bounds for fixed bird's-eye and isometric views
        walls = np.asarray(self.scenario.walls, dtype=float).reshape(-1, 4)
        if len(walls):
            x_extent = walls[:, [0, 2]].max() - walls[:, [0, 2]].min()
            y_extent = walls[:, [1, 3]].max() - walls[:, [1, 3]].min()
            cx = (walls[:, [0, 2]].min() + walls[:, [0, 2]].max()) / 2
            cy = (walls[:, [1, 3]].min() + walls[:, [1, 3]].max()) / 2
        else:
            mid = (spawn + np.asarray(self.scenario.goal, dtype=float)) / 2
            cx, cy = float(mid[0]), float(mid[1])
            x_extent = abs(float(self.scenario.goal[0] - spawn[0])) + 2.0
            y_extent = abs(float(self.scenario.goal[1] - spawn[1])) + 2.0

        margin = float(getattr(cam, "bird_margin", 0.35))
        tan_half_h = 20.955 / (2.0 * 24.0)
        tan_half_v = tan_half_h * 720.0 / 1280.0
        h_bird = 1.02 * max((x_extent + 2 * margin) / (2 * tan_half_h),
                            (y_extent + 2 * margin) / (2 * tan_half_v))

        # Bird fixed view
        bird_fixed_pos0 = np.array([cx, cy - 0.005 * y_extent, h_bird])
        bird_fixed_lookat0 = np.array([cx, cy, 0.0])

        # Bird follow view
        back, h_bf = float(cam.bird_back), float(cam.bird_height)
        bird_pos0 = np.array([spawn[0] - d[0] * back, spawn[1] - d[1] * back, h_bf])
        bird_lookat0 = np.array([spawn[0], spawn[1], 0.0])
        bird_offset = bird_pos0 - np.array([spawn[0], spawn[1], 0.0])

        # Isometric view
        iso_pos0 = np.array([cx - x_extent * 0.72, cy - y_extent * 0.88, h_bird * 0.78])
        iso_lookat0 = np.array([cx, cy, 0.10])

        self._cameras_cfg = [
            ("chase", chase_pos0, chase_lookat0, True, chase_offset),
            ("bird_fixed", bird_fixed_pos0, bird_fixed_lookat0, False, None),
        ]
        if view == "bird":
            self._cameras_cfg.append(("bird", bird_pos0, bird_lookat0, True, bird_offset))
        elif view in ("isometric", "iso_fixed"):
            self._cameras_cfg.append(("isometric", iso_pos0, iso_lookat0, False, None))

        # Backwards-compatible attributes
        self._follow_cam = True
        self._cam_pos0 = chase_pos0
        self._cam_lookat0 = chase_lookat0
        self._cam_offset = chase_offset

    # ------------------------------------------------------------------
    # Simulator setup
    # ------------------------------------------------------------------
    def _build_handler(self) -> np.ndarray:
        """Write the robot MJCF, build the scenario + handler. Returns bar dirs."""
        mjcf_xml, dirs_body = build_robot_mjcf(
            n_bars=self.n_bars,
            sphere_radius=float(self.cfg.robot.sphere_radius),
            max_extend=self.max_extend,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        mjcf_path = str(self.output_dir / "radial_sphere.xml")
        with open(mjcf_path, "w") as f:
            f.write(mjcf_xml)
        log.info(f"Wrote MJCF → {mjcf_path}")

        scene_xml = build_arena_scene_mjcf()
        scene_path = str(self.output_dir / "arena_scene.xml")
        with open(scene_path, "w") as f:
            f.write(scene_xml)
        log.info(f"Wrote Scene MJCF → {scene_path}")

        robot = RobotCfg(
            name=ROBOT_NAME,
            num_joints=self.n_bars,
            mjcf_path=mjcf_path,
            usd_path=None,
            urdf_path=None,
            enabled_gravity=True,
            fix_base_link=False,
            enabled_self_collisions=False,
            actuators={n: BaseActuatorCfg(stiffness=float(self.cfg.robot.kp),
                                          damping=float(self.cfg.robot.kv))
                       for n in self.slide_names},
            joint_limits={n: (0.0, self.max_extend) for n in self.slide_names},
            control_type={n: "position" for n in self.slide_names},
            default_joint_positions={n: self.base_ext for n in self.slide_names},
        )

        scenario = ScenarioCfg(
            scene=SceneCfg(mjcf_path=scene_path),
            robots=[robot],
            simulator=str(self.cfg.sim.simulator),
            headless=bool(self.cfg.sim.headless),
            num_envs=1,
            add_default_ground=False,
        )
        scenario.cameras = [
            PinholeCameraCfg(name=name, width=1280, height=720,
                             pos=tuple(pos), look_at=tuple(look_at))
            for (name, pos, look_at, _, _) in self._cameras_cfg
        ] if self.enable_camera else []

        # Visual markers: small red breadcrumbs plus a glowing emerald goal beacon with target pad.
        scenario.objects = [
            PrimitiveSphereCfg(name=f"marker_{i}", radius=0.03,
                               color=[1.0, 0.2, 0.2], physics=PhysicStateType.RIGIDBODY)
            for i in range(len(self.marker_pts))
        ]
        scenario.objects.append(
            PrimitiveSphereCfg(name=GOAL_NAME, radius=0.10,
                               color=[0.1, 0.95, 0.45], physics=PhysicStateType.GEOM,
                               fix_base_link=True,
                               default_position=[float(self.scenario.goal[0]),
                                                 float(self.scenario.goal[1]), 0.10])
        )
        scenario.objects.append(
            PrimitiveCylinderCfg(name=GOAL_PAD_NAME, radius=0.35, height=0.01,
                                 color=[0.08, 0.80, 0.38], physics=PhysicStateType.GEOM,
                                 fix_base_link=True,
                                 default_position=[float(self.scenario.goal[0]),
                                                   float(self.scenario.goal[1]), 0.005])
        )

        # Obstacle pillars: always instantiate the maximum count (the sim world
        # is built once); episodes park the unused ones outside the arena.
        # Free-base but 100 kg — set_states can move them, the ball cannot.
        ob_cfg = getattr(self.cfg.scenario, "obstacles", None)
        pillar_r = float(getattr(ob_cfg, "radius", 0.25)) if ob_cfg else 0.25
        self.n_pillars = int(getattr(ob_cfg, "n_range", (3, 6))[1]) \
            if self.scenario.kind == "obstacle" else len(self.scenario.obstacles)
        for i in range(self.n_pillars):
            scenario.objects.append(
                PrimitiveCylinderCfg(name=f"pillar_{i}", radius=pillar_r,
                                     height=PILLAR_HEIGHT, mass=100.0,
                                     color=[0.55, 0.35, 0.20],
                                     physics=PhysicStateType.RIGIDBODY)
            )

        # Architectural maze walls: sleek graphite walls with substantial depth.
        mz = getattr(self.cfg.scenario, "maze", None)
        wall_t = float(getattr(mz, "wall_thickness", 0.08)) if mz else 0.08
        wall_h = float(getattr(mz, "wall_height", 0.50)) if mz else 0.50
        self.wall_height = wall_h
        self.n_walls = len(self.scenario.walls)
        for i, (wx1, wy1, wx2, wy2) in enumerate(
                np.asarray(self.scenario.walls, dtype=float).reshape(-1, 4)):
            length = float(np.hypot(wx2 - wx1, wy2 - wy1))
            yaw = float(np.arctan2(wy2 - wy1, wx2 - wy1))
            scenario.objects.append(
                PrimitiveCubeCfg(
                    name=f"wall_{i}",
                    size=[length + wall_t, wall_t, wall_h],
                    color=[0.18, 0.22, 0.28],
                    physics=PhysicStateType.GEOM,
                    fix_base_link=True,
                    default_position=[(wx1 + wx2) / 2, (wy1 + wy2) / 2, wall_h / 2],
                    default_orientation=[float(np.cos(yaw / 2)), 0.0, 0.0,
                                         float(np.sin(yaw / 2))],
                )
            )

        log.info(f"Using simulator: {self.cfg.sim.simulator}")
        self.handler = get_handler(scenario)
        self._cache_goal_contact_geoms()
        return dirs_body

    def _init_state(self) -> dict:
        """Per-env reset dict: breadcrumb markers + goal marker + sphere at spawn."""
        spawn = self.scenario.spawn_xy
        goal = self.scenario.goal
        spawn_z = rolling_radius(float(self.cfg.robot.sphere_radius), self.base_ext) + 0.005
        objects = {
            f"marker_{i}": {
                "pos": torch.tensor([float(pt[0]), float(pt[1]), 0.03]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            }
            for i, pt in enumerate(self.marker_pts)
        }
        objects[GOAL_NAME] = {
            "pos": torch.tensor([float(goal[0]), float(goal[1]), 0.10]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }
        objects[GOAL_PAD_NAME] = {
            "pos": torch.tensor([float(goal[0]), float(goal[1]), 0.005]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }
        # place this episode's pillars; park the surplus outside the arena
        pillars = np.asarray(self.scenario.obstacles, dtype=float).reshape(-1, 3)
        for i in range(self.n_pillars):
            if i < len(pillars):
                px, py = float(pillars[i, 0]), float(pillars[i, 1])
            else:
                px, py = PILLAR_PARK + 3.0 * i, PILLAR_PARK
            objects[f"pillar_{i}"] = {
                "pos": torch.tensor([px, py, PILLAR_HEIGHT / 2]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            }
        # Fixed MuJoCo bodies have no free joint, but metasim's set_states
        # updates their model pose directly. This is how level 3 changes maze
        # layout without turning walls into movable rigid bodies.
        for i, (x1, y1, x2, y2) in enumerate(self.scenario.walls):
            yaw = float(np.arctan2(y2 - y1, x2 - x1))
            objects[f"wall_{i}"] = {
                "pos": torch.tensor([(float(x1) + float(x2)) / 2,
                                     (float(y1) + float(y2)) / 2,
                                     self.wall_height / 2]),
                "rot": torch.tensor([float(np.cos(yaw / 2)), 0.0, 0.0,
                                     float(np.sin(yaw / 2))]),
            }
        return {
            "objects": objects,
            "robots": {
                ROBOT_NAME: {
                    "pos": torch.tensor([float(spawn[0]), float(spawn[1]), spawn_z]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {n: self.base_ext for n in self.slide_names},
                },
            },
        }

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Task randomisation only repositions existing objects, so no simulator
        # rebuild is needed. Random mazes have a constant wall-piece count.
        maze_level = int(getattr(getattr(self.cfg.scenario, "maze", None),
                                 "level", 1))
        if self._randomize and (self.scenario.kind in ("goal", "obstacle") or
                                (self.scenario.kind == "maze" and maze_level == 3)):
            self._set_scenario(generate_scenario(
                self.scenario.kind, self.cfg, seed=int(self.rng.integers(2 ** 31 - 1))))

        self.handler.set_states([self._init_state()])

        # Settle: hold motors at base extension while gravity seats the sphere.
        settle = [{ROBOT_NAME: {"dof_pos_target": {n: self.base_ext for n in self.slide_names}}}]
        for _ in range(int(self.cfg.env.n_settle_steps)):
            self.handler.set_dof_targets(settle)
            self.handler.simulate()

        self._track_camera()
        self._state = self.handler.get_states(mode="tensor")
        self.step_count = 0
        root, _ = self._root_and_joints()
        self._prev_dist = self._distance(root[:2])
        return self._observe(), self._info(root, self._prev_dist)

    def step(self, action):
        self.step_count += 1
        targets = self.action_model.decode(action)
        cmd = [{ROBOT_NAME: {"dof_pos_target": targets}}]
        goal_contact = False
        wall_contact = False
        for _ in range(self.action_repeat):
            self.handler.set_dof_targets(cmd)
            self.handler.simulate()
            goal_contact = self._has_goal_contact() or goal_contact
            wall_contact = self._has_wall_contact() or wall_contact
        self._track_camera()
        self._state = self.handler.get_states(mode="tensor")

        root, _ = self._root_and_joints()
        ball_xy = root[:2]
        dist = self._distance(ball_xy)
        reward, reached = self.reward_model.compute(
            dist, self._prev_dist, reached=goal_contact, wall_contact=wall_contact)
        self._prev_dist = dist

        terminated = bool(reached)
        truncated = self.step_count >= self.max_steps
        return self._observe(), reward, terminated, truncated, self._info(
            root, dist, success=reached, goal_contact=goal_contact,
            wall_contact=wall_contact)

    def render(self, camera_name: str | None = None):
        """Return the camera's RGB frame as ``(H, W, 3)`` uint8."""
        return self.renderer.render(self._state, camera_name=camera_name)

    def render_all(self) -> dict[str, np.ndarray]:
        """Return all available camera frames as a dict of ``{camera_name: frame}``."""
        return self.renderer.render_all(self._state)

    def close(self):
        if getattr(self, "handler", None) is not None:
            self.handler.close()
            self.handler = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _cache_goal_contact_geoms(self) -> None:
        """Resolve MuJoCo geom IDs used by the physical-touch success and wall contact tests."""
        model = self.handler.physics.model
        goal_ids = []
        robot_ids = []
        wall_ids = []
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name or ""
            local_name = name.rsplit("/", 1)[-1]
            if name.endswith(f"{GOAL_NAME}_model/sphere_geom"):
                goal_ids.append(geom_id)
            if name.startswith(f"{ROBOT_NAME}/") and (
                    local_name == "core_geom" or local_name.startswith("foot_")):
                robot_ids.append(geom_id)
            if "wall_" in name or "obstacle_" in name:
                wall_ids.append(geom_id)
        if len(goal_ids) != 1 or not robot_ids:
            raise RuntimeError(
                "could not resolve goal and robot collision geoms for contact success")
        self._goal_geom_id = goal_ids[0]
        self._robot_contact_geom_ids = frozenset(robot_ids)
        self._wall_contact_geom_ids = frozenset(wall_ids)

    def _has_goal_contact(self) -> bool:
        """True only for a current core/foot contact with the green goal."""
        data = self.handler.physics.data
        goal_id = self._goal_geom_id
        robot_ids = self._robot_contact_geom_ids
        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if ((geom1 == goal_id and geom2 in robot_ids) or
                    (geom2 == goal_id and geom1 in robot_ids)):
                return True
        return False

    def _has_wall_contact(self) -> bool:
        """True if any robot core/foot geom touches a wall or obstacle."""
        if not getattr(self, "_wall_contact_geom_ids", None):
            return False
        data = self.handler.physics.data
        wall_ids = self._wall_contact_geom_ids
        robot_ids = self._robot_contact_geom_ids
        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if ((geom1 in wall_ids and geom2 in robot_ids) or
                    (geom2 in wall_ids and geom1 in robot_ids)):
                return True
        return False

    def _set_scenario(self, scenario) -> None:
        """Swap the task in place (same sim world; used by goal randomisation)."""
        if len(scenario.walls) != self.n_walls:
            raise ValueError("randomized scenario changed the wall count; "
                             "the simulator world cannot be resized at reset")
        self.scenario = scenario
        self.path_pts = np.asarray(scenario.path_pts, dtype=np.float32)
        self.path_length = float(scenario.path_length)
        self.obs_model = ObservationModel(self.cfg, self.path_pts, self.path_length)

    def _root_and_joints(self):
        rs = self._state.robots[ROBOT_NAME]
        root = rs.root_state[0].detach().cpu().numpy()
        joints = rs.joint_pos[0].detach().cpu().numpy()
        return root, joints

    def _track_camera(self) -> None:
        """Translate follow cameras so the sphere stays centred (mujoco only).

        Reads the core body position straight from physics (no extra render) and
        moves the follow cameras by their configured offsets.
        """
        if not self.enable_camera:
            return
        phys = self.handler.physics
        model = phys.model
        if self._core_bid is None:
            self._core_bid = next(i for i in range(model.nbody)
                                  if model.body(i).name.endswith("core"))
        bx, by = phys.data.xpos[self._core_bid][:2]
        for name, _, _, follow, offset in self._cameras_cfg:
            if not follow or offset is None:
                continue
            if name not in self._cam_ids:
                try:
                    self._cam_ids[name] = next(i for i in range(model.ncam)
                                               if name in model.camera(i).name)
                except StopIteration:
                    continue
            cam_id = self._cam_ids[name]
            model.cam_pos[cam_id] = [bx + offset[0],
                                     by + offset[1],
                                     offset[2]]
        phys.forward()

    def _distance(self, ball_xy: np.ndarray) -> float:
        """Distance to the goal: geodesic (through walls' free space) when the
        scenario carries a field, else straight-line."""
        geo = self.scenario.nav_distance(ball_xy)
        if geo is not None:
            return geo
        return float(np.linalg.norm(self.obs_model.goal - ball_xy))

    def _info(self, root: np.ndarray, dist: float, *, success: bool = False,
              goal_contact: bool = False, wall_contact: bool = False) -> dict:
        return {
            "ball_xy": root[:2].copy(),
            "quat": root[3:7].copy(),
            "lin_vel": root[7:10].copy(),
            "ang_vel": root[10:13].copy(),
            "distance": dist,
            "step": self.step_count,
            "success": bool(success),
            "goal_contact": bool(goal_contact),
            "wall_contact": bool(wall_contact),
        }

    def _observe(self) -> np.ndarray:
        root, joints = self._root_and_joints()
        return self.obs_model.observe(root, joints)

    @property
    def state(self):
        """The last raw ``TensorState`` (handy for ObsSaver / cameras)."""
        return self._state


class GymCompatWrapper(gym.Wrapper):
    """Classic OpenAI Gym 4-tuple API: ``obs = reset()``, ``(obs,rew,done,info)``."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info["terminated"] = terminated
        info["truncated"] = truncated
        return obs, reward, done, info


def make_compat_env(**kwargs):
    return GymCompatWrapper(RadialSphereEnv(**kwargs))
