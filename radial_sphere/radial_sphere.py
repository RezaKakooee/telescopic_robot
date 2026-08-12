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
from metasim.scenario.objects import PrimitiveSphereCfg
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
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
CAMERA_NAME = "chase"


class RadialSphereEnv(gym.Env):
    """Gym/Gymnasium env: roll a telescoping sphere along a sinusoidal path.

    Single environment (``num_envs=1`` under the hood) to match the classic
    gym contract; observations and actions are per-env (no leading batch dim).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config=None, *, scenario=None, max_steps=None, output_dir=None,
                 render_mode="rgb_array", seed=None):
        super().__init__()
        # config may be a namespace, a path to a config.yaml, or None (default).
        self.cfg = config if (config is not None and not isinstance(config, (str, Path))) \
            else load_config(config)
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

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
        self.renderer = Renderer(self.cfg, camera_name=CAMERA_NAME)
        self.action_space = self.action_model.space()
        self.observation_space = self.obs_model.space()

        # Chase camera: sit behind the sphere and look along the travel direction
        # (spawn → goal) so the goal stays ahead in frame. Computed from the
        # scenario; _track_camera then translates it with the ball each step.
        self._setup_camera_pose()

        # build the simulator (robot MJCF + scenario + handler)
        self.dirs_body = self._build_handler()
        self._cam_id = None
        self._core_bid = None

        self._state = None          # last TensorState from the handler
        self._prev_dist = 0.0
        self.step_count = 0

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _setup_camera_pose(self) -> None:
        """Compute the initial camera pose; follows the ball along spawn → goal.

        ``view: chase`` sits behind/over the shoulder; ``view: bird`` is a steep
        near-top-down (kept slightly off vertical, which is degenerate for the
        renderer's up = +Z frame).  Both aim toward the goal.
        """
        cam = self.cfg.camera
        self._follow_cam = bool(getattr(cam, "follow", True))
        view = getattr(cam, "view", "chase")
        spawn = np.asarray(self.scenario.spawn_xy, dtype=float)
        # Aim along the path's initial tangent, not spawn → goal: on an
        # out-and-back course the goal sits beside the spawn, and the camera
        # should watch the outbound leg (the ball then returns toward it).
        k = min(5, len(self.path_pts) - 1)
        d = np.asarray(self.path_pts[k], dtype=float) - spawn
        n = float(np.linalg.norm(d))
        d = d / n if n > 1e-6 else np.array([1.0, 0.0])   # travel direction (xy)

        if view == "bird":
            back, h = float(cam.bird_back), float(cam.bird_height)
            self._cam_pos0 = np.array([spawn[0] - d[0] * back, spawn[1] - d[1] * back, h])
            self._cam_lookat0 = np.array([spawn[0], spawn[1], 0.0])   # look at the ball
        else:  # "chase"
            dist, h = float(cam.distance), float(cam.height)
            ahead, lh = float(cam.look_ahead), float(cam.look_height)
            self._cam_pos0 = np.array([spawn[0] - d[0] * dist, spawn[1] - d[1] * dist, h])
            self._cam_lookat0 = np.array([spawn[0] + d[0] * ahead, spawn[1] + d[1] * ahead, lh])

        # constant offset from the ball's ground point → keeps the view angle fixed
        self._cam_offset = self._cam_pos0 - np.array([spawn[0], spawn[1], 0.0])

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
            robots=[robot],
            simulator=str(self.cfg.sim.simulator),
            headless=bool(self.cfg.sim.headless),
            num_envs=1,
        )
        scenario.cameras = [
            PinholeCameraCfg(name=CAMERA_NAME, width=1280, height=720,
                             pos=tuple(self._cam_pos0), look_at=tuple(self._cam_lookat0))
        ]

        # Visual markers: small red breadcrumbs along the path + a big green goal.
        scenario.objects = [
            PrimitiveSphereCfg(name=f"marker_{i}", radius=0.03,
                               color=[1.0, 0.2, 0.2], physics=PhysicStateType.RIGIDBODY)
            for i in range(len(self.marker_pts))
        ]
        scenario.objects.append(
            PrimitiveSphereCfg(name=GOAL_NAME, radius=0.08,
                               color=[0.2, 0.9, 0.3], physics=PhysicStateType.RIGIDBODY)
        )

        log.info(f"Using simulator: {self.cfg.sim.simulator}")
        self.handler = get_handler(scenario)
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
            "pos": torch.tensor([float(goal[0]), float(goal[1]), 0.08]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
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
        for _ in range(self.action_repeat):
            self.handler.set_dof_targets(cmd)
            self.handler.simulate()
        self._track_camera()
        self._state = self.handler.get_states(mode="tensor")

        root, _ = self._root_and_joints()
        ball_xy = root[:2]
        dist = self._distance(ball_xy)
        reward, reached = self.reward_model.compute(dist, self._prev_dist)
        self._prev_dist = dist

        terminated = bool(reached)
        truncated = self.step_count >= self.max_steps
        return self._observe(), reward, terminated, truncated, self._info(root, dist)

    def render(self):
        """Return the chase camera's RGB frame as ``(H, W, 3)`` uint8."""
        return self.renderer.render(self._state)

    def close(self):
        if getattr(self, "handler", None) is not None:
            self.handler.close()
            self.handler = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _root_and_joints(self):
        rs = self._state.robots[ROBOT_NAME]
        root = rs.root_state[0].detach().cpu().numpy()
        joints = rs.joint_pos[0].detach().cpu().numpy()
        return root, joints

    def _track_camera(self) -> None:
        """Translate the chase camera so the sphere stays centred (mujoco only).

        Reads the core body position straight from physics (no extra render) and
        moves the camera by the same xy, keeping the configured viewing angle.
        """
        if not self._follow_cam:
            return
        phys = self.handler.physics
        model = phys.model
        if self._cam_id is None:
            self._cam_id = next(i for i in range(model.ncam) if CAMERA_NAME in model.camera(i).name)
            self._core_bid = next(i for i in range(model.nbody)
                                  if model.body(i).name.endswith("core"))
        bx, by = phys.data.xpos[self._core_bid][:2]
        model.cam_pos[self._cam_id] = [bx + self._cam_offset[0],
                                       by + self._cam_offset[1],
                                       self._cam_offset[2]]
        phys.forward()

    def _distance(self, ball_xy: np.ndarray) -> float:
        return float(np.linalg.norm(self.obs_model.goal - ball_xy))

    def _info(self, root: np.ndarray, dist: float) -> dict:
        return {
            "ball_xy": root[:2].copy(),
            "quat": root[3:7].copy(),
            "distance": dist,
            "step": self.step_count,
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
