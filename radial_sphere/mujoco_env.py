"""Direct Native MuJoCo Gym Environment for the Radial-Sphere Robot.

A sphere with 60 Fibonacci radial telescoping bars rolling in an arena/maze.
Directly interfaces with the official `mujoco` C-bindings (`import mujoco`),
delivering true rigid-body physics, realistic ground reaction forces, anisotropic
contact friction, and offscreen camera rendering with zero tensor overhead.
"""
from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import mujoco
from loguru import logger as log

from ._gym import gym, spaces
from .action import ActionModel
from .config import load_config
from .controller import desired_direction
from .geometry import quat_to_rotmat
from .mujoco_mjcf import build_mujoco_scene_mjcf
from .observation import ObservationModel
from .reward import RewardModel
from .scenario import generate_scenario


class MujocoRadialSphereEnv(gym.Env):
    """OpenAI-Gym-compatible environment running directly on native MuJoCo."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config=None,
        scenario=None,
        randomize: bool | None = None,
        max_steps: int | None = None,
        render_mode: str | None = "rgb_array",
        render_size: tuple[int, int] = (480, 640),
    ):
        super().__init__()
        self.cfg = config if config is not None else load_config()
        self._randomize = bool(randomize if randomize is not None
                               else getattr(self.cfg.scenario, "randomize", False))

        self.n_bars = int(self.cfg.robot.n_bars)
        self.max_extend = float(self.cfg.robot.max_extend)
        self.sphere_radius = float(self.cfg.robot.sphere_radius)
        self.base_ext = 0.15 * self.max_extend
        self.action_repeat = max(int(self.cfg.env.action_repeat) * 5, 5)
        self.max_steps = int(max_steps if max_steps is not None else self.cfg.env.max_steps)
        self.render_mode = render_mode
        self.render_size = render_size

        # Scenario setup
        self.scenario = scenario if scenario is not None \
            else generate_scenario(getattr(self.cfg.scenario, "kind", "maze"), self.cfg)
        self.path_pts = np.asarray(self.scenario.path_pts, dtype=np.float32)
        self.marker_pts = np.asarray(self.scenario.markers, dtype=np.float32).reshape(-1, 2)
        self.path_length = float(self.scenario.path_length)

        # Action, observation and reward models
        self.action_model = ActionModel(self.cfg)
        self.obs_model = ObservationModel(self.cfg, self.path_pts, self.path_length)
        self.reward_model = RewardModel(self.cfg)
        self.action_space = self.action_model.space()
        self.observation_space = self.obs_model.space()

        # Build MuJoCo Model & Data
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self.dirs_body: np.ndarray | None = None
        self._build_scene(self.scenario)

        self.step_count = 0
        self._prev_dist = 0.0
        self._info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Scene & Geom ID Management
    # ------------------------------------------------------------------
    def _build_scene(self, scenario) -> None:
        """Compile MJCF XML and initialize MuJoCo Model, Data, and Geom IDs."""
        wall_h = getattr(self.cfg.scenario, "maze", self.cfg.scenario)
        wall_height = float(getattr(wall_h, "wall_height", getattr(self.cfg.scenario, "wall_height", 0.22)))
        xml_str, dirs = build_mujoco_scene_mjcf(
            scenario=scenario,
            n_bars=self.n_bars,
            sphere_radius=self.sphere_radius,
            max_extend=self.max_extend,
            core_mass=float(getattr(self.cfg.robot, "core_mass", 0.5)),
            wall_height=wall_height,
        )
        self.dirs_body = dirs
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

        # Cache IDs
        self.core_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "core")
        self.core_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "core_geom")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")

        # Foot geom IDs & sleeve geom IDs
        self.foot_geom_ids = set()
        self.sleeve_geom_ids = set()
        for k in range(self.n_bars):
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{k}")
            if fid >= 0:
                self.foot_geom_ids.add(fid)
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"sleeve_{k}")
            if sid >= 0:
                self.sleeve_geom_ids.add(sid)

        # Wall geom IDs
        self.wall_geom_ids = set()
        for i in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if gname and (gname.startswith("wall_") or gname.startswith("pillar_")):
                self.wall_geom_ids.add(i)

        # Robot all geom IDs
        self.robot_geom_ids = {self.core_geom_id} | self.foot_geom_ids | self.sleeve_geom_ids
        for k in range(self.n_bars):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"inner_geom_{k}")
            if gid >= 0:
                self.robot_geom_ids.add(gid)

        # Precompute sector base hues (3 lat lines x 3 lon lines = 16 sectors)
        n_lat_bins, n_lon_bins = 4, 4
        sector_hues = [
            0.00, 0.08, 0.16, 0.25,  # Red, Orange, Amber, Green
            0.33, 0.45, 0.55, 0.62,  # Lime, Cyan, Sky Blue, Blue
            0.72, 0.80, 0.88, 0.95,  # Indigo, Purple, Magenta, Rose
            0.12, 0.28, 0.50, 0.85   # Gold, Emerald, Azure, Violet
        ]
        self._bar_hues = []
        self._bar_geom_ids = []
        for k, (ux, uy, uz) in enumerate(self.dirs_body):
            lat = np.arcsin(np.clip(uz, -1.0, 1.0))
            lon = np.arctan2(uy, ux)
            lat_idx = int(np.clip((lat + np.pi / 2) / np.pi * n_lat_bins, 0, n_lat_bins - 1))
            lon_idx = int(np.clip((lon + np.pi) / (2 * np.pi) * n_lon_bins, 0, n_lon_bins - 1))
            sec_id = (lat_idx * n_lon_bins + lon_idx) % len(sector_hues)
            self._bar_hues.append(sector_hues[sec_id])
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{k}")
            iid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"inner_geom_{k}")
            self._bar_geom_ids.append((fid, iid))

        # Renderer
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(self.model, height=self.render_size[0], width=self.render_size[1])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        if self._randomize:
            self.scenario = generate_scenario(
                getattr(self.cfg.scenario, "kind", "maze"), self.cfg, seed=seed
            )
            self.path_pts = np.asarray(self.scenario.path_pts, dtype=np.float32)
            self.marker_pts = np.asarray(self.scenario.markers, dtype=np.float32).reshape(-1, 2)
            self.path_length = float(self.scenario.path_length)
            self.obs_model = ObservationModel(self.cfg, self.path_pts, self.path_length)
            self._build_scene(self.scenario)

        mujoco.mj_resetData(self.model, self.data)

        # Set spawn position & resting bar extensions
        spawn_xy = np.asarray(self.scenario.spawn_xy, dtype=float)[:2]
        spawn_z = self.sphere_radius + 0.006 + 0.004 + self.base_ext + 0.013 + 0.005
        self.data.qpos[0:3] = [spawn_xy[0], spawn_xy[1], spawn_z]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity wxyz quaternion
        self.data.qpos[7:7 + self.n_bars] = self.base_ext
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.base_ext

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Let the ball settle on the floor for 50 sub-steps
        for _ in range(50):
            self.data.ctrl[:] = self.base_ext
            mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        ball_xy = self.data.qpos[0:2]
        self._prev_dist = self._nav_distance(ball_xy)
        self._info = self._get_info(wall_contact=False, goal_contact=False)
        obs = self._get_obs()
        return obs, self._info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray | dict) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Decode action targets
        if isinstance(action, dict):
            targets = np.array([action.get(f"slide_{k}", self.base_ext) for k in range(self.n_bars)], dtype=float)
        else:
            act_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if act_arr.min() < 0.0:
                act_arr = np.clip(act_arr, -1.0, 1.0)
                targets = (act_arr + 1.0) * 0.5 * self.max_extend
            else:
                targets = np.clip(act_arr, 0.0, self.max_extend)

        self.data.ctrl[:] = targets

        # 2. Physics Simulation Loop (action_repeat sub-steps)
        wall_contact = False
        goal_contact = False
        total_substep_reward = 0.0

        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)

            # Check contacts
            sub_wall, sub_goal = self._check_contacts()
            wall_contact = wall_contact or sub_wall
            goal_contact = goal_contact or sub_goal

            # Compute dense reward
            curr_d = self._nav_distance(self.data.qpos[0:2])
            sub_r, _ = self.reward_model.compute(
                dist=float(curr_d),
                prev_dist=float(self._prev_dist),
                reached=bool(sub_goal),
                wall_contact=bool(sub_wall),
            )
            self._prev_dist = curr_d
            total_substep_reward += float(sub_r)

        self.step_count += 1

        # 3. Check Termination & Info
        curr_dist = self._nav_distance(self.data.qpos[0:2])
        success = bool(goal_contact or curr_dist < 0.35)
        terminated = success
        truncated = bool(self.step_count >= self.max_steps)

        self._info = self._get_info(wall_contact=wall_contact, goal_contact=goal_contact, success=success)
        obs = self._get_obs()

        return obs, float(total_substep_reward), terminated, truncated, self._info

    # ------------------------------------------------------------------
    # Contact Detection
    # ------------------------------------------------------------------
    def _check_contacts(self) -> tuple[bool, bool]:
        """Inspect MuJoCo contacts for robot-wall and robot-goal collisions."""
        wall_contact = False
        goal_contact = False

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            r1 = g1 in self.robot_geom_ids
            r2 = g2 in self.robot_geom_ids
            if not (r1 or r2):
                continue
            other = g2 if r1 else g1
            if other in self.wall_geom_ids:
                wall_contact = True
            elif other == self.goal_geom_id:
                goal_contact = True

        return wall_contact, goal_contact

    # ------------------------------------------------------------------
    # Distance Metric
    # ------------------------------------------------------------------
    def _nav_distance(self, xy: np.ndarray) -> float:
        """Geodesic distance through maze if available, else Euclidean to goal."""
        d = self.scenario.nav_distance(xy)
        if d is not None:
            return float(d)
        return float(np.linalg.norm(self.scenario.goal[:2] - xy[:2]))

    # ------------------------------------------------------------------
    # State Extraction & Observation
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        root_state = np.concatenate([
            self.data.qpos[0:3],       # pos (3)
            self.data.qpos[3:7],       # quat wxyz (4)
            self.data.qvel[0:3],       # lin_vel (3)
            self.data.qvel[3:6],       # ang_vel (3)
        ]).astype(np.float32)
        joint_pos = self.data.qpos[7:7 + self.n_bars].astype(np.float32)
        return self.obs_model.observe(root_state, joint_pos)

    def _get_info(self, wall_contact: bool = False, goal_contact: bool = False, success: bool = False) -> dict[str, Any]:
        ball_xy = self.data.qpos[0:2].copy().astype(np.float32)
        quat = self.data.qpos[3:7].copy().astype(np.float32)
        lin_vel = self.data.qvel[0:3].copy().astype(np.float32)
        ang_vel = self.data.qvel[3:6].copy().astype(np.float32)
        joint_pos = self.data.qpos[7:7 + self.n_bars].copy().astype(np.float32)
        dist = self._nav_distance(ball_xy)

        return {
            "ball_xy": ball_xy,
            "quat": quat,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
            "joint_pos": joint_pos,
            "distance": float(dist),
            "goal_contact": bool(goal_contact),
            "wall_contact": bool(wall_contact),
            "success": bool(success),
            "step_count": int(self.step_count),
        }

    # ------------------------------------------------------------------
    # Fast LiDAR Raycasting
    # ------------------------------------------------------------------
    def raycast_lidar(self, n_rays: int = 16, max_range: float = 3.0, g: np.ndarray | None = None) -> np.ndarray:
        """Cast n_rays in the goal frame (ray 0 along g, rotating CCW)."""
        if g is None:
            g = np.array([1.0, 0.0], dtype=float)
        pnt = np.array([self.data.qpos[0], self.data.qpos[1], 0.20], dtype=np.float64)
        distances = np.ones(n_rays, dtype=np.float32)
        geom_id = np.zeros(1, dtype=np.int32)

        for i in range(n_rays):
            a = i / n_rays * 2.0 * np.pi
            ca, sa = np.cos(a), np.sin(a)
            dx = ca * g[0] - sa * g[1]
            dy = ca * g[1] + sa * g[0]
            vec = np.array([dx, dy, 0.0], dtype=np.float64)
            dist = mujoco.mj_ray(
                self.model, self.data, pnt, vec,
                geomgroup=None, flg_static=1,
                bodyexclude=self.core_body_id, geomid=geom_id
            )
            if dist >= 0 and dist < max_range:
                if geom_id[0] != self.floor_geom_id:
                    distances[i] = float(dist / max_range)
            else:
                distances[i] = 1.0

        return distances

    def _update_dynamic_colors(self) -> None:
        """Update bar geom colors based on sector base hue and instantaneous extension."""
        extensions = self.data.qpos[7:7 + self.n_bars]
        for k in range(self.n_bars):
            ext_frac = float(np.clip(extensions[k] / self.max_extend, 0.0, 1.0))
            hue = self._bar_hues[k]
            val = 0.28 + 0.72 * ext_frac
            sat = 0.65 + 0.35 * ext_frac
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            fid, iid = self._bar_geom_ids[k]
            if fid >= 0:
                self.model.geom_rgba[fid] = [r, g, b, 1.0]
            if iid >= 0:
                self.model.geom_rgba[iid] = [r, g, b, 1.0]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, mode: str = "chase", camera_name: str | None = None) -> np.ndarray:
        """Render RGB image from 'chase', 'bird_fixed', or 'dual' view."""
        self._update_dynamic_colors()

        if camera_name is None:
            camera_name = mode

        if camera_name == "dual":
            img_bird = self.render(camera_name="bird_fixed")
            img_chase = self.render(camera_name="chase")
            return np.concatenate([img_bird, img_chase], axis=1)

        if camera_name == "chase":
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.core_body_id
            cam.distance = 1.35
            cam.elevation = -32.0
            # Track azimuth in travel direction or goal direction
            v = self.data.qvel[0:2]
            if np.linalg.norm(v) > 0.05:
                yaw = float(np.degrees(np.arctan2(v[1], v[0])))
            else:
                g = self.scenario.goal[:2] - self.data.qpos[:2]
                yaw = float(np.degrees(np.arctan2(g[1], g[0])))
            cam.azimuth = yaw - 90.0
            self.renderer.update_scene(self.data, camera=cam)
        else:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id < 0:
                cam_id = 0
            self.renderer.update_scene(self.data, camera=cam_id)

        rgb = self.renderer.render()
        return rgb

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
