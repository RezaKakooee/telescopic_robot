"""High-level RL Steering Environment for Native MuJoCo.

The RL policy emits a 2D desired heading [cmd_x, cmd_y] (in the look-ahead goal frame),
and the open-loop bar controller computes the 60 radial telescoping extensions at
every physics step.
"""
from __future__ import annotations

import numpy as np

from ._gym import gym, spaces
from .controller import bar_targets, desired_direction
from .mujoco_env import MujocoRadialSphereEnv

N_BASE_OBS = 7
N_ENDPOINT_OBS = 8
FAR_PILLAR = (20.0, 0.0, 20.0)


class MujocoSteeringEnv(gym.Env):
    """High-level 2D steering RL interface on native MuJoCo physics."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: DictConfig | None = None,
        decision_every: int | None = None,
        include_drive: bool | None = None,
        max_steps: int | None = None,
        env: MujocoRadialSphereEnv | None = None,
        **env_kwargs,
    ):
        super().__init__()
        # If max_steps is passed for steering, avoid early low-level truncation
        low_level_kwargs = dict(env_kwargs)
        if "max_steps" in low_level_kwargs:
            del low_level_kwargs["max_steps"]
        self.env = env if env is not None else MujocoRadialSphereEnv(config, max_steps=1000000, **low_level_kwargs)
        cfg = self.env.cfg
        rl = getattr(cfg, "rl", None)

        self.max_steps = int(max_steps if max_steps is not None else getattr(rl, "max_steps", cfg.env.max_steps))
        self.k = int(decision_every if decision_every is not None
                     else getattr(rl, "decision_every", 10))
        self.include_drive = bool(include_drive if include_drive is not None
                                  else getattr(rl, "include_drive", False))
        self.ctrl = cfg.controller

        self.k_obstacles = int(getattr(rl, "obs_obstacles", 3))
        self.n_lidar = int(getattr(rl, "obs_lidar", 16))
        self.lidar_range = float(getattr(rl, "lidar_range", 3.0))
        self.obs_endpoints = bool(getattr(rl, "obs_endpoints", True))
        self.smooth_alpha = float(getattr(rl, "smooth_alpha", 0.35))
        self.action_smoothness_cost = float(getattr(rl, "action_smoothness_cost", 0.02))

        n_act = 3 if self.include_drive else 2
        obs_dim = (N_BASE_OBS + 3 * self.k_obstacles + self.n_lidar +
                   (N_ENDPOINT_OBS if self.obs_endpoints else 0))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._last_cmd = np.array([1.0, 0.0], dtype=np.float32)
        self._last_raw_cmd = np.array([1.0, 0.0], dtype=np.float32)
        self._smoothed_cmd_world = None
        self._last_drive = 1.0
        self._info = None

    # ------------------------------------------------------------------
    # Goal Frame Heading & Observation
    # ------------------------------------------------------------------
    def _goal_dir(self, ball_xy: np.ndarray) -> np.ndarray:
        """Unit look-ahead direction toward the goal/path (world frame)."""
        g, _drive = desired_direction(ball_xy, self.env.path_pts,
                                      float(self.ctrl.lookahead))
        return g

    def _observe(self, info: dict) -> np.ndarray:
        g = self._goal_dir(info["ball_xy"])
        v = info["lin_vel"][:2]
        v_gf = np.array([v[0] * g[0] + v[1] * g[1],        # forward
                         g[0] * v[1] - g[1] * v[0]])       # lateral (left +)
        dist = info["distance"] / self.env.path_length
        parts = [v_gf, [info["ang_vel"][2]], [dist], self._last_cmd,
                 [self._last_drive]]

        if self.obs_endpoints:
            ball_xy = np.asarray(info["ball_xy"][:2], dtype=np.float32)
            goal_xy = np.asarray(self.env.scenario.goal[:2], dtype=np.float32)
            start_xy = np.asarray(self.env.scenario.spawn_xy[:2], dtype=np.float32)
            rel_goal = goal_xy - ball_xy
            parts.extend([ball_xy, goal_xy, start_xy, rel_goal])

        # Obstacle slots (padding with FAR_PILLAR)
        for _ in range(self.k_obstacles):
            parts.append(np.array(FAR_PILLAR, dtype=np.float32))

        # Raycast LiDAR (in goal frame)
        lidar_ranges = self.env.raycast_lidar(
            n_rays=self.n_lidar,
            max_range=self.lidar_range,
            g=g,
        )
        parts.append(lidar_ranges)

        return np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        obs_raw, info = self.env.reset(seed=seed, options=options)
        self._last_cmd = np.array([1.0, 0.0], dtype=np.float32)
        self._last_raw_cmd = np.array([1.0, 0.0], dtype=np.float32)
        self._smoothed_cmd_world = None
        self._last_drive = 1.0
        self._info = info
        self.rl_step_count = 0
        return self._observe(info), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        act = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        cmd_gf = act[:2]
        n = float(np.linalg.norm(cmd_gf))
        d_gf = cmd_gf / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
        drive = float((act[2] + 1.0) * 0.5) if (self.include_drive and len(act) > 2) else 1.0

        # Action rate penalty for smoothness
        action_delta = float(np.sum((cmd_gf - self._last_raw_cmd) ** 2))
        action_penalty = self.action_smoothness_cost * action_delta
        self._last_raw_cmd = cmd_gf.copy()
        self._last_cmd = d_gf
        self._last_drive = drive

        # Transform goal frame heading into world frame
        g = self._goal_dir(self._info["ball_xy"])
        raw_d_world = d_gf[0] * g + d_gf[1] * np.array([-g[1], g[0]], dtype=np.float32)

        if self._smoothed_cmd_world is None:
            self._smoothed_cmd_world = raw_d_world.copy()
        else:
            self._smoothed_cmd_world = (1.0 - self.smooth_alpha) * self._smoothed_cmd_world + self.smooth_alpha * raw_d_world

        s_norm = float(np.linalg.norm(self._smoothed_cmd_world))
        d_world = self._smoothed_cmd_world / s_norm if s_norm > 1e-6 else raw_d_world

        total_r = -action_penalty
        term, trunc = False, False
        info = self._info

        for _ in range(self.k):
            targets = bar_targets(
                quat=info["quat"],
                dirs_body=self.env.dirs_body,
                max_extend=self.env.max_extend,
                d_hat=d_world,
                drive=drive,
                back_gain=float(self.ctrl.back_gain),
                down_gain=float(self.ctrl.down_gain),
                base=float(getattr(self.ctrl, "base", 0.04)),
            )
            _sub_obs, sub_r, sub_term, _sub_trunc, info = self.env.step(targets)
            total_r += float(sub_r)
            if sub_term:
                term = True
                break

        self.rl_step_count += 1
        if self.rl_step_count >= self.max_steps:
            trunc = True

        self._info = info
        obs = self._observe(info)
        return obs, float(total_r), term, trunc, self._info

    def render(self, mode: str = "chase") -> np.ndarray:
        return self.env.render(mode=mode)

    def close(self) -> None:
        self.env.close()
