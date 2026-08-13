"""High-level steering environment for RL.

The RL agent plans the motion ("go 45° to the right"); the scripted bar
controller executes it.  This factors the problem:

    RL policy      →  desired direction (+ optional drive/speed)
    bar_targets    →  60 per-bar extension targets
    RadialSphereEnv →  physics

Action (2 or 3 dims, all in [-1, 1]):
    [0:2]  desired direction, expressed in the *goal frame* — the frame whose
           x-axis points along the controller's look-ahead direction.  (1, 0)
           means "straight toward the goal/path"; rotations are relative, so
           the policy is invariant to where the goal happens to be.
    [2]    (if ``include_drive``) drive in [-1, 1] → [0, 1]; 0 freezes the
           bars at base extension (stop), 1 is full effort.

Each high-level action is held for ``decision_every`` low-level env steps
(temporal abstraction), with the commanded direction fixed in the world frame
for the duration of the hold.

Observation (7 + 3·K dims, goal frame):
    [v_forward, v_lateral,   # ball xy velocity in the goal frame
     yaw_rate,               # angular velocity about z
     goal_dist,              # distance to goal / path_length
     last_cmd_x, last_cmd_y, # previous commanded direction (goal frame)
     last_drive,
     # K nearest obstacle pillars, each as
     (fwd, lat, gap),        # centre offset in the goal frame + distance from
                             # the ball to the pillar surface; empty slots are
                             # padded with a far dummy (20, 0, 20)
     lidar_0 .. lidar_L]     # L ray distances to walls/pillars (goal frame,
                             # normalised by rl.lidar_range; 1 = nothing hit)

Reward is the base env's (progress + success), summed over the hold.
"""
from __future__ import annotations

import numpy as np

from ._gym import gym, spaces
from .controller import bar_targets, desired_direction
from .radial_sphere import RadialSphereEnv

N_BASE_OBS = 7
FAR_PILLAR = (20.0, 0.0, 20.0)   # padding for absent obstacle slots


class SteeringEnv(gym.Env):
    """Plan headings with RL; roll the sphere with the scripted controller."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config=None, *, decision_every=None, include_drive=None,
                 env=None, **env_kwargs):
        super().__init__()
        self.env = env if env is not None else RadialSphereEnv(config, **env_kwargs)
        cfg = self.env.cfg
        rl = getattr(cfg, "rl", None)
        self.k = int(decision_every if decision_every is not None
                     else getattr(rl, "decision_every", 10))
        self.include_drive = bool(include_drive if include_drive is not None
                                  else getattr(rl, "include_drive", True))
        self.ctrl = cfg.controller

        self.k_obstacles = int(getattr(rl, "obs_obstacles", 3))
        self.n_lidar = int(getattr(rl, "obs_lidar", 16))
        self.lidar_range = float(getattr(rl, "lidar_range", 3.0))
        n_act = 3 if self.include_drive else 2
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=(N_BASE_OBS + 3 * self.k_obstacles + self.n_lidar,),
            dtype=np.float32)
        self._last_cmd = np.array([1.0, 0.0], dtype=np.float32)
        self._last_drive = 1.0
        self._info = None

    # ------------------------------------------------------------------
    def _goal_dir(self, ball_xy: np.ndarray) -> np.ndarray:
        """Unit look-ahead direction toward the goal/path (world frame)."""
        g, _drive = desired_direction(ball_xy, self.env.path_pts,
                                      float(self.ctrl.lookahead))
        return g

    def _observe(self, info) -> np.ndarray:
        g = self._goal_dir(info["ball_xy"])
        v = info["lin_vel"][:2]
        v_gf = np.array([v[0] * g[0] + v[1] * g[1],        # forward
                         g[0] * v[1] - g[1] * v[0]])       # lateral (left +)
        dist = info["distance"] / self.env.path_length
        parts = [v_gf, [info["ang_vel"][2]], [dist], self._last_cmd,
                 [self._last_drive]]
        if self.k_obstacles > 0:
            parts.append(self._obstacle_obs(info["ball_xy"], g))
        if self.n_lidar > 0:
            parts.append(self._lidar_obs(info["ball_xy"], g))
        return np.concatenate(parts).astype(np.float32)

    def _obstacle_obs(self, ball_xy: np.ndarray, g: np.ndarray) -> np.ndarray:
        """K nearest pillars as (forward, lateral, surface gap) in the goal frame."""
        slots = np.tile(np.asarray(FAR_PILLAR, dtype=np.float32),
                        (self.k_obstacles, 1))
        pillars = np.asarray(self.env.scenario.obstacles, dtype=np.float32)
        if len(pillars):
            rel = pillars[:, :2] - ball_xy[None, :]
            d = np.linalg.norm(rel, axis=1)
            order = np.argsort(d)[: self.k_obstacles]
            for s, j in enumerate(order):
                fwd = rel[j, 0] * g[0] + rel[j, 1] * g[1]
                lat = g[0] * rel[j, 1] - g[1] * rel[j, 0]
                slots[s] = (fwd, lat, d[j] - pillars[j, 2])
        return slots.reshape(-1)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        _obs, info = self.env.reset(seed=seed)
        self._info = info
        self._last_cmd = np.array([1.0, 0.0], dtype=np.float32)
        self._last_drive = 1.0
        return self._observe(info), info

    def _lidar_obs(self, ball_xy: np.ndarray, g: np.ndarray) -> np.ndarray:
        """L ray distances to walls + pillars, cast in the goal frame.

        Ray 0 points along g (toward the goal/path); rays go counter-clockwise.
        Values are distance / lidar_range, clipped to [0, 1].
        """
        rng = self.lidar_range
        out = np.full(self.n_lidar, rng, dtype=np.float32)
        walls = np.asarray(self.env.scenario.walls, dtype=np.float32).reshape(-1, 4)
        pillars = np.asarray(self.env.scenario.obstacles,
                             dtype=np.float32).reshape(-1, 3)
        ox, oy = float(ball_xy[0]), float(ball_xy[1])
        for i in range(self.n_lidar):
            a = i / self.n_lidar * 2.0 * np.pi
            ca, sa = np.cos(a), np.sin(a)
            # rotate the goal-frame ray into the world frame
            dx = ca * g[0] - sa * g[1]
            dy = ca * g[1] + sa * g[0]
            best = rng
            for x1, y1, x2, y2 in walls:
                ex, ey = x2 - x1, y2 - y1
                den = dx * ey - dy * ex
                if abs(den) < 1e-9:
                    continue
                t = ((x1 - ox) * ey - (y1 - oy) * ex) / den
                u = ((x1 - ox) * dy - (y1 - oy) * dx) / den
                if 0.0 < t < best and 0.0 <= u <= 1.0:
                    best = t
            for px, py, pr in pillars:
                fx, fy = px - ox, py - oy
                proj = fx * dx + fy * dy
                if proj <= 0:
                    continue
                perp2 = fx * fx + fy * fy - proj * proj
                if perp2 < pr * pr:
                    t = proj - float(np.sqrt(pr * pr - perp2))
                    if 0.0 < t < best:
                        best = t
            out[i] = best
        return out / rng

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        d = a[:2]
        n = float(np.linalg.norm(d))
        cmd = d / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
        drive = float((a[2] + 1.0) * 0.5) if self.include_drive else 1.0

        # goal frame → world frame at decision time; held for the whole hold
        g = self._goal_dir(self._info["ball_xy"])
        d_world = cmd[0] * g + cmd[1] * np.array([-g[1], g[0]])

        total_r, terminated, truncated = 0.0, False, False
        info = self._info
        for _ in range(self.k):
            targets = bar_targets(
                info["quat"], self.env.dirs_body, self.env.max_extend,
                d_world, drive,
                back_gain=float(self.ctrl.back_gain),
                down_gain=float(self.ctrl.down_gain),
            )
            _obs, r, terminated, truncated, info = self.env.step(
                self.env.action_model.encode(targets))
            total_r += float(r)
            if terminated or truncated:
                break

        self._info = info
        self._last_cmd = cmd
        self._last_drive = drive
        return self._observe(info), total_r, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def state(self):
        return self.env.state
