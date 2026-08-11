"""Observation component: build the agent observation and its space.

Flat float32 vector (13 + n_bars):
    [quat            (4),   # wxyz orientation of the sphere
     lin_vel         (3),   # root linear velocity
     ang_vel         (3),   # root angular velocity
     bar_extensions  (n),   # per-bar extension / max_extend  ∈ [0, 1]
     goal_dir        (2),   # unit xy direction to the path look-ahead point
     goal_dist       (1)]   # distance to the path end, normalised by PATH_LENGTH

The goal_dir / goal_dist block is the navigation cue the policy needs to follow
the sinusoidal path without hard-coding the controller.
"""
from __future__ import annotations

import numpy as np

from ._gym import spaces
from .controller import desired_direction

N_ROOT = 4 + 3 + 3   # quat, lin_vel, ang_vel
N_NAV = 2 + 1        # goal_dir, goal_dist


class ObservationModel:
    def __init__(self, cfg, path_pts: np.ndarray, path_length: float):
        self.n_bars = int(cfg.robot.n_bars)
        self.max_extend = float(cfg.robot.max_extend)
        self.lookahead = float(cfg.controller.lookahead)
        self.path_pts = np.asarray(path_pts, dtype=np.float32)
        self.path_length = float(path_length)
        self.goal = self.path_pts[-1]
        self.dim = N_ROOT + self.n_bars + N_NAV

    def space(self) -> spaces.Box:
        return spaces.Box(-np.inf, np.inf, shape=(self.dim,), dtype=np.float32)

    def observe(self, root: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
        """Build the observation from one env's root_state (13,) + joint_pos (n,)."""
        quat = root[3:7]
        lin_vel = root[7:10]
        ang_vel = root[10:13]
        ball_xy = root[:2]

        d_hat, _drive = desired_direction(ball_xy, self.path_pts, self.lookahead)
        goal_dist = float(np.linalg.norm(self.goal - ball_xy)) / self.path_length
        bars = np.asarray(joint_pos, dtype=np.float32) / self.max_extend

        return np.concatenate(
            [quat, lin_vel, ang_vel, bars, d_hat, [goal_dist]]
        ).astype(np.float32)
