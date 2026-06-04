"""Open-loop locomotion controller for the radial-sphere robot.

Two pure functions:
    desired_direction — look-ahead path tracker returning a unit xy direction.
    bar_targets       — per-bar extension targets from orientation + drive direction.
"""
from __future__ import annotations

import numpy as np

from .geometry import quat_to_rotmat


def desired_direction(
    ball_xy: np.ndarray,
    path_pts: np.ndarray,
    lookahead: float = 0.9,
    goal_eps: float = 0.45,
) -> tuple[np.ndarray, float]:
    """Pick a look-ahead point on the path and return a unit xy direction.

    Returns:
        d_hat: unit direction vector (2,).
        drive: 1.0 while heading toward path; 0.0 once within goal_eps of the end.
    """
    dists = np.linalg.norm(path_pts - ball_xy[None, :], axis=1)
    closest = int(np.argmin(dists))
    end_dist = float(np.linalg.norm(path_pts[-1] - ball_xy))
    if end_dist < goal_eps:
        return np.array([1.0, 0.0]), 0.0

    target_idx = len(path_pts) - 1
    accum = 0.0
    for j in range(closest, len(path_pts) - 1):
        accum += np.linalg.norm(path_pts[j + 1] - path_pts[j])
        if accum >= lookahead:
            target_idx = j + 1
            break
    target = path_pts[target_idx]
    d = target - ball_xy
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.array([1.0, 0.0]), 0.0
    return d / n, 1.0


def bar_targets(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    drive: float = 1.0,
    back_gain: float = 0.9,
    down_gain: float = 0.4,
    base: float = 0.15,
) -> np.ndarray:
    """Compute per-bar extension targets in metres.

    Args:
        quat: wxyz quaternion of the sphere's orientation.
        dirs_body: (n_bars, 3) bar directions in the body frame.
        max_extend: maximum bar extension (metres).
        d_hat: unit xy direction to drive toward.
        drive: 0.0 freezes bars at ``base`` extension (goal reached).
        back_gain: weight on the backward-facing component (push factor).
        down_gain: weight on the downward component (stance factor).
        base: resting extension fraction of max_extend.

    Returns:
        targets: (n_bars,) extension values in [0, max_extend].
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T
    align = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    downward = np.clip(-dirs_world[:, 2], 0.0, 1.0)
    score = -back_gain * align + down_gain * downward
    frac = np.clip(base + 0.5 * drive * (1.0 + score), 0.0, 1.0)
    return frac * max_extend
