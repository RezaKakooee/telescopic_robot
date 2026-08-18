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
    goal_eps: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Pick a look-ahead point on the path and return a unit xy direction.

    Returns:
        d_hat: unit direction vector (2,).
        drive: 1.0 while heading toward the path; 0.0 only at its endpoint.
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
    min_offset: float = 0.025,
    back_gain: float = 1.6,
) -> np.ndarray:
    """Compute per-bar extension targets based on physical peristaltic cam mechanics.

    Physical Locomotion Principle:
    - Retracted / Neutral state: Maintains a baseline offset standoff (min_offset ≈ 2.5 cm)
      so rods never fully submerge, providing continuous rolling clearance and structured aesthetics.
    - Rear-downward quadrant (trailing side): Expands smoothly from min_offset -> max_extend
      into an eccentric wave that drives the sphere forward.
    - Front & Top: Held at baseline offset (min_offset).

    Args:
        quat: wxyz quaternion of the sphere's orientation.
        dirs_body: (n_bars, 3) bar directions in the body frame.
        max_extend: maximum bar extension (metres).
        d_hat: unit xy direction to drive toward.
        drive: 0.0 freezes bars at ``min_offset`` extension (goal reached).
        min_offset: minimum baseline rod extension when compressed (metres).
        back_gain: gain scaling the eccentric pushing wave amplitude.

    Returns:
        targets: (n_bars,) extension values in [min_offset, max_extend].
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    # Longitudinal coordinate along travel direction (-1 rear, +1 front)
    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    # Vertical coordinate (-1 down, +1 up)
    u_z = dirs_world[:, 2]

    # Rear factor: non-zero only for trailing bars (u_long < 0)
    rear = np.clip(-u_long, 0.0, 1.0)
    # Downward factor: non-zero only for lower hemisphere (u_z < 0)
    down = np.clip(-u_z + 0.1, 0.0, 1.0)

    # Smooth eccentric pushing wave in the rear-downward quadrant
    wave = np.clip((rear ** 1.1) * (down ** 0.9) * back_gain, 0.0, 1.0)
    return min_offset + drive * (max_extend - min_offset) * wave
