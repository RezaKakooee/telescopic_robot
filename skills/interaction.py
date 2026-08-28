"""Environment interaction skills: wall pushing, bracing, surface contact.

Every function follows the same contract:
    Input:  robot state (quat, dirs_body, max_extend) + skill params.
    Output: np.ndarray of shape (n_bars,) with rod extension targets.
"""
from __future__ import annotations

import numpy as np

from radial_sphere.geometry import quat_to_rotmat


# ---------------------------------------------------------------------------
# 8. push_against_wall
# ---------------------------------------------------------------------------

def push_against_wall(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    wall_normal: np.ndarray,
    *,
    push_strength: float = 0.85,
    min_offset: float = 0.025,
    stance_height: float = 0.045,
) -> np.ndarray:
    """Extend rods on the side facing the wall to brace/push against it.

    Parameters
    ----------
    wall_normal : (2,) unit vector pointing FROM the wall TOWARD the robot.
        E.g. if the wall is on the robot's right side, wall_normal ≈ [0, +1]
        (pointing leftward from the wall toward the robot center).

    The skill activates rods whose world-frame direction points TOWARD
    the wall (opposite to wall_normal) — i.e., the rods that would
    make contact with the wall surface.
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T
    u_z = dirs_world[:, 2]

    wn = np.asarray(wall_normal, dtype=np.float64)
    wn_norm = np.linalg.norm(wn)
    if wn_norm < 1e-6:
        # No valid wall direction — just stand
        targets = np.zeros(len(dirs_body), dtype=np.float32)
        targets[u_z < -0.30] = stance_height
        return targets

    wn = wn / wn_norm

    # Project each rod's xy direction onto the wall-facing axis
    # Rods pointing TOWARD the wall have negative projection onto wall_normal
    wall_proj = dirs_world[:, 0] * wn[0] + dirs_world[:, 1] * wn[1]

    # Rods facing the wall: wall_proj < -0.10
    toward_wall = np.clip((-wall_proj - 0.10) / 0.90, 0.0, 1.0)

    # Favor mid-height rods (not just bottom, not just top)
    height_factor = np.clip(1.0 - abs(u_z) / 0.80, 0.2, 1.0)

    wave = np.clip(toward_wall * height_factor * 1.5, 0.0, 1.0)

    # Suppress rods on the opposite side and top
    wave[wall_proj > 0.05] = 0.0
    wave[u_z > 0.50] = 0.0

    targets = min_offset + push_strength * (max_extend - min_offset) * wave

    # Maintain bottom support stance
    bottom_mask = u_z < -0.30
    targets[bottom_mask] = np.maximum(targets[bottom_mask], stance_height)

    return targets.astype(np.float32)
