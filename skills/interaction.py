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


# ---------------------------------------------------------------------------
# 16. chimney_climb  (wall-jump up, clamp to hold, friction-servo down)
# ---------------------------------------------------------------------------

def chimney_climb(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    wall_axis: np.ndarray | None = None,
    *,
    phase: str = "hold",
    side: int = +1,
    clamp_ext: float | None = None,
    push_lat: float = 0.45,
    push_z_lo: float = 0.20,
    push_z_hi: float = 0.85,
    clamp_lat: float = 0.70,
    clamp_z: float = 0.50,
    gear: float = 0.5,
    near_floor: bool = False,
    push_frac: float = 1.0,
    x_off: float = 0.0,
    tuck: float = 0.010,
    stance_height: float = 0.045,
) -> np.ndarray:
    """Climb a chimney -- two facing walls a little wider than the ball.

    Three mechanisms, all measured under free physics (no pinned state):

    * **push** -- the ball leans on one wall and fires the rods that point
      into that wall and downward. Radial rods make no torque about the core,
      so the reaction is a clean up-and-across shove: the ball flies to the
      other wall, gaining height. Alternating sides is the ascent. With the
      standard 0.16 m rods it climbs a 0.40 m shaft at about 1 m/s.
    * **hold** -- the near-horizontal rods on both sides press the walls at
      full stroke. Ten to sixteen feet, about 1 kN of clamp, friction holds
      the ball's weight with a creep of roughly a centimetre per second.
    * **descend** -- the same clamp at a *commanded* extension. Less
      extension, less friction, faster slide. The caller servos `clamp_ext`
      on the measured vertical speed to descend at whatever rate it wants.

    Phases: "stand", "launch", "push", "fly", "hold", "descend".
    `side` is +1 to push off the wall in the +`wall_axis` direction, -1 for
    the other. `near_floor` opens the landing gear underneath.
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    axis = np.asarray(wall_axis if wall_axis is not None else [0.0, 1.0], dtype=np.float64)
    n = float(np.linalg.norm(axis))
    axis = axis / n if n > 1e-6 else np.array([0.0, 1.0])

    u_lat = dirs_world[:, 0] * axis[0] + dirs_world[:, 1] * axis[1]
    u_z = dirs_world[:, 2]
    targets = np.zeros(len(dirs_body), dtype=np.float32)

    if phase == "push":
        m = (side * u_lat > push_lat) & (u_z < -push_z_lo) & (u_z > -push_z_hi)
        if abs(x_off) > 0.08:
            # Drifting along the shaft: drop the rods whose push would carry
            # the ball further out, so each shove also steers it back.
            u_x = dirs_world[:, 0] * axis[1] - dirs_world[:, 1] * axis[0]
            m &= (u_x * np.sign(x_off) > -0.25)
        targets[m] = float(np.clip(push_frac, 0.2, 1.0)) * max_extend

    elif phase == "launch":
        # Straight up off the floor: every downward rod, lateral ones held
        # in so nothing jams against the walls on the way up.
        m = (u_z < -0.10) & (np.abs(u_lat) < 0.60)
        targets[m] = max_extend

    elif phase == "fly":
        targets[:] = tuck

    elif phase in ("hold", "descend"):
        ext = max_extend if clamp_ext is None else float(np.clip(clamp_ext, 0.0, max_extend))
        m = (np.abs(u_lat) > clamp_lat) & (np.abs(u_z) < clamp_z)
        targets[m] = ext
        if near_floor:
            g = float(np.clip(gear, 0.0, 1.0)) * max_extend
            targets[u_z < -0.35] = np.maximum(targets[u_z < -0.35], g)

    else:  # "stand"
        targets[(u_z < -0.30) & (np.abs(u_lat) < 0.35)] = stance_height

    return np.clip(targets, 0.0, max_extend).astype(np.float32)
