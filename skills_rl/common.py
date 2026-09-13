"""Rod-pattern helpers more than one primitive needs.

Anything used by exactly one primitive lives in that primitive's file. This is
only the genuinely shared machinery, and most of it is a thin wrapper over
`radial_sphere.gait`, which is the single source of truth for how a rod sector
is scored. Wrapping rather than copying keeps `skills_rl` and `skills` in step:
a change to the gait core reaches both.
"""

from __future__ import annotations

import numpy as np

from radial_sphere.gait import (TOP_LOCKOUT, drive_wave, lock_out_leading,
                                travel_frame)
from radial_sphere.geometry import quat_to_rotmat


def rods_world(state) -> np.ndarray:
    """(n_bars, 3) rod directions in the world frame."""
    return np.asarray(state.dirs_body) @ quat_to_rotmat(state.quat).T


def surface_frame(state, along: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project every rod onto the frame of the surface being ridden.

    Returns ``(u_long, u_lat, u_into)`` where ``u_into`` is how far a rod
    points into the surface, the role world ``-z`` plays on flat ground.

    `move` in `skills/` assumes the ground is below and scores rods on their
    z-component. That is one case of a general rule: push against whatever is
    carrying you. Using the surface normal here means the same primitive drives
    on a floor, a banked wall or the inside of a pipe.
    """
    n = state.floor_normal()
    dirs = rods_world(state)
    a = np.asarray(along, dtype=np.float64)
    if a.shape[0] == 2:
        a = np.array([a[0], a[1], 0.0])
    a = a - np.dot(a, n) * n                 # travel must lie in the surface
    mag = float(np.linalg.norm(a))
    if mag < 1e-9:                           # travel was parallel to the normal
        seed = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(seed, n)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        a = seed - np.dot(seed, n) * n
        mag = float(np.linalg.norm(a))
    a = a / mag
    lat = np.cross(n, a)

    u_long = dirs @ a
    u_lat = dirs @ lat
    u_into = dirs @ n
    return u_long, u_lat, u_into


def push_wave(u_long, u_lat, u_into, gain: float) -> np.ndarray:
    """The trailing push that makes the ball roll, with leading rods shut.

    ``u_into`` stands in for the z-component the gait core expects. On flat
    ground the surface normal is world ``-z``, so ``u_into`` is ``-u_z`` and
    the sign flip below reproduces `move` exactly.
    """
    u_z = -np.asarray(u_into, dtype=np.float64)
    wave = drive_wave(u_long, u_lat, u_z, gain)
    return lock_out_leading(wave, u_long, u_z)


def sector(u_axis, *, centre: float, width: float) -> np.ndarray:
    """A smooth 0..1 weight over rods pointing near ``centre`` along an axis.

    A hard mask makes a rod switch role in one step, which the actuator sees
    as a step input and the ball feels as a kick. The smoothstep here is the
    same shape `gait.support_weight` uses, for the same reason.
    """
    t = np.clip(1.0 - np.abs(np.asarray(u_axis, dtype=np.float64) - centre)
                / max(float(width), 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def shut_top(wave: np.ndarray, u_into) -> np.ndarray:
    """Zero rods on the far side of the core from the surface.

    A rod pointing away from the ground reaches nothing. Extending it only
    moves mass and, in a corridor, risks catching the ceiling.
    """
    u_z = -np.asarray(u_into, dtype=np.float64)
    out = np.asarray(wave, dtype=np.float64).copy()
    out[u_z > TOP_LOCKOUT] = 0.0
    return out


__all__ = ["rods_world", "surface_frame", "push_wave", "sector", "shut_top",
           "travel_frame"]
