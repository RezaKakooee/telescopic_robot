"""The peristaltic drive wave, in one place.

Two callers grew their own copy of this arithmetic: ``controller.bar_targets``
and ``skills.low_level.locomotion.traverse_rough_terrain``. The copies shared every
magic number, so a fix applied to one silently left the other behind. Widening
the curb-vault sector did exactly that.

Everything here works in the *travel frame*: each rod's world direction
projected onto the commanded heading.

    u_long  +1 straight ahead, -1 straight behind
    u_lat   +1 to the left of travel, -1 to the right
    u_z     +1 straight up, -1 straight down

The functions return drive *weights* in [0, 1], not lengths. A caller turns a
weight into a target with ``min_offset + drive * (max_extend - min_offset) * w``.
Callers keep their own extras: incline assist, pipe bracing, braking rods,
suspension corrections. Only the shared core lives here.
"""
from __future__ import annotations

import numpy as np

from .geometry import quat_to_rotmat

#: Rods this far forward never extend. A leading rod snags on the terrain it
#: is about to roll over, and it fights the drive rods behind it.
LEADING_LOCKOUT = -0.05
#: Rods above this never extend. They cannot reach the ground.
TOP_LOCKOUT = 0.10
#: Baseline retracted rod length, in metres. Eleven skills defaulted to this
#: number independently. `stop` is the deliberate exception: it retracts every
#: rod but its stance cluster to a true zero.
MIN_OFFSET = 0.025
#: Ground-stance extension a skill holds when it is not driving.
STANCE_HEIGHT = 0.045


def travel_frame(quat, dirs_body, d_hat):
    """Project each rod direction onto the travel frame.

    Returns ``(u_long, u_lat, u_z)``, each shaped like ``dirs_body[:, 0]``.
    """
    d = np.asarray(d_hat, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    d = np.array([1.0, 0.0]) if norm < 1e-6 else d / norm
    lat = np.array([-d[1], d[0]])
    dirs_world = np.asarray(dirs_body) @ quat_to_rotmat(quat).T
    u_long = dirs_world[:, 0] * d[0] + dirs_world[:, 1] * d[1]
    u_lat = dirs_world[:, 0] * lat[0] + dirs_world[:, 1] * lat[1]
    return u_long, u_lat, dirs_world[:, 2]


def drive_wave(u_long, u_lat, u_z, gain):
    """Rear-bottom push wave, before any lockout.

    Three factors multiply: how far behind the rod points, how close it is to
    the rear-down quadrant, and how little it points sideways. A rod pushing
    sideways spends its stroke on lateral scrub, not on travel.
    """
    rear = np.clip((-u_long - 0.10) / 0.90, 0.0, 1.0)
    down = np.clip(1.0 - np.abs(u_z + 0.35) / 0.85, 0.0, 1.0)
    lat_tuck = np.clip(1.0 - 1.8 * (u_lat ** 2), 0.0, 1.0)
    return np.clip((rear ** 1.1) * down * float(gain) * lat_tuck, 0.0, 1.0)


def curb_vault(wave, u_long, u_z, boost_gain):
    """Boost the rods bearing on the face of a step, in place.

    A rod a little above the equator still presses on a tall rock, so the
    sector reaches up to ``u_z < 0.10`` rather than stopping at the waist.
    """
    is_rear_pusher = (u_long < -0.10) & (u_z < 0.10)
    wave[is_rear_pusher] = np.clip(wave[is_rear_pusher] * float(boost_gain), 0.0, 1.0)
    return wave


def underbelly_stance(u_long, u_lat, u_z, stance_gain, threshold_z=-0.20):
    """Minimum drive weight for the rods directly under the core.

    Returns ``(mask, weight)``. The weight grows with how steeply a rod points
    down and fades for rods pointing sideways. It is a floor, not a
    replacement: apply it with ``np.maximum``.
    """
    mask = (u_z < threshold_z) & (u_long <= -0.05)
    depth = np.clip((-u_z - abs(threshold_z)) / (1.0 - abs(threshold_z)), 0.0, 1.0)
    weight = depth * float(stance_gain) * np.clip(1.0 - 1.5 * (u_lat ** 2), 0.0, 1.0)
    return mask, weight


def lock_out_leading(wave, u_long, u_z, keep=None):
    """Zero every leading and top rod, in place.

    ``keep`` is an optional boolean mask of rods exempt from the leading
    lockout, used for braking rods that must press ahead of the core.
    """
    if keep is None:
        wave[u_long > LEADING_LOCKOUT] = 0.0
    else:
        wave[(u_long > LEADING_LOCKOUT) & ~keep] = 0.0
    wave[u_z > TOP_LOCKOUT] = 0.0
    return wave


def support_weight(u_long, brake_mask=None):
    """Per-rod weight in [0, 1] naming the rods that carry the robot.

    Corrections that ride on a leading rod defeat the lockout above: a leading
    rod driven into a pit plants against the far wall and stops the robot. The
    ramp is smooth so a rod does not gain or lose its correction in one step.
    """
    weight = np.clip((-u_long - 0.05) / 0.35, 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)
    if brake_mask is not None:
        weight = np.maximum(weight, np.asarray(brake_mask, dtype=np.float64))
    return weight
