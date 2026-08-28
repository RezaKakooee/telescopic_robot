"""Jumping skills: vertical jump, forward jump from standstill, running hurdle leap.

Every function follows the same contract:
    Input:  robot state (quat, dirs_body, max_extend) + skill params.
    Output: np.ndarray of shape (n_bars,) with rod extension targets.

Jump skills are **phase-based**: the caller passes a `phase` string
indicating the current stage of the jump maneuver. The skill returns
the rod targets for that single phase step.

Typical phase sequencing (managed by caller):
    "crouch" (N steps) → "takeoff" (N steps) → "airborne" (until z < threshold) → "landing"
"""
from __future__ import annotations

import numpy as np

from radial_sphere.geometry import quat_to_rotmat


# ---------------------------------------------------------------------------
# 9. jump_up  (stationary vertical jump)
# ---------------------------------------------------------------------------

def jump_up(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    *,
    phase: str = "stand",
    stance_height: float = 0.045,
    landing_standoff: float = 0.055,
) -> np.ndarray:
    """Stationary vertical jump.

    Phases
    ------
    "stand"    : Stable resting posture (z ~ 0.21m).
    "crouch"   : Deep retraction of all rods to store kinematic stroke (z → 0.16m).
    "takeoff"  : Simultaneous 100% impulse on all downward rods (vz ≈ +3.3 m/s).
    "airborne" : Mid-air tuck holding compact spherical profile.
    "landing"  : Compliant touchdown suspension to absorb impact.
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T
    u_z = dirs_world[:, 2]

    targets = np.zeros(len(dirs_body), dtype=np.float32)

    if phase == "crouch":
        targets[:] = 0.00
    elif phase == "takeoff":
        ground_mask = u_z < 0.10
        targets[ground_mask] = max_extend
        targets[u_z > 0.15] = 0.0
    elif phase == "airborne":
        targets[:] = 0.015
    elif phase == "landing":
        bottom_mask = u_z < -0.20
        targets[bottom_mask] = landing_standoff
    else:  # "stand"
        bottom_mask = u_z < -0.30
        targets[bottom_mask] = stance_height

    return targets


# ---------------------------------------------------------------------------
# 10. jump_forward_while_stopped  (directional forward jump from standstill)
# ---------------------------------------------------------------------------

def jump_forward_while_stopped(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    phase: str = "stand",
    power: float = 1.0,
    stance_height: float = 0.045,
    landing_standoff: float = 0.055,
    rollout_gain: float = 0.12,
) -> np.ndarray:
    """Forward-biased jump from standstill.

    The takeoff impulse is rear-biased so the ball launches both upward
    and forward simultaneously.

    `power` scales the take-off stroke, from a full-effort leap at 1.0 down to
    a short hop. Unlike the locomotion gait -- where cutting the stroke stops
    the rods reaching the ground at all -- here they start fully crouched, so
    a partial extension still pushes off; it just pushes off less. That makes
    jump height a continuous, commandable quantity rather than one fixed leap.

    Phases
    ------
    "stand"    : Stable resting posture.
    "crouch"   : Deep retraction to store stroke.
    "takeoff"  : Rear-biased 100% impulse (vx ≈ +0.4 m/s, vz ≈ +2.6 m/s).
    "airborne" : Mid-air tuck.
    "landing"  : Compliant touchdown with forward rollout torque.
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_z = dirs_world[:, 2]

    targets = np.zeros(len(dirs_body), dtype=np.float32)

    if phase == "crouch":
        targets[:] = 0.00
    elif phase == "takeoff":
        ground_mask = u_z < 0.10
        forward_bias = np.clip(1.0 - 0.85 * np.maximum(u_long, -0.3), 0.35, 1.0)
        scale = float(np.clip(power, 0.0, 1.0))
        targets[ground_mask] = max_extend * forward_bias[ground_mask] * scale
        targets[u_long > 0.15] = 0.0
        targets[u_z > 0.15] = 0.0
    elif phase == "airborne":
        targets[:] = 0.015
    elif phase == "landing":
        bottom_mask = u_z < -0.20
        targets[bottom_mask] = landing_standoff
        rear_pusher = (u_long < -0.15) & (u_z < 0.0)
        targets[rear_pusher] = rollout_gain
        targets[u_long > 0.0] = 0.0
    else:  # "stand"
        bottom_mask = u_z < -0.30
        targets[bottom_mask] = stance_height

    return targets


# ---------------------------------------------------------------------------
# 11. jump_forward_while_moving  (running hurdle leap)
# ---------------------------------------------------------------------------

def jump_forward_while_moving(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    phase: str = "sprint",
    landing_standoff: float = 0.055,
    rollout_gain: float = 0.10,
) -> np.ndarray:
    """Running hurdle leap — explosive launch while already sprinting.

    This is the proven high-speed forward jump that produces
    vx ≈ +2.5 m/s, vz ≈ +2.6 m/s and flies +1.1m in mid-air.

    Phases
    ------
    "sprint"   : High-speed rear-pusher drive building vx ~ 2.0 m/s.
    "dip"      : Kinematic pre-leap dip — all rods retract to 0 to store stroke.
    "launch"   : Full-cluster explosive impulse on all downward rods.
    "airborne" : Mid-air tuck for hurdle clearance.
    "landing"  : Compliant touchdown with forward rollout.
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_z = dirs_world[:, 2]

    targets = np.zeros(len(dirs_body), dtype=np.float32)

    if phase == "sprint":
        # Rear-pusher peristaltic drive (proven high-speed mechanics)
        rear_pusher = (u_long < -0.10) & (u_z < 0.10)
        targets[rear_pusher] = max_extend
        targets[u_long > -0.05] = 0.0
        targets[u_z > 0.10] = 0.0
    elif phase == "dip":
        # Full retraction to store kinematic stroke
        targets[:] = 0.00
    elif phase == "launch":
        # Explosive impulse on the ground-facing rods, biased to the rear and
        # with the leading sector held shut. A leading rod that extends while
        # the ball is rolling acts as a kickstand: it brakes the run-up and
        # converts forward speed into vertical speed, so the ball takes off
        # nearly stationary and drops onto the obstacle instead of over it.
        # Same masking as the standing forward jump, which is the strongest
        # take-off the 60 rods can produce while keeping forward carry.
        ground_mask = u_z < 0.10
        forward_bias = np.clip(1.0 - 0.85 * np.maximum(u_long, -0.3), 0.35, 1.0)
        targets[ground_mask] = max_extend * forward_bias[ground_mask]
        targets[u_long > 0.15] = 0.0
        targets[u_z > 0.15] = 0.0
    elif phase == "airborne":
        # Compact mid-air tuck for obstacle clearance
        targets[:] = 0.015
    elif phase == "landing":
        # Compliant touchdown with forward rollout
        bottom_mask = u_z < -0.20
        targets[bottom_mask] = landing_standoff
        rear_pusher = (u_long < -0.15) & (u_z < 0.0)
        targets[rear_pusher] = rollout_gain
    else:
        targets[:] = 0.0

    return targets


# ---------------------------------------------------------------------------
# 13. jump_to  (aimable standing jump: servo the take-off to a velocity)
# ---------------------------------------------------------------------------

def jump_to(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    phase: str = "stand",
    vel: np.ndarray | None = None,
    vx_target: float = 0.5,
    vz_target: float = 2.6,
    servo_band: float = 0.30,
    wall_lock: bool = False,
    drop_height: float | None = None,
    stance_height: float = 0.045,
    landing_standoff: float = 0.060,
) -> np.ndarray:
    """Standing jump aimed by TAKE-OFF VELOCITY, not by a fixed impulse.

    The open-loop jumps fire a fixed rod pattern and hope: identical commands
    scatter the landing by half a metre, because which rods happen to sit in
    the firing sector depends on the ball's orientation that instant. This
    skill closes the loop during the burn instead. While the rods are still
    on the ground it compares the measured velocity against the commanded
    ``(vx_target, vz_target)`` every step:

    * horizontal: the *leading* rod sector is attenuated in proportion to how
      much forward speed is still missing (and the *trailing* sector if it
      overshoots), steering the net push;
    * vertical: the CALLER ends the "takeoff" phase the moment measured vz
      reaches ``vz_target`` -- burn duration is the vertical knob.

    Thrust is weighted by each rod's downward-ness, so near-horizontal rods
    -- whose push direction is orientation lottery -- barely fire. That alone
    removes most of the old horizontal scatter.

    Phases: "stand", "crouch", "takeoff", "airborne", "landing".
    Landing has NO rollout drive: precision hops brake on touchdown instead.

    `vel` is the world-frame velocity (the runner routes it automatically).
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T
    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
    u_z = dirs_world[:, 2]

    targets = np.zeros(len(dirs_body), dtype=np.float32)

    if phase == "crouch":
        targets[:] = 0.00

    elif phase == "takeoff":
        vx_now, vy_now, vz_now = 0.0, 0.0, 0.0
        if vel is not None:
            vx_now = float(vel[0] * d_hat[0] + vel[1] * d_hat[1])
            vy_now = float(vel[0] * (-d_hat[1]) + vel[1] * d_hat[0])
            vz_now = float(vel[2]) if len(vel) > 2 else 0.0
        err = (float(vx_target) - vx_now) / max(servo_band, 1e-6)
        fwd = float(np.clip(err, 0.0, 1.0))     # still missing forward speed
        back = float(np.clip(-err, 0.0, 1.0))   # overshot; pull back
        if wall_lock:
            # An obstacle stands just ahead: the leading sector must never
            # extend, or it punches the face mid-burn. Horizontal speed is
            # then regulated from above only -- the asymmetric burn drives vx
            # up and the trailing-sector trim caps it at the target.
            fwd = 1.0

        # Vertical: rods that genuinely point down fire at FULL stroke while
        # vz is far from the target -- an early version weighted them by
        # downward-ness and lost up to 0.35 m of rise at some orientations,
        # because whichever rods sit under the ball are all the thrust there
        # is. Only the near-horizontal ring is tapered out: those rods push
        # sideways in whatever direction the orientation lottery put them,
        # which was most of the landing scatter, and they add almost no lift.
        vscale = float(np.clip((float(vz_target) - vz_now) / 0.60, 0.18, 1.0))
        ring_taper = np.clip(-u_z / 0.35, 0.0, 1.0)
        w = vscale * ring_taper
        w[u_long > 0.12] *= (1.0 - fwd)         # attenuate leading sector
        w[u_long < -0.12] *= (1.0 - back)       # or the trailing one

        # Lateral: nothing commands sideways speed, so trim it to zero. A rod
        # firing while it points to one side shoves the core the other way,
        # and un-trimmed that drift threw the ball off the SIDE of a pad as
        # often as short of it. Cut the sector feeding the drift.
        side = vy_now / max(servo_band, 1e-6)
        w[u_lat < -0.12] *= (1.0 - float(np.clip(side, 0.0, 1.0)))
        w[u_lat > 0.12] *= (1.0 - float(np.clip(-side, 0.0, 1.0)))

        ground = u_z < 0.10
        targets[ground] = max_extend * w[ground]
        targets[u_z > 0.15] = 0.0

    elif phase == "airborne":
        targets[:] = 0.015

    elif phase == "landing":
        cushion = landing_standoff
        if drop_height is not None:
            # Touchdown speed grows as sqrt(2 g h); give the cushion stroke
            # to match, against the 0.25 m drop the default was sized for.
            cushion = float(np.clip(
                landing_standoff * np.sqrt(max(drop_height, 0.01) / 0.25),
                0.04, max_extend * 0.75))
        targets[u_z < -0.20] = cushion

    else:  # "stand"
        targets[u_z < -0.30] = stance_height

    return targets
