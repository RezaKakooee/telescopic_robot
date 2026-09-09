"""Rough-terrain travel: the drive gait plus bounded terrain feedback.

Split out of ``locomotion`` because it is a different job. The primitives
there shape one push wave and stop. This one closes a loop on measured
ground: ride height, foot load and a per-rod terrain ray.
"""

from __future__ import annotations

import numpy as np

from radial_sphere.gait import (curb_vault, drive_wave, lock_out_leading,
                                support_weight, underbelly_stance)
from radial_sphere.geometry import quat_to_rotmat

from .locomotion import gain_for_speed
from .suspension import SuspensionGains, SuspensionState, apply_suspension



# ---------------------------------------------------------------------------
# 16. traverse_rough_terrain (active suspension & underbelly stance over rocks)
# ---------------------------------------------------------------------------

def traverse_rough_terrain(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray = np.array([1.0, 0.0], dtype=np.float32),
    *,
    speed: float = 1.2,
    min_offset: float = 0.025,
    back_gain: float | None = None,
    lin_vel: np.ndarray | None = None,
    core_z: float | None = None,
    core_vz: float | None = None,
    target_ride_height: float = 0.28,
    suspension_kp: float = 0.75,
    suspension_kd: float = 0.15,
    suspension_force_compliance: float = 0.0018,
    nominal_support_force: float = 10.0,
    contact_forces: np.ndarray | None = None,
    enable_underbelly_contact: bool = True,
    underbelly_stance_gain: float = 0.42,
    underbelly_threshold_z: float = -0.20,
    enable_curb_vaulting: bool = True,
    curb_boost_gain: float = 2.6,
    terrain_clearances: np.ndarray | None = None,
    terrain_adaptation_gain: float = 0.85,
    hole_reach_gain: float = 0.045,
    suspension: SuspensionGains | None = None,
    suspension_state: SuspensionState | None = None,
    control_dt: float = 0.01,
    max_target_speed: float = 0.45,
    suspension_filter_time: float = 0.06,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Traverse non-smooth, rocky terrain using active terrain-filtering suspension.

    Rear-bottom drive is combined with bounded height, terrain and contact
    feedback. Suspension participation fades continuously towards the sides.
    A positive terrain ray offset, not zero contact force alone, requests
    extra reach into a depression. Leading drive is suppressed, but bounded
    downward corrections may still support leading rods.

    For continuous filtering pass one persistent SuspensionState per robot,
    seeded from current actuator targets, and recreate it after reset. The
    standard skill runner manages this automatically. Calls without state
    are deterministic but do not apply temporal filtering or rate limiting.

    Parameters
    ----------
    quat : (4,) body orientation quaternion [w, x, y, z].
    dirs_body : (60, 3) rod direction vectors in body frame.
    max_extend : maximum rod stroke (m).
    d_hat : (2,) commanded 2D travel direction.
    speed : target cruising speed (m/s).
    min_offset : baseline retracted rod length (m).
    back_gain : override for traveling wave gain.
    core_z : current vertical position of the ball core (m).
    core_vz : current vertical velocity of the ball core (m/s).
    target_ride_height : nominal world-frame core altitude (m), default 0.28m.
        Set this inside what the build can hold. The 15 cm core with a 16 cm
        stroke settles near 0.23 m in this rolling gait, and a target above
        the reachable range pins the height term at its clip, which turns the
        PD correction into a constant offset.
    suspension_kp : dimensionless height-error gain.
    suspension_kd : vertical-velocity gain in seconds.
    suspension_force_compliance : compliant compliance factor (m/N).
    nominal_support_force : threshold normal contact force before yielding (N).
    contact_forces : (60,) normal force per rod (N) from simulation sensors.
    enable_underbelly_contact : maintain low-profile ground support under chassis.
    underbelly_stance_gain : underbelly extension depth fraction.
    underbelly_threshold_z : vertical projection boundary for underbelly rods.
    enable_curb_vaulting : boost rear pusher rods against steep rock edges.
    suspension : a `SuspensionGains` holding the whole correction tuning. It
        replaces the nine separate gain keywords above, which stay accepted
        for existing callers. Pass one of these instead of spelling out five
        zeros to switch the feedback off; see `SuspensionGains.without_feedback`.
    suspension_state : persistent filter history; one instance per robot/run.
    control_dt : elapsed control interval in seconds (default 0.01).
    max_target_speed : rod-target slew limit in metres per second (default 0.45).
    suspension_filter_time : filter time constant in seconds (default 0.06).
    curb_boost_gain : multiplier on rear pushers.
    return_metadata : if True, return (targets, meta_dict).

    Returns
    -------
    targets : (60,) rod extension targets in [min_offset, max_extend].
    meta (optional) : telemetry dictionary with heave correction and compliance.
    """
    gains = suspension or SuspensionGains(
        target_ride_height=target_ride_height, kp=suspension_kp, kd=suspension_kd,
        force_compliance=suspension_force_compliance,
        nominal_support_force=nominal_support_force,
        terrain_adaptation=terrain_adaptation_gain, hole_reach=hole_reach_gain,
        max_target_speed=max_target_speed, filter_time=suspension_filter_time)

    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    # 1. Heading decomposition
    d = np.asarray(d_hat, dtype=np.float64)
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-6:
        d = np.array([1.0, 0.0], dtype=np.float64)
    else:
        d = d / d_norm

    lat = np.array([-d[1], d[0]], dtype=np.float64)

    u_long = dirs_world[:, 0] * d[0] + dirs_world[:, 1] * d[1]
    u_lat = dirs_world[:, 0] * lat[0] + dirs_world[:, 1] * lat[1]
    u_z = dirs_world[:, 2]

    gain = float(back_gain) if back_gain is not None else gain_for_speed(speed, max_extend)

    # Closed-loop velocity regulation: throttle wave gain and apply gentle drag brake if overshooting
    brake = 0.0
    if lin_vel is not None and len(lin_vel) >= 2:
        v_fwd = float(lin_vel[0] * d[0] + lin_vel[1] * d[1])
        if v_fwd > speed:
            overshoot = v_fwd - speed
            gain = gain * max(0.02, 1.0 - 3.0 * overshoot / max(speed, 0.2))
            brake = float(np.clip(1.5 * overshoot, 0.0, 0.35))

    # 2. Peristaltic traveling wave
    wave = drive_wave(u_long, u_lat, u_z, gain)

    # 3. Steep rock edge / curb boost
    if enable_curb_vaulting:
        wave = curb_vault(wave, u_long, u_z, curb_boost_gain)

    # 4. Low-profile underbelly ground support
    if enable_underbelly_contact:
        is_underbelly, support_stance = underbelly_stance(
            u_long, u_lat, u_z, underbelly_stance_gain, underbelly_threshold_z)
        if brake > 0.0:
            support_stance = support_stance * max(0.2, 1.0 - brake)
        wave = np.where(is_underbelly, np.maximum(wave, support_stance), wave)
    else:
        is_underbelly = np.zeros(len(dirs_body), dtype=bool)

    # 5. Strict anti-counter-torque lockout on rising terrain (drag brake allowed only on downward brake rods)
    is_brake = np.zeros(len(dirs_body), dtype=bool)
    if brake > 0.0:
        is_brake = (u_long > 0.05) & (u_z < -0.30)
        wave = np.where(is_brake, brake, wave)
    wave = lock_out_leading(wave, u_long, u_z, keep=is_brake if brake > 0.0 else None)

    # 6. Suspension corrections ride only on rods the gait already uses for
    # support. A leading rod extended into a pit plants itself against the far
    # wall and stops the robot.
    support = support_weight(u_long, brake_mask=is_brake.astype(np.float64))

    targets = min_offset + (max_extend - min_offset) * wave

    vz = core_vz if core_vz is not None else (float(lin_vel[2]) if lin_vel is not None and len(lin_vel) > 2 else 0.0)
    targets, meta = apply_suspension(
        targets, u_z, max_extend, core_z=core_z, core_vz=vz,
        min_offset=min_offset, contact_forces=contact_forces,
        terrain_clearances=terrain_clearances, support_weight=support,
        state=suspension_state, dt=control_dt, **gains.as_kwargs(),
    )
    meta["underbelly_active_count"] = int(np.sum(is_underbelly))
    return (targets, meta) if return_metadata else targets
