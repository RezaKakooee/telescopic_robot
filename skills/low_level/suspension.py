"""Shared, bounded terrain feedback for rolling skills (SI units)."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class SuspensionGains:
    """One tuning set for the rod-support corrections.

    These nine numbers used to travel as nine separate keywords through
    ``apply_suspension``, ``traverse_rough_terrain`` and ``stay_in_boundary``,
    with each signature restating the defaults. Passing one object instead
    keeps a tuning together and makes "feedback off" a named thing rather
    than five zeros written out by hand at every call site.

    ``target_ride_height`` is a world-frame core altitude. Keep it inside what
    the build can hold: a 15 cm core with a 16 cm stroke settles near 23 cm in
    the rolling gait, and a higher target pins the height term at its clip,
    which turns the PD correction into a constant offset.
    """

    target_ride_height: float = 0.28
    kp: float = 0.75
    kd: float = 0.15
    force_compliance: float = 0.0018
    nominal_support_force: float = 10.0
    terrain_adaptation: float = 0.85
    hole_reach: float = 0.045
    max_target_speed: float = 0.45
    filter_time: float = 0.06

    def without_feedback(self) -> "SuspensionGains":
        """The same tuning with every feedback term zeroed.

        Useful as the control arm of a comparison: the gait, the ride-height
        target and the slew limit stay identical, only the corrections stop.
        """
        return replace(self, kp=0.0, kd=0.0, force_compliance=0.0,
                       terrain_adaptation=0.0, hole_reach=0.0)



@dataclass
class SuspensionState:
    """Per-robot filter history. Create a fresh instance after every reset."""

    targets: np.ndarray | None = None
    forces: np.ndarray | None = None
    terrain: np.ndarray | None = None
    vz: float = 0.0


def apply_suspension(targets, u_z, max_extend, *, core_z, core_vz,
                     gains=SuspensionGains(), min_offset=0.025,
                     contact_forces=None, terrain_clearances=None,
                     support_weight=None, state=None, dt=0.01):
    """Apply smooth, limited extension corrections, never forces or pose edits.

    Ray offsets are signed vertical terrain heights relative to a z=0 floor.
    Positive means a measured depression, not merely an unloaded foot.
    Pass persistent state for temporal filtering and a physical slew limit.

    ``gains`` carries the whole tuning. Those numbers live on
    :class:`SuspensionGains` and nowhere else, so a default cannot drift
    between this function and its callers.

    ``support_weight`` is a per-rod weight in [0, 1] naming the rods the
    caller already uses to carry the robot. Corrections are scaled by it.
    Without it every downward rod is corrected, including leading rods that
    the gait deliberately keeps retracted. A leading rod driven into a pit
    becomes an anchor and stops the robot, so callers with a travel
    direction should always supply this weight.
    """
    target_ride_height = gains.target_ride_height
    suspension_kp, suspension_kd = gains.kp, gains.kd
    suspension_force_compliance = gains.force_compliance
    nominal_support_force = gains.nominal_support_force
    terrain_adaptation_gain = gains.terrain_adaptation
    hole_reach_gain = gains.hole_reach
    max_target_speed, filter_time = gains.max_target_speed, gains.filter_time

    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    if max_target_speed <= 0 or not np.isfinite(max_target_speed):
        raise ValueError("max_target_speed must be finite and positive")
    targets = np.asarray(targets, dtype=np.float64)
    u_z = np.asarray(u_z, dtype=np.float64)
    forces = np.zeros_like(targets) if contact_forces is None else np.asarray(contact_forces, dtype=float)
    terrain = np.zeros_like(targets) if terrain_clearances is None else np.asarray(terrain_clearances, dtype=float)
    if forces.shape != targets.shape or terrain.shape != targets.shape:
        raise ValueError("Terrain and force arrays must have one entry per rod")
    forces = np.nan_to_num(forces, nan=0.0, posinf=100.0, neginf=0.0).clip(0, 100)
    terrain = np.nan_to_num(terrain, nan=0.0, posinf=0.0, neginf=0.0).clip(-0.12, 0.12)
    vz = float(core_vz or 0.0)
    alpha = -np.expm1(-dt / max(filter_time, 1e-6))
    if state is not None:
        if state.forces is None or state.forces.shape != targets.shape:
            state.forces = np.zeros_like(targets)
            state.terrain = np.zeros_like(targets)
        state.forces += alpha * (forces - state.forces)
        state.terrain += alpha * (terrain - state.terrain)
        state.vz += alpha * (vz - state.vz)
        forces, terrain, vz = state.forces, state.terrain, state.vz

    # Smooth participation at the bottom/side boundary prevents role-switch kicks.
    blend = np.clip((-u_z - 0.25) / 0.45, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    if support_weight is not None:
        weight = np.asarray(support_weight, dtype=np.float64)
        if weight.shape != targets.shape:
            raise ValueError("support_weight must have one entry per rod")
        blend = blend * np.clip(np.nan_to_num(weight, nan=0.0), 0.0, 1.0)
    z_error = (target_ride_height if core_z is None else core_z) - target_ride_height
    height = float(np.clip(-suspension_kp * z_error - suspension_kd * vz, -0.025, 0.025))
    bump = -np.clip(suspension_force_compliance * np.maximum(forces - nominal_support_force, 0), 0, 0.012)
    terrain_trim = np.clip(terrain_adaptation_gain * terrain, -0.045, 0.045)
    # Only a positive ray measurement can trigger extra hole reach.
    hole = np.minimum(np.maximum(terrain, 0), hole_reach_gain) * np.clip(1 - forces / max(nominal_support_force, 1e-6), 0, 1)
    requested = np.clip(targets + blend * (height + terrain_trim + bump + hole), min_offset, max_extend)
    if state is not None:
        if state.targets is None:
            state.targets = np.full_like(targets, min_offset)
        smooth = state.targets + alpha * (requested - state.targets)
        step = max_target_speed * dt
        requested = state.targets + np.clip(smooth - state.targets, -step, step)
        state.targets = np.clip(requested, min_offset, max_extend).copy()
        requested = state.targets
    return requested.astype(np.float32), {
        "delta_skyhook": height, "z_error": float(z_error), "heave_vz": vz,
        "delta_bump_mean": float(np.mean(blend * bump)),
        "delta_terrain_mean": float(np.mean(blend * terrain_trim)),
        "delta_terrain_min": float(np.min(terrain_trim)),
        "delta_terrain_max": float(np.max(terrain_trim)),
        "delta_hole_mean": float(np.mean(blend * hole)),
        "support_rod_count": int(np.count_nonzero(blend > 0.05)),
    }
