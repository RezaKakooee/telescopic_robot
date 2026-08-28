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
    # --- Optional Smooth Maneuver Enhancements ---
    enable_spline_heading: bool = False,
    spline_smoothing_weight: float = 0.8,
    enable_curvature_deceleration: bool = False,
    curvature_lookahead_dist: float = 1.2,
    curvature_brake_gain: float = 1.8,
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

    if enable_spline_heading and target_idx > closest + 1:
        # Smooth interpolation between adjacent waypoints to prevent angular snapping
        p_prev = path_pts[target_idx - 1]
        p_curr = path_pts[target_idx]
        p_smooth = (1.0 - spline_smoothing_weight) * p_prev + spline_smoothing_weight * p_curr
        target = p_smooth
    else:
        target = path_pts[target_idx]

    d = target - ball_xy
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.array([1.0, 0.0]), 0.0

    d_hat = d / n

    # Curvature-Adaptive Exponential Deceleration (Glide into Turns)
    drive_val = 1.0
    if enable_curvature_deceleration and target_idx < len(path_pts) - 1:
        v1 = path_pts[target_idx] - path_pts[max(0, target_idx - 1)]
        v2 = path_pts[min(len(path_pts) - 1, target_idx + 1)] - path_pts[target_idx]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-4 and n2 > 1e-4:
            cos_turn = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            turn_angle = np.arccos(cos_turn)  # radians of corner turn
            curvature = turn_angle / (0.5 * (n1 + n2))
            drive_val = float(1.0 / (1.0 + curvature_brake_gain * (curvature ** 2)))

    return d_hat, drive_val


def bar_targets(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    drive: float = 1.0,
    min_offset: float = 0.025,
    back_gain: float = 1.6,
    # --- 5 Optional Low-Level Control Enhancements ---
    enable_power_wave: bool = False,
    wave_power_exponent: float = 1.4,
    enable_flank_retraction: bool = False,
    lidar_ranges: np.ndarray | None = None,
    lidar_max_range: float = 3.0,
    flank_retract_dist: float = 0.45,
    flank_min_offset: float = 0.005,
    enable_camber_banking: bool = False,
    yaw_rate: float = 0.0,
    camber_bank_gain: float = 0.035,
    enable_contact_compliance: bool = False,
    contact_forces: np.ndarray | None = None,
    compliance_gain: float = 0.0005,
    max_contact_force: float = 40.0,
    enable_anti_stall_reflex: bool = False,
    forward_vel: float = 1.0,
    sim_time: float = 0.0,
    anti_stall_speed_threshold: float = 0.15,
    anti_stall_pulse_freq: float = 10.0,
    anti_stall_pulse_amp: float = 0.02,
    # --- 5 Optional Smooth Maneuver Enhancements ---
    enable_gaussian_stance: bool = False,
    gaussian_stance_sigma: float = 0.38,
    enable_gyroscopic_damping: bool = False,
    ang_vel: np.ndarray | None = None,
    gyroscopic_damping_gain: float = 0.025,
    enable_actuator_slew_rate: bool = False,
    last_targets: np.ndarray | None = None,
    actuator_max_vel: float = 0.35,
    actuator_dt: float = 0.05,
    # --- Adaptive Grouping Enhancement ---
    enable_adaptive_grouping: bool = False,
    group_size: int = 10,
    # --- Obstacle Passover / High-Step Curb Vaulting ---
    enable_curb_vaulting: bool = False,
    curb_boost_gain: float = 2.4,
    # --- Ground-Contacting Underbelly Stance Strategy ---
    enable_underbelly_contact: bool = False,
    underbelly_stance_gain: float = 0.55,
    underbelly_threshold_z: float = -0.20,
    # --- Active Terrain-Filtering Suspension Mechanism (Skyhook & Bump Absorber) ---
    enable_active_suspension: bool = False,
    core_z: float = 0.28,
    core_vz: float = 0.0,
    target_ride_height: float = 0.28,
    suspension_kp: float = 0.65,
    suspension_kd: float = 0.12,
    suspension_force_compliance: float = 0.0018,
    nominal_support_force: float = 10.0,
    # --- In-Pipe Circumferential Bracing Strategy (Transparent Glass Tube Inspection) ---
    enable_pipe_bracing: bool = False,
    pipe_bracing_gain: float = 0.42,
    # --- Incline Slope Traction Assist & Downhill Braking ---
    enable_incline_assist: bool = False,
    incline_pitch: float = 0.0,
    incline_boost_gain: float = 1.45,
) -> np.ndarray:
    """Compute per-bar extension targets based on physical peristaltic cam mechanics."""
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    # 1. Base Push Wave (Dynamic Peristaltic Drive)
    # Longitudinal coordinate along travel direction (-1 rear, +1 front)
    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
    u_z = dirs_world[:, 2]

    # Rear factor: trailing rods in rear hemisphere generate strong forward torque
    rear_factor = np.clip((-u_long - 0.10) / 0.90, 0.0, 1.0)
    # Downward stance factor: concentrates push in rear-downward quadrant
    down_factor = np.clip(1.0 - abs(u_z + 0.35) / 0.85, 0.0, 1.0)
    # Lateral tuck factor: suppress side flank rods from extending laterally
    lat_tuck = np.clip(1.0 - 1.8 * (u_lat ** 2), 0.0, 1.0)

    # Effective back gain (boosted on steep incline slopes)
    effective_gain = back_gain
    if enable_incline_assist and incline_pitch > 0.04:
        effective_gain = back_gain * (1.0 + incline_boost_gain * np.sin(incline_pitch))

    wave = np.clip((rear_factor ** 1.1) * down_factor * effective_gain * lat_tuck, 0.0, 1.0)

    # 2. Obstacle / Curb Vaulting Boost
    if enable_curb_vaulting:
        is_rear_pusher = (u_long < -0.10) & (u_z < 0.10)
        wave[is_rear_pusher] = np.clip(wave[is_rear_pusher] * curb_boost_gain, 0.0, 1.0)

    # 3. Ground-Contacting Underbelly Support (Capped low-profile so no seesaw tipping occurs)
    if enable_underbelly_contact:
        is_underbelly = (u_z < underbelly_threshold_z) & (u_long <= -0.05)
        depth_fraction = np.clip((-u_z - abs(underbelly_threshold_z)) / (1.0 - abs(underbelly_threshold_z)), 0.0, 1.0)
        support_stance = depth_fraction * underbelly_stance_gain * np.clip(1.0 - 1.5 * (u_lat ** 2), 0.0, 1.0)
        wave = np.where(is_underbelly, np.maximum(wave, support_stance), wave)

    # 4. In-Pipe Circumferential Bracing (Transparent Glass Conduit Inspection)
    if enable_pipe_bracing:
        radial_dist = np.sqrt(u_lat ** 2 + u_z ** 2)
        is_side_guide = (radial_dist > 0.35) & (u_z < 0.10) & (u_long <= 0.05)
        pipe_stance = pipe_bracing_gain * np.clip(1.0 - 0.70 * np.maximum(u_long, 0.0), 0.30, 0.60)
        is_rear_pusher = (u_long < -0.08) & (u_z < 0.10)
        wave = np.where(is_side_guide, np.maximum(wave, pipe_stance), wave)
        wave[is_rear_pusher] = np.maximum(wave[is_rear_pusher], 0.90)

    # Absolute guarantee: no rod in the leading forward sector (u_long > -0.05) or top (u_z > 0.10) ever extends
    wave[u_long > -0.05] = 0.0
    wave[u_z > 0.10] = 0.0

    targets = min_offset + drive * (max_extend - min_offset) * wave

    # Active Terrain-Filtering Suspension Mechanism:
    # 1. Skyhook Core Heave Canceling: Regulates core vertical height to target_ride_height (e.g. 0.28m)
    #    When climbing over boulders, downward rods actively yield/retract so the core stays flat.
    # 2. Local Bump Absorber: Individual rods hitting protruding rock peaks absorb load without lifting core.
    if enable_active_suspension:
        is_downward = u_z < underbelly_threshold_z
        z_err = float(core_z - target_ride_height)
        # Skyhook stroke adjustment: retracts if core is pushed up, extends if core is falling
        delta_skyhook = -float(suspension_kp * z_err + suspension_kd * core_vz)
        
        # Local bump absorption if contact forces are provided
        if contact_forces is not None and len(contact_forces) == len(targets):
            excess_force = np.maximum(0.0, contact_forces - nominal_support_force)
            delta_bump = -suspension_force_compliance * excess_force
        else:
            delta_bump = 0.0

        suspension_delta = np.where(is_downward, delta_skyhook + delta_bump, 0.0)
        targets = np.clip(targets + suspension_delta, min_offset, max_extend)

    # 2. Dynamic Camber Banking for Centrifugal Drift Cancellation
    if enable_camber_banking and abs(yaw_rate) > 0.01:
        turn_normal = np.array([-d_hat[1], d_hat[0]])
        lateral_proj = dirs_world[:, 0] * turn_normal[0] + dirs_world[:, 1] * turn_normal[1]
        bank_offset = camber_bank_gain * lateral_proj * np.clip(yaw_rate, -2.0, 2.0)
        targets = np.clip(targets + bank_offset, min_offset, max_extend)

    # 3. Gyroscopic Precession Damping (Anti-Wobble during Turns)
    if enable_gyroscopic_damping and ang_vel is not None:
        # tau_gyro = omega_yaw x L_roll -> Counteract lateral precession roll
        w_yaw = float(ang_vel[2])
        w_roll = float(ang_vel[0] * d_hat[0] + ang_vel[1] * d_hat[1])
        gyro_precession = w_yaw * w_roll
        if abs(gyro_precession) > 0.05:
            roll_normal = np.array([-d_hat[1], d_hat[0]])
            roll_proj = dirs_world[:, 0] * roll_normal[0] + dirs_world[:, 1] * roll_normal[1]
            gyro_offset = gyroscopic_damping_gain * roll_proj * np.clip(gyro_precession, -4.0, 4.0)
            targets = np.clip(targets + gyro_offset, min_offset, max_extend)

    # 4. Active Flank Retraction (Narrow envelope on wall side)
    if enable_flank_retraction and lidar_ranges is not None and len(lidar_ranges) > 0:
        n_rays = len(lidar_ranges)
        angles_rel = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        d_angle = np.arctan2(d_hat[1], d_hat[0])
        angles_world = (d_angle + angles_rel) % (2 * np.pi)

        for k in range(len(targets)):
            ux, uy, uz = dirs_world[k]
            if abs(uz) < 0.70:  # Sideways / flank rod
                rod_angle = np.arctan2(uy, ux) % (2 * np.pi)
                diffs = np.abs((angles_world - rod_angle + np.pi) % (2 * np.pi) - np.pi)
                closest_ray = int(np.argmin(diffs))
                d_wall = float(lidar_ranges[closest_ray]) * lidar_max_range
                if d_wall < flank_retract_dist:
                    targets[k] = min(targets[k], flank_min_offset)

    # 5. Contact Force Compliance (Admittance load relief)
    if enable_contact_compliance and contact_forces is not None:
        excess_force = np.maximum(0.0, contact_forces - max_contact_force)
        force_relief = compliance_gain * excess_force
        targets = np.clip(targets - force_relief, min_offset, max_extend)

    # 6. Anti-Stall Reflex (High-frequency stiction breaker)
    if enable_anti_stall_reflex and forward_vel < anti_stall_speed_threshold and abs(drive) > 0.4:
        pulse = anti_stall_pulse_amp * np.sin(2 * np.pi * anti_stall_pulse_freq * sim_time)
        is_pushing = (rear > 0.3) & (down_bias > 0.4)
        targets[is_pushing] = np.clip(targets[is_pushing] + pulse, min_offset, max_extend)

    # 7. Actuator Slew-Rate Limiter (Jerk-Free S-Smoothing)
    if enable_actuator_slew_rate and last_targets is not None:
        max_delta = actuator_max_vel * actuator_dt
        targets = np.clip(targets, last_targets - max_delta, last_targets + max_delta)

    return targets


def standing_jump_targets(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    *,
    phase: str = "stand",  # "stand", "crouch", "takeoff", "airborne", "landing"
    landing_standoff: float = 0.055,
    stand_height_offset: float = 0.045,
) -> np.ndarray:
    """Compute 60-bar radial extension targets for explosive standing vertical jump.

    Phases:
    1. 'stand': Stable resting posture on ground contact cluster (z ~ 0.21m).
    2. 'crouch': Deep retraction of bottom rods to store travel stroke (z ~ 0.16m).
    3. 'takeoff': Simultaneous 100% full-stroke impulse on all downward rods (+3.3 m/s launch velocity).
    4. 'airborne': Mid-air tuck holding spherical profile (+45-50 cm net vertical launch clearance).
    5. 'landing': Compliant touchdown suspension to absorb landing impact.
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T
    u_z = dirs_world[:, 2]

    targets = np.zeros(len(dirs_body), dtype=np.float32)

    if phase == "crouch":
        # Full retraction to store kinematic stroke
        targets[:] = 0.00
    elif phase == "takeoff":
        # Maximum explosive impulse across all downward-facing ground rods
        ground_mask = (u_z < 0.10)
        targets[ground_mask] = max_extend
        targets[u_z > 0.15] = 0.0
    elif phase == "airborne":
        # Mid-air aerodynamic profile
        targets[:] = 0.015
    elif phase == "landing":
        # Compliant touchdown damper
        bottom_mask = (u_z < -0.20)
        targets[bottom_mask] = landing_standoff
    return targets


def forward_jump_targets(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    phase: str = "stand",  # "stand", "crouch", "takeoff", "airborne", "landing"
    landing_standoff: float = 0.055,
    stand_height_offset: float = 0.045,
    rollout_gain: float = 0.12,
) -> np.ndarray:
    """Compute 60-bar radial extension targets for explosive directional forward jump / hurdle leap.

    Phases:
    1. 'stand': Stable resting posture on ground contact cluster (z ~ 0.21m).
    2. 'crouch': Deep retraction of bottom rods to store travel stroke (z ~ 0.16m).
    3. 'takeoff': Rear-biased 100% full-stroke impulse on ground rods (+2.5 m/s vx, +2.6 m/s vz launch).
    4. 'airborne': Mid-air tuck holding spherical profile (+35-45 cm net clearance, > 1.1m flight distance).
    5. 'landing': Compliant touchdown suspension to absorb landing impact and roll forward.
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
    u_z = dirs_world[:, 2]

    targets = np.zeros(len(dirs_body), dtype=np.float32)

    if phase == "crouch":
        # Full retraction to store kinematic stroke
        targets[:] = 0.00
    elif phase == "takeoff":
        # Directional forward impulse: rear-downward rods fire with 100% stroke, front rods retract
        ground_mask = (u_z < 0.10)
        forward_bias = np.clip(1.0 - 0.85 * np.maximum(u_long, -0.3), 0.35, 1.0)
        targets[ground_mask] = max_extend * forward_bias[ground_mask]
        targets[u_long > 0.15] = 0.0
        targets[u_z > 0.15] = 0.0
    elif phase == "airborne":
        # Mid-air aerodynamic profile
        targets[:] = 0.015
    elif phase == "landing":
        # Compliant touchdown damper with forward rollout torque
        bottom_mask = (u_z < -0.20)
        targets[bottom_mask] = landing_standoff
        rear_pusher = (u_long < -0.15) & (u_z < 0.0)
        targets[rear_pusher] = rollout_gain
        targets[u_long > 0.0] = 0.0
    else:
        # Stationary standing rest
        bottom_mask = (u_z < -0.30)
        targets[bottom_mask] = stand_height_offset

    return targets


