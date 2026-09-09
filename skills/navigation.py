"""Skills that compose the primitives to get somewhere.

``follow_path`` tracks a line of waypoints. ``stay_in_boundary`` roams a
wall-less circle and turns itself inward at the edge. Both pick a
sub-skill each step and delegate, so they belong beside the primitives
rather than among them.
"""

from __future__ import annotations

import numpy as np

from radial_sphere.geometry import quat_to_rotmat

from .locomotion import (MAX_SPEED, MIN_SPEED, _rotate, curve, move, stop,
                         turn)
from .terrain_following import traverse_rough_terrain
from .suspension import SuspensionGains, SuspensionState, apply_suspension



# ---------------------------------------------------------------------------
# 15. follow_path (closed-loop ground path tracking via skill orchestration)
# ---------------------------------------------------------------------------

def follow_path(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    *,
    ball_xy: np.ndarray,
    path_pts: np.ndarray | list,
    lin_vel: np.ndarray | None = None,
    lookahead: float = 0.85,
    speed: float = 1.2,
    goal_tolerance: float = 0.35,
    turn_angle_threshold_deg: float = 28.0,
    curve_threshold: float = 0.18,
    curvature_brake_gain: float = 1.6,
    return_metadata: bool = False,
    rod_mechanism: str | None = None,
    curve_rod_mechanism: str | None = "multi_stage",
    min_offset: float = 0.025,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Autonomous closed-loop ground path tracking via skill orchestration.

    Perceives a ground path (represented as 2D waypoints with visual ground
    markings), computes Frenet-frame tracking errors, and maneuvers by
    dynamically delegating to other existing primitives:
      * `stop`: when approaching and securing the terminal goal waypoint;
      * `turn`: when negotiating sharp angular corners (> turn_angle_threshold_deg);
      * `curve`: when carving smooth continuous curved roads and arcs (|kappa| > curve_threshold);
      * `move`: for nominal along-track cruising with closed-loop cross-track correction.

    Parameters
    ----------
    quat : (4,) body orientation quaternion [w, x, y, z].
    dirs_body : (60, 3) body-frame rod unit vectors.
    max_extend : maximum rod extension stroke (m).
    ball_xy : (2,) current world (x, y) coordinates of the robot center.
    path_pts : (N, 2) sequence of ground path waypoints.
    lin_vel : (2,) or (3,) current linear velocity vector in world frame.
    lookahead : lookahead pursuit distance along path arc (default 0.85 m).
    speed : nominal cruising speed in m/s (default 1.2 m/s).
    goal_tolerance : distance threshold to terminal waypoint to trigger stopping (default 0.35 m).
    turn_angle_threshold_deg : heading deflection threshold to dispatch `turn` (default 28 deg).
    curve_threshold : path curvature threshold in 1/m to dispatch `curve` (default 0.18 1/m).
    curvature_brake_gain : gain for curvature-adaptive deceleration into sharp bends (default 1.6).
    return_metadata : if True, returns (targets, metadata_dict).

    Returns
    -------
    targets : (60,) np.ndarray of rod target extensions.
    (optional) metadata : dict with active sub-skill, curvature, cross-track error, etc.
    """
    p = np.asarray(ball_xy, dtype=np.float64)[:2]
    pts = np.asarray(path_pts, dtype=np.float64)[:, :2]
    n_pts = len(pts)

    if n_pts < 2:
        # `stop` sets its own floor through `stance_height` and retracts
        # everything else to zero, so it takes no `min_offset`.
        res_targets = stop(quat, dirs_body, max_extend, lin_vel=lin_vel)
        if return_metadata:
            return res_targets, {
                "sub_skill": "stop", "dist_to_goal": 0.0, "cross_track_error": 0.0,
                "curvature": 0.0, "heading_error_deg": 0.0, "commanded_speed": 0.0,
            }
        return res_targets

    # 1. Goal arrival check
    goal = pts[-1]
    dist_to_goal = float(np.linalg.norm(goal - p))
    if dist_to_goal < goal_tolerance:
        res_targets = stop(quat, dirs_body, max_extend, lin_vel=lin_vel,
                           stop_distance=dist_to_goal)
        meta = {
            "sub_skill": "stop",
            "dist_to_goal": dist_to_goal,
            "cross_track_error": 0.0,
            "curvature": 0.0,
            "heading_error_deg": 0.0,
            "commanded_speed": 0.0,
        }
        return (res_targets, meta) if return_metadata else res_targets

    # 2. Find closest point on path polyline
    dists_sq = np.sum((pts - p) ** 2, axis=1)
    i_closest = int(np.argmin(dists_sq))
    
    # Segment index for local tangent
    i_seg = min(i_closest, n_pts - 2)
    p_seg_start = pts[i_seg]
    p_seg_end = pts[i_seg + 1]
    seg_vec = p_seg_end - p_seg_start
    seg_len = float(np.linalg.norm(seg_vec))
    t_hat = seg_vec / max(seg_len, 1e-6)
    n_hat = np.array([-t_hat[1], t_hat[0]])  # left-pointing normal

    # Signed cross-track error (positive = left of path, negative = right of path)
    cross_track_err = float(np.dot(p - p_seg_start, n_hat))

    # 3. Lookahead pursuit waypoint
    accum_dist = float(np.dot(p - pts[i_closest], t_hat)) if i_closest < n_pts - 1 else 0.0
    accum_dist = max(0.0, accum_dist)
    target_idx = n_pts - 1
    
    cur_accum = 0.0
    for j in range(i_closest, n_pts - 1):
        ds = float(np.linalg.norm(pts[j + 1] - pts[j]))
        if cur_accum + ds >= lookahead:
            frac = (lookahead - cur_accum) / max(ds, 1e-6)
            target_pt = pts[j] + frac * (pts[j + 1] - pts[j])
            target_idx = j + 1
            break
        cur_accum += ds
    else:
        target_pt = pts[-1]

    d_target_vec = target_pt - p
    d_target_norm = float(np.linalg.norm(d_target_vec))
    d_target_hat = d_target_vec / max(d_target_norm, 1e-6)

    # 4. Local curvature estimation across lookahead window
    i_ahead = min(target_idx, n_pts - 1)
    if i_ahead > i_seg:
        ahead_vec = pts[i_ahead] - pts[max(0, i_ahead - 1)]
        ahead_hat = ahead_vec / max(float(np.linalg.norm(ahead_vec)), 1e-6)
        dot_tan = float(np.clip(np.dot(t_hat, ahead_hat), -1.0, 1.0))
        cross_tan = float(t_hat[0] * ahead_hat[1] - t_hat[1] * ahead_hat[0])
        d_theta = float(np.arctan2(cross_tan, dot_tan))
        arc_len = max(float(np.linalg.norm(pts[i_ahead] - pts[i_seg])), 0.5)
        curvature = float(d_theta / arc_len)
    else:
        curvature = 0.0

    # 5. Curvature-adaptive speed scaling (smooth deceleration into sharp corners)
    curv_factor = 1.0 + curvature_brake_gain * (abs(curvature) ** 1.4)
    v_cmd = float(np.clip(speed / curv_factor, MIN_SPEED, MAX_SPEED))
    if dist_to_goal < 1.20:
        # Decelerate smoothly on terminal approach
        approach_scale = max(dist_to_goal / 1.20, 0.30)
        v_cmd = float(np.clip(v_cmd * approach_scale, MIN_SPEED, speed))

    # 6. Current heading and angular error to target direction
    if lin_vel is not None and np.linalg.norm(lin_vel[:2]) > 0.20:
        d_now = np.asarray(lin_vel[:2], dtype=np.float64)
        d_now /= np.linalg.norm(d_now)
    else:
        d_now = t_hat

    dot_head = float(np.clip(np.dot(d_now, d_target_hat), -1.0, 1.0))
    cross_head = float(d_now[0] * d_target_hat[1] - d_now[1] * d_target_hat[0])
    heading_err_rad = float(np.arctan2(cross_head, dot_head))
    heading_err_deg = float(np.degrees(heading_err_rad))

    # 7. Dynamic sub-skill delegation
    if abs(heading_err_deg) >= turn_angle_threshold_deg:
        # Sharp angular deflection -> delegate to `turn`
        # Turn sign in turn(): positive = right, negative = left
        angle_cmd = float(np.clip(-heading_err_deg, -50.0, 50.0))
        sub_skill = "turn"
        targets = turn(
            quat, dirs_body, max_extend, d_now,
            angle_deg=angle_cmd,
            speed=v_cmd,
            lin_vel=lin_vel,
            cross_track_error=cross_track_err,
            min_offset=min_offset,
            rod_mechanism=rod_mechanism,
        )
    elif abs(curvature) >= curve_threshold:
        # Continuous road bend -> delegate to `curve`
        sub_skill = "curve"
        targets = curve(
            quat, dirs_body, max_extend, d_target_hat,
            curvature=curvature,
            speed=v_cmd,
            ball_xy=p,
            min_offset=min_offset,
            rod_mechanism=curve_rod_mechanism,
        )
    else:
        # Nominal path tracking -> delegate to `move`
        sub_skill = "move"
        targets = move(
            quat, dirs_body, max_extend, d_target_hat,
            speed=v_cmd,
            lin_vel=lin_vel,
            cross_track_error=cross_track_err,
            min_offset=min_offset,
            rod_mechanism=rod_mechanism,
        )

    meta = {
        "sub_skill": sub_skill,
        "curvature": curvature,
        "cross_track_error": cross_track_err,
        "heading_error_deg": heading_err_deg,
        "commanded_speed": v_cmd,
        "dist_to_goal": dist_to_goal,
        "target_point": target_pt,
    }
    return (targets, meta) if return_metadata else targets



# ---------------------------------------------------------------------------
# 18. stay_in_boundary (autonomous wall-less boundary containment & slow roaming)
# ---------------------------------------------------------------------------

def stay_in_boundary(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    *,
    ball_xy: np.ndarray,
    lin_vel: np.ndarray | None = None,
    boundary_center: np.ndarray | tuple[float, float] = (0.0, 0.0),
    boundary_radius: float = 2.0,
    speed: float = 0.80,
    safety_margin: float = 0.60,
    step_count: int | None = None,
    rod_mechanism: str | None = None,
    min_offset: float = 0.025,
    core_z: float | None = None,
    core_vz: float | None = None,
    target_ride_height: float = 0.28,
    contact_forces: np.ndarray | None = None,
    terrain_clearances: np.ndarray | None = None,
    enable_suspension: bool = True,
    rough_terrain_gait: bool = False,
    rough_drive_gain: float | None = None,
    climb_drive_gain: float = 2.8,
    obstruction_steps: int = 0,
    climb_patience: int = 150,
    suspension_kp: float = 0.75,
    suspension_kd: float = 0.15,
    suspension_force_compliance: float = 0.0018,
    nominal_support_force: float = 10.0,
    terrain_adaptation_gain: float = 0.85,
    hole_reach_gain: float = 0.045,
    suspension: SuspensionGains | None = None,
    suspension_state: SuspensionState | None = None,
    control_dt: float = 0.01,
    max_target_speed: float = 0.45,
    suspension_filter_time: float = 0.06,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Confine locomotion inside a wall-less circular boundary on the floor.

    The arena has NO physical walls. The robot roams inside dynamically, executing
    diverse exploratory actions (straight forward cruising, arcing curves,
    in-place heading pivots, and balance pauses) while continuously monitoring
    its radial distance to the boundary line.

    When approaching the perimeter (distance to boundary < safety_margin):
    1. It overrides open-loop exploration with active boundary deflection.
    2. Blends current travel heading with the inward normal vector.
    3. Smoothly scales down speed to prevent overshoot.
    4. Dynamically delegates to existing primitives (`stop` for emergency brake,
       `turn` for sharp angular deflection, `curve` for tangential deflection,
       or `move` for inward cruising).

    Parameters
    ----------
    quat : (4,) body orientation quaternion [w, x, y, z].
    dirs_body : (60, 3) body-frame rod unit vectors.
    max_extend : maximum rod extension stroke in metres.
    ball_xy : (2,) world-frame position of the ball center.
    lin_vel : (2,) or (3,) world-frame linear velocity vector.
    boundary_center : (2,) center of the circular boundary (default [0, 0]).
    boundary_radius : radius of the circular boundary in metres (default 2.0 m).
    speed : commanded cruising speed inside the boundary (default 0.80 m/s).
    safety_margin : distance from the boundary perimeter where containment
        deflection activates (default 0.60 m).
    step_count : optional simulation step index used to alternate exploratory actions.
    rough_terrain_gait : drive straight travel with `traverse_rough_terrain`
        instead of `move`. The rolling `move` gait is stopped by a step of
        about 4 cm; the rough-terrain gait with curb vaulting clears 6 cm.
        Turn this on for rocky ground, where climbing over beats steering
        around. The rough gait applies its own suspension, so the shared
        correction below is not applied twice.
    obstruction_steps : how many consecutive steps the caller has seen an
        obstruction, from `meta["is_obstructed"]`. While it stays under
        `climb_patience` an obstructed robot on rough terrain tries to climb
        over. Past that it gives up and pivots away.
    rough_drive_gain : drive-wave amplitude for rough-terrain travel, used
        instead of the flat-ground speed calibration. Speed feedback throttles
        the calibrated gain by up to 98 % once the robot runs above its
        commanded speed, which starves the push exactly when the next rock
        needs it. Commanding amplitude directly avoids that. On this build
        2.0 lifts the climbable step from 6 cm to 8 cm; it cruises near
        1.2 m/s on flat ground.
    climb_drive_gain : drive-wave amplitude while climbing over an obstacle.
    climb_patience : steps spent climbing before pivoting away instead.
    suspension : a `SuspensionGains` holding the whole correction tuning, in
        place of the nine separate gain keywords, which stay accepted.
    rod_mechanism : "single_stage" or "multi_stage".
    curve_rod_mechanism : calibration for the arc phases only. The tracking
        error published for this skill was measured with the multi-stage
        curve, which asks for about 15 % less gain than the generic one, so
        that stays the default. Set it to `rod_mechanism` to run one
        calibration across every phase; that travels further per second and
        holds the line less tightly.
    min_offset : baseline retracted rod length, forwarded to the `move`,
        `curve` and `turn` branches. The `stop` branch does not take it:
        `stop` holds its stance on `stance_height` and retracts every
        other rod to zero.
    return_metadata : if True, returns (targets, meta_dict).

    Returns
    -------
    targets : (60,) np.ndarray of rod target extensions.
    meta (optional) : dictionary with sub_skill, distance_to_edge, action_name, etc.
    """
    p = np.asarray(ball_xy, dtype=np.float64)[:2]
    c = np.asarray(boundary_center, dtype=np.float64)[:2]
    rel = p - c
    r = float(np.linalg.norm(rel))
    d_edge = float(boundary_radius - r)

    if r > 1e-4:
        n_out = rel / r
        n_in = -n_out
    else:
        n_out = np.array([1.0, 0.0], dtype=np.float64)
        n_in = np.array([-1.0, 0.0], dtype=np.float64)

    # Current velocity and heading estimation
    if lin_vel is not None and len(lin_vel) >= 2:
        v_vec = np.asarray(lin_vel[:2], dtype=np.float64)
        v_mag = float(np.linalg.norm(v_vec))
    else:
        v_vec = np.zeros(2, dtype=np.float64)
        v_mag = 0.0

    v_radial = float(np.dot(v_vec, n_out))  # positive = moving towards edge

    if v_mag > 0.06:
        d_now = v_vec / v_mag
    else:
        d_now = n_in.copy()

    # Determine tangent direction along perimeter aligned with current motion
    t_left = np.array([-n_out[1], n_out[0]], dtype=np.float64)
    t_dot = float(np.dot(d_now, t_left))
    t_dir = t_left if t_dot >= 0.0 else -t_left

    gains = suspension or SuspensionGains(
        target_ride_height=target_ride_height, kp=suspension_kp, kd=suspension_kd,
        force_compliance=suspension_force_compliance,
        nominal_support_force=nominal_support_force,
        terrain_adaptation=terrain_adaptation_gain, hole_reach=hole_reach_gain,
        max_target_speed=max_target_speed, filter_time=suspension_filter_time)

    # Targets that already went through apply_suspension must not be
    # corrected a second time; the slew limiter and filter are stateful.
    suspension_applied = False

    def _rough_gait(direction, gait_speed, curb_boost_gain=2.6, drive_gain=None):
        return traverse_rough_terrain(
            quat, dirs_body, max_extend, d_hat=direction, speed=gait_speed,
            back_gain=drive_gain, min_offset=min_offset, lin_vel=lin_vel,
            core_z=core_z, core_vz=core_vz,
            contact_forces=contact_forces, terrain_clearances=terrain_clearances,
            enable_curb_vaulting=True, curb_boost_gain=curb_boost_gain,
            suspension=gains, suspension_state=suspension_state,
            control_dt=control_dt,
        )

    # Check boundary proximity
    is_in_boundary_zone = (d_edge <= safety_margin)
    is_moving_outward = (v_radial > -0.03) or (float(np.dot(d_now, n_out)) > -0.08)

    if is_in_boundary_zone and is_moving_outward:
        # -------------------------------------------------------------------
        # Active Boundary Containment Mode
        # -------------------------------------------------------------------
        action_name = "boundary_containment"
        s = float(np.clip((safety_margin - d_edge) / max(safety_margin, 1e-3), 0.0, 1.0))

        # Commanded safe speed scales down smoothly near the line
        v_contain = float(np.clip(speed * (0.25 + 0.60 * (1.0 - s)), 0.18, speed))

        # Critical proximity brake: if close to line and moving outward with momentum
        if (d_edge < 0.28 and v_radial > 0.08) or (d_edge < 0.16 and v_radial > 0.02):
            sub_skill = "stop"
            action_name = "boundary_emergency_brake"
            targets = stop(
                quat, dirs_body, max_extend,
                lin_vel=lin_vel,
                brake_gain=3.2,
                stance_height=0.045,
            )
        else:
            # Desired inward deflection heading: blend perimeter tangent and inward normal
            d_steer = (1.0 - 0.75 * s) * t_dir + (1.0 + 1.25 * s) * n_in
            d_steer_norm = float(np.linalg.norm(d_steer))
            d_cmd = d_steer / max(d_steer_norm, 1e-6)

            dot_cmd = float(np.clip(np.dot(d_now, d_cmd), -1.0, 1.0))
            cross_cmd = float(d_now[0] * d_cmd[1] - d_now[1] * d_cmd[0])
            heading_err_deg = float(np.degrees(np.arctan2(cross_cmd, dot_cmd)))

            if abs(heading_err_deg) > 28.0:
                # Sharp deflection required -> delegate to `turn`
                sub_skill = "turn"
                action_name = "boundary_turn_inward"
                turn_deg = float(np.clip(-heading_err_deg, -55.0, 55.0))
                targets = turn(
                    quat, dirs_body, max_extend, d_now,
                    angle_deg=turn_deg,
                    speed=v_contain,
                    lin_vel=lin_vel,
                    rod_mechanism=rod_mechanism,
                )
            elif abs(t_dot) > 0.35 and d_edge > 0.20:
                # Smooth perimeter arcing -> delegate to `curve`
                sub_skill = "curve"
                action_name = "boundary_arc_away"
                curve_dir = "left" if t_dot >= 0.0 else "right"
                targets = curve(
                    quat, dirs_body, max_extend, d_now,
                    radius=max(0.8, boundary_radius * 0.70),
                    speed=v_contain,
                    direction=curve_dir,
                    ball_xy=p,
                    center_xy=c,
                    rod_mechanism=rod_mechanism,
                )
            else:
                # General inward cruising -> delegate to `move`
                sub_skill = "move"
                action_name = "boundary_inward_drive"
                targets = move(
                    quat, dirs_body, max_extend, d_cmd,
                    speed=v_contain,
                    lin_vel=lin_vel,
                    rod_mechanism=rod_mechanism,
                )
    else:
        # -------------------------------------------------------------------
        # Exploratory Slow Roaming Mode (Diverse Actions Inside Boundary)
        # -------------------------------------------------------------------
        step_idx = 0 if step_count is None else int(step_count)
        # 360-step cycle (~3.6s) divided into 5 distinct behavioral phases
        phase_time = step_idx % 360

        # Obstruction reflex: if forward motion stalls against a rock or depression, immediately pivot away
        is_obstructed = (v_mag < 0.04) and (step_idx > 30) and (phase_time < 225 or phase_time >= 255)

        if is_obstructed and rough_terrain_gait and obstruction_steps < climb_patience:
            # Climbing over beats steering around. Curb vaulting boosts the
            # rear pushers against the face of the rock.
            sub_skill = "traverse_rough_terrain"
            action_name = "roam_climb_over"
            targets = _rough_gait(d_now, speed * 1.15, curb_boost_gain=3.2,
                                  drive_gain=climb_drive_gain)
            suspension_applied = True
        elif is_obstructed:
            sub_skill = "turn"
            action_name = "roam_obstacle_escape"
            escape_angle = 65.0 if (step_idx // 40) % 2 == 0 else -65.0
            targets = turn(
                quat, dirs_body, max_extend, d_now,
                angle_deg=escape_angle,
                speed=speed * 1.05,
                lin_vel=lin_vel,
                rod_mechanism=rod_mechanism,
            )
        elif phase_time < 90:
            # Action Phase 1: Calm Forward Cruise
            sub_skill = "move"
            action_name = "roam_forward_cruise"
            # Command gentle roaming heading modulated over long cycles
            cycle_num = step_idx // 360
            angle_bias = float((cycle_num * 1.35) % (2 * np.pi))
            d_roam = np.array([np.cos(angle_bias), np.sin(angle_bias)], dtype=np.float64)
            if rough_terrain_gait:
                sub_skill = "traverse_rough_terrain"
                targets = _rough_gait(d_roam, speed, drive_gain=rough_drive_gain)
                suspension_applied = True
            else:
                targets = move(
                    quat, dirs_body, max_extend, d_roam,
                    speed=speed,
                    lin_vel=lin_vel,
                    rod_mechanism=rod_mechanism,
                )
        elif phase_time < 170:
            # Action Phase 2: Gentle Sweeping Curve Left
            action_name = "roam_curve_left"
            if rough_terrain_gait:
                # Arc by steering the rough gait, so the climbing drive stays
                # available. `curve` cannot cross a rock.
                sub_skill = "traverse_rough_terrain"
                targets = _rough_gait(_rotate(d_now, np.deg2rad(25.0)),
                                      speed * 0.95, drive_gain=rough_drive_gain)
                suspension_applied = True
            else:
                sub_skill = "curve"
                targets = curve(
                    quat, dirs_body, max_extend, d_now,
                    radius=1.85,
                    speed=speed * 0.95,
                    direction="left",
                    rod_mechanism=rod_mechanism,
                )
        elif phase_time < 225:
            # Action Phase 3: Deliberate In-Place Heading Pivot / Turn
            sub_skill = "turn"
            action_name = "roam_pivot_turn"
            # Steer 35 deg right relative to current direction
            targets = turn(
                quat, dirs_body, max_extend, d_now,
                angle_deg=35.0,
                speed=speed * 0.85,
                lin_vel=lin_vel,
                rod_mechanism=rod_mechanism,
            )
        elif phase_time < 255:
            # Action Phase 4: Stance Hold / Momentary Balance Pause
            sub_skill = "stop"
            action_name = "roam_stance_pause"
            targets = stop(
                quat, dirs_body, max_extend,
                lin_vel=lin_vel,
                stance_height=0.045,
                brake_gain=1.2,
            )
        else:
            # Action Phase 5: Sweeping Curve Right / S-turn exploration
            action_name = "roam_curve_right"
            if rough_terrain_gait:
                sub_skill = "traverse_rough_terrain"
                targets = _rough_gait(_rotate(d_now, np.deg2rad(-25.0)),
                                      speed * 0.95, drive_gain=rough_drive_gain)
                suspension_applied = True
            else:
                sub_skill = "curve"
                targets = curve(
                    quat, dirs_body, max_extend, d_now,
                    radius=2.10,
                    speed=speed * 0.95,
                    direction="right",
                    rod_mechanism=rod_mechanism,
                )

    suspension_meta = {}
    if (enable_suspension and not suspension_applied
            and (core_z is not None or terrain_clearances is not None
                 or contact_forces is not None)):
        dirs_world = dirs_body @ quat_to_rotmat(quat).T
        u_z = dirs_world[:, 2]
        # Same trailing-rod gate as traverse_rough_terrain, measured along the
        # heading this step is actually travelling.
        u_long_b = dirs_world[:, 0] * d_now[0] + dirs_world[:, 1] * d_now[1]
        support_b = np.clip((-u_long_b - 0.05) / 0.35, 0.0, 1.0)
        support_b = support_b * support_b * (3.0 - 2.0 * support_b)
        vz = core_vz if core_vz is not None else (float(lin_vel[2]) if lin_vel is not None and len(lin_vel) > 2 else 0.0)
        targets, suspension_meta = apply_suspension(
            targets, u_z, max_extend, core_z=core_z, core_vz=vz,
            min_offset=min_offset, contact_forces=contact_forces,
            terrain_clearances=terrain_clearances, support_weight=support_b,
            state=suspension_state, dt=control_dt, **gains.as_kwargs(),
        )

    meta = {
        **suspension_meta,
        "sub_skill": sub_skill,
        "action_name": action_name,
        "radius": r,
        "distance_to_edge": d_edge,
        "v_radial": v_radial,
        "is_boundary_active": is_in_boundary_zone and is_moving_outward,
        "is_obstructed": bool(action_name in ("roam_climb_over", "roam_obstacle_escape")),
        "speed_mag": v_mag,
    }

    return (targets.astype(np.float32), meta) if return_metadata else targets.astype(np.float32)
