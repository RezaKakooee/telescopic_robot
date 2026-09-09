"""Direct MuJoCo MJCF Scene and Robot Builder.

Generates a complete, self-contained MuJoCo XML scene string containing the arena,
floor, maze walls, obstacles, goal marker, lighting, cameras, and the 60-bar radial
sphere robot with physical slide joints, contact feet, and position actuators.
"""
from __future__ import annotations

import colorsys
import numpy as np

from .geometry import fibonacci_sphere
from .mjcf import SLEEVE_STUB, TIP_GAP, FOOT_RADIUS, rolling_radius


from . import mjcf_features as features


def _decompose_floor_slabs(x_range: tuple[float, float], y_range: tuple[float, float], holes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Partition a 2D bounding box into non-overlapping rectangular slabs that exclude the given holes."""
    x0, x1 = x_range
    y0, y1 = y_range
    x_cuts = sorted(set([x0, x1] + [h[0] for h in holes] + [h[1] for h in holes]))
    slabs = []
    for i in range(len(x_cuts) - 1):
        xa, xb = x_cuts[i], x_cuts[i + 1]
        if xb - xa < 1e-5:
            continue
        xm = (xa + xb) / 2.0
        active_holes = [h for h in holes if h[0] <= xm <= h[1]]
        if not active_holes:
            slabs.append((xa, xb, y0, y1))
        else:
            y_cuts = sorted(set([y0, y1] + [h[2] for h in active_holes] + [h[3] for h in active_holes]))
            for j in range(len(y_cuts) - 1):
                ya, yb = y_cuts[j], y_cuts[j + 1]
                if yb - ya < 1e-5:
                    continue
                ym = (ya + yb) / 2.0
                is_hole = any(h[2] <= ym <= h[3] for h in active_holes)
                if not is_hole:
                    slabs.append((xa, xb, ya, yb))
    return slabs


def build_mujoco_scene_mjcf(
    scenario,
    n_bars: int = 60,
    sphere_radius: float = 0.15,
    max_extend: float = 0.12,
    core_mass: float = 0.5,
    wall_height: float = 0.22,
    wall_thickness: float = 0.06,
    sleeve_radius: float = 0.012,
    inner_radius: float = 0.008,
    bar_length: float | None = None,
    timestep: float = 0.002,
    sim2real_cfg: dict | None = None,
    appearance_theme: str = "rainbow",  # "realistic", "aerospace_white", "rainbow"
    rod_mechanism: str = "single_stage",  # "single_stage", "multi_stage", "zip_chain"
    kp: float = 1200.0,
    kv: float = 22.0,
    floor_half_extent: float = 200.0,
    floor_square_m: float = 0.4,
    floor_rgb1: str = "0.42 0.45 0.51",
    floor_rgb2: str = "0.20 0.23 0.28",
) -> tuple[str, np.ndarray]:
    """Build a complete MuJoCo XML containing arena, walls, goal, and robot.

    Args:
        scenario: Scenario object with spawn_xy, goal, walls, markers.
        n_bars: Number of radial telescoping bars (Fibonacci distributed).
        sphere_radius: Central ball radius in metres.
        max_extend: Maximum stroke of each telescoping rod.
        core_mass: Mass of the central sphere.
        wall_height: Height of the maze walls in metres.
        wall_thickness: Thickness of the maze walls.
        sleeve_radius: Outer guide sleeve radius.
        inner_radius: Inner sliding rod radius.
        bar_length: Rod length inside sleeve.
        timestep: Physics integration step (seconds).
        enable_sim2real: If True, activates physical hardware damping, frictionloss, solref, and force caps.
        appearance_theme: Visual material theme ("realistic", "aerospace_white", "rainbow").

    Returns:
        xml_str: Complete MJCF XML string ready for mujoco.MjModel.from_xml_string().
        dirs: (n_bars, 3) unit direction vectors for each bar (body frame).
    """
    dirs = fibonacci_sphere(n_bars)
    tip0 = sphere_radius + SLEEVE_STUB + TIP_GAP
    sleeve_mouth = sphere_radius + SLEEVE_STUB
    if bar_length is None:
        bar_length = max_extend + TIP_GAP + 0.35 * sphere_radius

    # 1. Build Robot Bars and Actuators
    bars_xml: list[str] = []
    actuators_xml: list[str] = []
    equalities_xml: list[str] = []

    # Theme colors
    is_realistic = appearance_theme in ["realistic", "carbon_gunmetal"]
    is_white = appearance_theme == "aerospace_white"

    # Physics parameters (High-Standard Sim-to-Real Benchmark)
    s2r = sim2real_cfg or {}
    enable_sim2real = bool(s2r.get("enabled", False))

    joint_damping = float(s2r.get("joint_damping", 0.35 if enable_sim2real else 0.20))
    joint_frictionloss = float(s2r.get("joint_frictionloss", 0.08 if enable_sim2real else 0.04))

    f_sl = float(s2r.get("rubber_friction_sliding", 0.95))
    f_t = float(s2r.get("rubber_friction_torsional", 0.015))
    f_r = float(s2r.get("rubber_friction_rolling", 0.005))
    foot_friction = f"{f_sl} {f_t} {f_r}"

    sr_time = float(s2r.get("rubber_solref_timeconst", 0.006))
    sr_damp = float(s2r.get("rubber_solref_dampratio", 1.10))
    foot_solref = f"{sr_time} {sr_damp}"

    foot_solimp = "0.90 0.95 0.002"

    max_f = float(s2r.get("actuator_force_limit", 120.0))
    actuator_forcerange = f"{-max_f} {max_f}"

    kp_act = kp
    kv_act = kv

    sensors_xml: list[str] = [
        '    <accelerometer name="imu_acc" site="imu_site"/>\n',
        '    <gyro name="imu_gyro" site="imu_site"/>\n',
        '    <framequat name="imu_quat" objtype="site" objname="imu_site"/>\n',
    ]

    for k, (ux, uy, uz) in enumerate(dirs):
        u = np.array([ux, uy, uz], dtype=float)
        u_unit = u / (np.linalg.norm(u) + 1e-12)

        if is_realistic:
            sleeve_rgba = "0.38 0.42 0.48 1"
            rod_rgba = "0.88 0.90 0.94 1"
            foot_rgba = "0.10 0.10 0.12 1"  # Molded Black Vulcanized Rubber
        elif is_white:
            sleeve_rgba = "0.20 0.45 0.75 1"
            rod_rgba = "0.90 0.92 0.95 1"
            foot_rgba = "0.12 0.12 0.14 1"
        else:
            rr, gg, bb = colorsys.hsv_to_rgb(k / n_bars, 0.90, 1.00)
            fr, fg, fb = colorsys.hsv_to_rgb(k / n_bars, 0.90, 0.65)
            sleeve_rgba = "1.0 0.82 0.15 1"
            rod_rgba = f"{rr:.3f} {gg:.3f} {bb:.3f} 1"
            foot_rgba = f"{fr:.3f} {fg:.3f} {fb:.3f} 1"

        if rod_mechanism in ["multi_stage", "concentric_telescopic"]:
            # 1. Multi-Stage Concentric Telescopic Nesting:
            # Fixed outer sleeve: [0.074m, 0.156m]
            # Intermediate collar: slides at 0.5 * extension, spans [0.072m, 0.160m]
            # Inner rod: slides at 1.0 * extension, spans [0.076m, 0.160m]
            # Overlap at full stroke (e=0.16m):
            #   sleeve mouth (0.156) > stage1 base (0.152) -> +4mm overlap
            #   stage1 tip (0.240) > inner base (0.236) -> +4mm overlap
            # ZERO DISCONTINUITY AT ALL EXTENSIONS, and central hub r < 7.2cm is 100% CLEAR!
            r_base = 0.493 * sphere_radius  # 0.074m
            sleeve_from = r_base * u_unit
            sleeve_to = sleeve_mouth * u_unit

            st1_p1 = (r_base - 0.002) * u_unit
            st1_p2 = tip0 * u_unit

            st2_p1 = (r_base + 0.002) * u_unit
            st2_p2 = (tip0 - FOOT_RADIUS * 0.9) * u_unit

            bars_xml.append(
                f"""
                <!-- Fixed Outer Sleeve Guide (Mounted to Shell) -->
                <geom name="sleeve_{k}" type="capsule"
                      fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                              {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"
                      size="{sleeve_radius * 1.10:.5f}" rgba="{sleeve_rgba}" mass="0.004"
                      contype="0" conaffinity="0"/>
                
                <!-- Intermediate Nested Stage Body -->
                <body name="stage1_{k}" pos="0 0 0">
                    <joint name="slide1_{k}" type="slide"
                           axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                           range="0 {max_extend * 0.5}" armature="0.001"
                           damping="{float(joint_damping)*0.5:.3f}" frictionloss="{float(joint_frictionloss)*0.5:.3f}"
                           margin="0.001" solreflimit="0.005 1.0" solimplimit="0.90 0.98 0.001"/>
                    <geom name="stage1_geom_{k}" type="capsule"
                          fromto="{st1_p1[0]:.5f} {st1_p1[1]:.5f} {st1_p1[2]:.5f}
                                  {st1_p2[0]:.5f} {st1_p2[1]:.5f} {st1_p2[2]:.5f}"
                          size="{sleeve_radius * 0.85:.5f}" rgba="{sleeve_rgba}" mass="0.003"
                          contype="0" conaffinity="0"/>
                </body>

                <!-- Inner Piston Stage (Connected to Actuator) -->
                <body name="inner_{k}" pos="0 0 0">
                    <joint name="slide_{k}" type="slide"
                           axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                           range="0 {max_extend}" armature="0.002"
                           damping="{joint_damping}" frictionloss="{joint_frictionloss}"
                           margin="0.001" solreflimit="0.005 1.0" solimplimit="0.90 0.98 0.001"/>
                    <geom name="inner_geom_{k}" type="capsule"
                          fromto="{st2_p1[0]:.5f} {st2_p1[1]:.5f} {st2_p1[2]:.5f}
                                  {st2_p2[0]:.5f} {st2_p2[1]:.5f} {st2_p2[2]:.5f}"
                          size="{inner_radius * 0.95:.5f}" rgba="{rod_rgba}" mass="0.004"
                          contype="0" conaffinity="0"/>
                    <geom name="foot_{k}" type="sphere"
                          pos="{tip0 * ux:.5f} {tip0 * uy:.5f} {tip0 * uz:.5f}"
                          size="{FOOT_RADIUS}" rgba="{foot_rgba}" mass="0.004"
                          contype="1" conaffinity="2" friction="{foot_friction}" condim="4" priority="1"
                          solref="{foot_solref}" solimp="{foot_solimp}"/>
                    <site name="foot_site_{k}" pos="{tip0 * ux:.5f} {tip0 * uy:.5f} {tip0 * uz:.5f}"
                          size="0.008" type="sphere" rgba="0 0 0 0"/>
                </body>
                """
            )
            equalities_xml.append(
                f'<joint joint1="slide1_{k}" joint2="slide_{k}" polycoef="0 0.5 0 0 0" solref="0.004 1.0" solimp="0.95 0.99 0.001"/>'
            )

        elif rod_mechanism in ["zip_chain", "push_chain"]:
            # 2. Interlocking Zip-Chain / Push-Chain Spool Drive:
            # Compact peripheral nozzle at shell wall with tangential chain magazine spool.
            # Chain column continuously extends from nozzle out to foot via coupled interlocking links.
            # ZERO DISCONTINUITY AT ALL EXTENSIONS, and central hub is 100% CLEAR!
            r_base = 0.493 * sphere_radius  # 0.074m
            sleeve_from = r_base * u_unit
            sleeve_to = sleeve_mouth * u_unit

            up = np.array([0, 0, 1.0]) if abs(uz) < 0.9 else np.array([1.0, 0, 0])
            tangent = np.cross(u_unit, up)
            tangent /= (np.linalg.norm(tangent) + 1e-12)
            c_p1 = (sphere_radius * 0.85) * u_unit
            c_p2 = c_p1 + 0.038 * tangent

            st1_p1 = (r_base - 0.002) * u_unit
            st1_p2 = tip0 * u_unit

            st2_p1 = (r_base + 0.002) * u_unit
            st2_p2 = (tip0 - FOOT_RADIUS * 0.9) * u_unit

            chain1_rgba = "0.70 0.73 0.80 1"

            bars_xml.append(
                f"""
                <!-- Compact Peripheral Nozzle (Mounted at Shell Wall) -->
                <geom name="sleeve_{k}" type="capsule"
                      fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                              {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"
                      size="{sleeve_radius * 1.15:.5f}" rgba="{sleeve_rgba}" mass="0.003"
                      contype="0" conaffinity="0"/>
                <!-- Tangential Flexible Chain Spool / Magazine Housing -->
                <geom name="cassette_{k}" type="capsule"
                      fromto="{c_p1[0]:.5f} {c_p1[1]:.5f} {c_p1[2]:.5f}
                              {c_p2[0]:.5f} {c_p2[1]:.5f} {c_p2[2]:.5f}"
                      size="{sleeve_radius * 0.92:.5f}" rgba="0.45 0.48 0.55 1" mass="0.004"
                      contype="0" conaffinity="0"/>
                
                <!-- Interlocking Push-Chain Base Column (Emerges at 0.5*e) -->
                <body name="stage1_{k}" pos="0 0 0">
                    <joint name="slide1_{k}" type="slide"
                           axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                           range="0 {max_extend * 0.5}" armature="0.001"
                           damping="{float(joint_damping)*0.5:.3f}" frictionloss="{float(joint_frictionloss)*0.5:.3f}"
                           margin="0.001" solreflimit="0.005 1.0" solimplimit="0.90 0.98 0.001"/>
                    <geom name="stage1_geom_{k}" type="capsule"
                          fromto="{st1_p1[0]:.5f} {st1_p1[1]:.5f} {st1_p1[2]:.5f}
                                  {st1_p2[0]:.5f} {st1_p2[1]:.5f} {st1_p2[2]:.5f}"
                          size="{sleeve_radius * 0.85:.5f}" rgba="{chain1_rgba}" mass="0.003"
                          contype="0" conaffinity="0"/>
                </body>

                <!-- Interlocking Push-Chain Tip Column (Reaches foot at 1.0*e) -->
                <body name="inner_{k}" pos="0 0 0">
                    <joint name="slide_{k}" type="slide"
                           axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                           range="0 {max_extend}" armature="0.002"
                           damping="{joint_damping}" frictionloss="{joint_frictionloss}"
                           margin="0.001" solreflimit="0.005 1.0" solimplimit="0.90 0.98 0.001"/>
                    <geom name="inner_geom_{k}" type="capsule"
                          fromto="{st2_p1[0]:.5f} {st2_p1[1]:.5f} {st2_p1[2]:.5f}
                                  {st2_p2[0]:.5f} {st2_p2[1]:.5f} {st2_p2[2]:.5f}"
                          size="{inner_radius * 1.05:.5f}" rgba="{rod_rgba}" mass="0.007"
                          contype="0" conaffinity="0"/>
                    <geom name="foot_{k}" type="sphere"
                          pos="{tip0 * ux:.5f} {tip0 * uy:.5f} {tip0 * uz:.5f}"
                          size="{FOOT_RADIUS}" rgba="{foot_rgba}" mass="0.004"
                          contype="1" conaffinity="2" friction="{foot_friction}" condim="4" priority="1"
                          solref="{foot_solref}" solimp="{foot_solimp}"/>
                    <site name="foot_site_{k}" pos="{tip0 * ux:.5f} {tip0 * uy:.5f} {tip0 * uz:.5f}"
                          size="0.008" type="sphere" rgba="0 0 0 0"/>
                </body>
                """
            )
            equalities_xml.append(
                f'<joint joint1="slide1_{k}" joint2="slide_{k}" polycoef="0 0.5 0 0 0" solref="0.004 1.0" solimp="0.95 0.99 0.001"/>'
            )

        else:
            # 3. Baseline Single-Stage Rigid Rod (pokes through center when retracted)
            sleeve_from = (0.55 * sphere_radius) * u_unit
            sleeve_to = sleeve_mouth * u_unit
            rod_to = (tip0 - FOOT_RADIUS * 0.9) * u_unit
            rod_from = (tip0 - bar_length) * u_unit
            foot = tip0 * u_unit

            bars_xml.append(
                f"""
                <geom name="sleeve_{k}" type="capsule"
                      fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                              {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"
                      size="{sleeve_radius}" rgba="{sleeve_rgba}" mass="0.005"
                      contype="0" conaffinity="0"/>
                <body name="inner_{k}" pos="0 0 0">
                    <joint name="slide_{k}" type="slide"
                           axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                           range="0 {max_extend}" armature="0.002"
                           damping="{joint_damping}" frictionloss="{joint_frictionloss}"
                           margin="0.001" solreflimit="0.005 1.0" solimplimit="0.90 0.98 0.001"/>
                    <geom name="inner_geom_{k}" type="capsule"
                          fromto="{rod_from[0]:.5f} {rod_from[1]:.5f} {rod_from[2]:.5f}
                                  {rod_to[0]:.5f}   {rod_to[1]:.5f}   {rod_to[2]:.5f}"
                          size="{inner_radius}" rgba="{rod_rgba}" mass="0.008"
                          contype="0" conaffinity="0"/>
                    <geom name="foot_{k}" type="sphere"
                          pos="{foot[0]:.5f} {foot[1]:.5f} {foot[2]:.5f}"
                          size="{FOOT_RADIUS}" rgba="{foot_rgba}" mass="0.004"
                          contype="1" conaffinity="2" friction="{foot_friction}" condim="4" priority="1"
                          solref="{foot_solref}" solimp="{foot_solimp}"/>
                    <site name="foot_site_{k}" pos="{foot[0]:.5f} {foot[1]:.5f} {foot[2]:.5f}"
                          size="0.008" type="sphere" rgba="0 0 0 0"/>
                </body>
                """
            )

        # Sensors for bar k (Menagerie Standard)
        sensors_xml.append(f'    <jointpos name="pos_{k}" joint="slide_{k}"/>\n')
        sensors_xml.append(f'    <jointvel name="vel_{k}" joint="slide_{k}"/>\n')
        sensors_xml.append(f'    <actuatorfrc name="frc_{k}" actuator="slide_{k}"/>\n')
        sensors_xml.append(f'    <touch name="touch_{k}" site="foot_site_{k}"/>\n')

        actuators_xml.append(
            f'<general name="slide_{k}" joint="slide_{k}" '
            f'gainprm="{kp_act} 0 0" biasprm="0 -{kp_act} -{kv_act}" biastype="affine" gaintype="fixed" '
            f'ctrlrange="0 {max_extend}" forcerange="{actuator_forcerange}"/>'
        )

    # 2. Spawn Position
    spawn_xy = np.asarray(scenario.spawn_xy, dtype=float)[:2]
    spawn_z = rolling_radius(sphere_radius, 0.15 * max_extend) + 0.005

    # 3. Maze Walls Geometry (Flat, Curved Arcs, and Banked Walls)
    walls_xml: list[str] = []
    half_th = wall_thickness / 2.0
    half_h = wall_height / 2.0
    walls = np.asarray(scenario.walls, dtype=float).reshape(-1, 4)
    bank_roll = float(getattr(scenario, "wall_bank_deg", 0.0))
    for idx, (x1, y1, x2, y2) in enumerate(walls):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        yaw_deg = float(np.degrees(np.arctan2(dy, dx)))
        # Pad length slightly so segments tile and overlap cleanly without cracks
        sx = max(length / 2.0 + half_th, half_th)
        sy = half_th
        walls_xml.append(
            f'<geom name="wall_{idx}" type="box" pos="{cx:.4f} {cy:.4f} {half_h:.4f}" '
            f'euler="{bank_roll:.2f} 0 {yaw_deg:.2f}" '
            f'size="{sx:.4f} {sy:.4f} {half_h:.4f}" material="wall_mat" '
            f'friction="1.60 0.005 0.0001" condim="3"/>'
        )

    # 4. Obstacle Pillars / Realistic Industrial Blockers (if any)
    obs_raw = getattr(scenario, "obstacles", None)
    if obs_raw is None or len(obs_raw) == 0:
        obs_raw = getattr(scenario, "pillars", None)

    if obs_raw is not None and len(obs_raw) > 0:
        obs_items = np.asarray(obs_raw, dtype=float)
        if obs_items.ndim == 1:
            obs_items = obs_items.reshape(1, -1)
        for p_idx, p_item in enumerate(obs_items):
            if len(p_item) == 3:
                # Cylindrical Industrial Safety Bollard with Hazard Reflective Collar
                px, py, pr = float(p_item[0]), float(p_item[1]), float(p_item[2])
                bh = max(wall_height * 1.15, 0.26)
                base_h = 0.02
                collar_h = 0.045
                collar_z = bh * 0.65

                # 4a. Base mounting flange ring
                walls_xml.append(
                    f'<geom name="bollard_base_{p_idx}" type="cylinder" pos="{px:.4f} {py:.4f} {base_h / 2:.4f}" '
                    f'size="{pr * 1.16:.4f} {base_h / 2:.4f}" material="bollard_base_mat" '
                    f'friction="0.9 0.01 0.001" condim="4"/>'
                )
                # 4b. Main structural cast-steel column (primary collision collider)
                walls_xml.append(
                    f'<geom name="pillar_{p_idx}" type="cylinder" pos="{px:.4f} {py:.4f} {bh / 2:.4f}" '
                    f'size="{pr:.4f} {bh / 2:.4f}" material="bollard_mat" '
                    f'friction="0.9 0.01 0.001" condim="4" priority="1" solref="0.005 1"/>'
                )
                # 4c. High-contrast reflective hazard yellow warning collar
                walls_xml.append(
                    f'<geom name="bollard_stripe_{p_idx}" type="cylinder" pos="{px:.4f} {py:.4f} {collar_z:.4f}" '
                    f'size="{pr * 1.018:.4f} {collar_h / 2:.4f}" material="bollard_stripe_mat" '
                    f'friction="0.9 0.01 0.001" condim="3"/>'
                )
                # 4d. Smooth hemispherical dome cap at the top
                walls_xml.append(
                    f'<geom name="bollard_cap_{p_idx}" type="sphere" pos="{px:.4f} {py:.4f} {bh - pr * 0.15:.4f}" '
                    f'size="{pr * 0.98:.4f}" material="bollard_mat" '
                    f'friction="0.9 0.01 0.001" condim="3"/>'
                )
            elif len(p_item) >= 4:
                # Rectangular Reinforced Barrier Block
                px, py, phx, phy = float(p_item[0]), float(p_item[1]), float(p_item[2]), float(p_item[3])
                phz = float(p_item[4]) if len(p_item) > 4 else half_h
                walls_xml.append(
                    f'<geom name="barrier_{p_idx}" type="box" pos="{px:.4f} {py:.4f} {phz:.4f}" '
                    f'size="{phx:.4f} {phy:.4f} {phz:.4f}" material="concrete_barrier_mat" '
                    f'friction="0.9 0.01 0.001" condim="4" priority="1" solref="0.005 1"/>'
                )

    walls_xml.extend(features.steps_xml(scenario))
    walls_xml.extend(features.gaps_xml(scenario))
    walls_xml.extend(features.sand_patch_xml(scenario))
    walls_xml.extend(features.stone_field_xml(scenario))
    walls_xml.extend(features.ramp_xml(scenario))
    walls_xml.extend(features.staircase_xml(scenario))
    walls_xml.extend(features.pipe_xml(scenario))
    walls_xml.extend(features.yardline_xml(scenario))
    walls_xml.extend(features.vertical_cylinder_xml(scenario))
    walls_xml.extend(features.motordrome_xml(scenario))
    walls_xml.extend(features.cone_xml(scenario))
    # 5. Goal Marker & Pad (optional)
    gx, gy = (float(scenario.goal[0]), float(scenario.goal[1])) if scenario.goal is not None else (spawn_xy[0] + 5.0, spawn_xy[1])
    has_goal = getattr(scenario, "has_goal", True) and scenario.goal is not None
    if has_goal:
        goal_xml = f"""
        <geom name="goal_pad" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.004"
              size="0.45 0.004" material="goal_pad_mat" contype="0" conaffinity="0"/>
        <geom name="goal_marker" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.25"
              size="0.25 0.25" material="goal_mat" contype="0" conaffinity="0"/>
        """
    else:
        goal_xml = ""


    # 6. Cameras Setup
    if len(walls) > 0:
        all_x = np.concatenate([walls[:, 0], walls[:, 2]])
        all_y = np.concatenate([walls[:, 1], walls[:, 3]])
        cx_arena = float(np.mean(all_x))
        cy_arena = float(np.mean(all_y))
        span = max(float(all_x.max() - all_x.min()), float(all_y.max() - all_y.min()))
        cam_h = max(span * 1.05, 7.5)
    else:
        cx_arena = (spawn_xy[0] + gx) / 2.0
        cy_arena = (spawn_xy[1] + gy) / 2.0
        span = float(np.hypot(gx - spawn_xy[0], gy - spawn_xy[1]))
        cam_h = max(span * 1.1, 6.0)

    # Initial tangent for chase camera
    path_pts = np.asarray(scenario.path_pts, dtype=float).reshape(-1, 2)
    k_pt = min(5, len(path_pts) - 1)
    d_tan = path_pts[k_pt] - spawn_xy
    n_tan = float(np.linalg.norm(d_tan))
    d_hat = d_tan / n_tan if n_tan > 1e-6 else np.array([1.0, 0.0])
    chase_cam_x = spawn_xy[0] - d_hat[0] * 1.3
    chase_cam_y = spawn_xy[1] - d_hat[1] * 1.3
    chase_cam_z = 0.55

    if is_realistic:
        core_rgba = "0.22 0.24 0.28 1"
    elif is_white:
        core_rgba = "0.92 0.94 0.96 1"
    else:
        core_rgba = "1.0 0.82 0.15 1"

    # Floor grid. The checker texture holds a 2x2 block of squares, so one
    # tile spans two squares. With texuniform the repeat count is per metre,
    # hence 1 / (2 * square size). A visible grid is what makes motion
    # readable in the videos; too fine a repeat blurs into flat colour.
    grid_repeat = 1.0 / max(2.0 * float(floor_square_m), 1e-6)
    floor_half = float(floor_half_extent)
    grid_rgb1 = floor_rgb1
    grid_rgb2 = floor_rgb2

    # Floor handling: check if scenario has floor holes / gaps
    holes = getattr(scenario, "holes", None) or getattr(scenario, "gaps", None)
    if holes is not None and len(holes) > 0:
        max_depth = max(float(h[4]) if len(h) > 4 else 0.10 for h in holes)
        # The walkable floor is cut into slabs around each hole. Those slabs
        # must be thick solid blocks, not thin sheets. A thin sheet lets a
        # fast rod tip cross the mid-plane, after which the contact normal
        # flips and pushes the rod out through the underside. Robots then
        # snag on the slab edges and stall. The slab also forms the visible
        # wall of the pit, so it has to reach below the deepest hole floor.
        slab_half_thickness = max(0.15, max_depth + 0.05)
        floor_plane_z = -(2.0 * slab_half_thickness + 0.05)
        hole_rects = []
        for h in holes:
            gx, gy, ghx, ghy = float(h[0]), float(h[1]), float(h[2]), float(h[3])
            hole_rects.append((gx - ghx, gx + ghx, gy - ghy, gy + ghy))

        arena_limit = min(float(floor_half), 40.0)
        slabs = _decompose_floor_slabs((-arena_limit, arena_limit), (-arena_limit, arena_limit), hole_rects)
        floor_elements = [
            f'<geom name="floor_base" type="plane" pos="0 0 {floor_plane_z:.4f}" size="{floor_half:.1f} {floor_half:.1f} 0.1" material="pit_floor_mat" friction="0.85 0.015 0.005" condim="4"/>'
        ]
        for s_i, (xa, xb, ya, yb) in enumerate(slabs):
            scx = (xa + xb) / 2.0
            scy = (ya + yb) / 2.0
            shx = (xb - xa) / 2.0
            shy = (yb - ya) / 2.0
            floor_elements.append(
                f'<geom name="floor_slab_{s_i}" type="box" '
                f'pos="{scx:.4f} {scy:.4f} {-slab_half_thickness:.4f}" '
                f'size="{shx:.4f} {shy:.4f} {slab_half_thickness:.4f}" '
                f'material="grid" friction="0.85 0.015 0.005" condim="4"/>'
            )
        floor_geom_xml = "\n        ".join(floor_elements)
    else:
        floor_geom_xml = f'<geom name="floor" type="plane" size="{floor_half:.1f} {floor_half:.1f} 0.1" material="grid" friction="0.85 0.015 0.005" condim="4"/>'

    xml_str = f"""<mujoco model="radial_sphere_arena">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="{timestep:.5f}" gravity="0 0 -9.81" integrator="implicitfast"/>

    <!-- Pin the model extent to robot scale. MuJoCo derives the near/far clip
         planes from it, so without this a large floor plane pushes znear past
         the close-up cameras and the robot disappears from the render. -->
    <statistic extent="4" center="0 0 0.4"/>

    <default>
        <geom contype="2" conaffinity="1"/>
    </default>

    <visual>
        <headlight ambient="0.45 0.45 0.45" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
        <rgba haze="0.12 0.20 0.30 1"/>
        <global azimuth="140" elevation="-30"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="{grid_rgb1}" rgb2="{grid_rgb2}"/>
        <texture name="skybox" type="skybox" builtin="gradient"
                 rgb1="0.20 0.35 0.55" rgb2="0.04 0.07 0.12" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="{grid_repeat:.4f} {grid_repeat:.4f}" reflectance="0.08" texuniform="true"/>
        <material name="pit_floor_mat" rgba="0.10 0.12 0.16 1" specular="0.2" shininess="0.3" reflectance="0.04"/>
        <material name="wall_mat" rgba="0.68 0.64 0.58 1" specular="0.2" shininess="0.3" reflectance="0.06"/>
        <material name="goal_mat" rgba="0.0 0.85 0.90 0.60" reflectance="0.1"/>
        <material name="goal_pad_mat" rgba="0.0 0.85 0.90 0.35" reflectance="0.05"/>
        <material name="core_mat" rgba="{core_rgba}" specular="0.6" shininess="0.8" reflectance="0.12"/>
        <!-- Realistic Industrial Blocker Materials -->
        <material name="bollard_mat" rgba="0.20 0.22 0.25 1" specular="0.5" shininess="0.7" reflectance="0.12"/>
        <material name="bollard_stripe_mat" rgba="0.96 0.78 0.08 1" specular="0.6" shininess="0.85" reflectance="0.22"/>
        <material name="bollard_base_mat" rgba="0.14 0.15 0.17 1" specular="0.3" shininess="0.4"/>
        <material name="concrete_barrier_mat" rgba="0.58 0.56 0.54 1" specular="0.1" shininess="0.1" reflectance="0.03"/>
        <!-- Realistic Wooden Plank Materials -->
        <material name="wood_plank_mat" rgba="0.56 0.36 0.20 1" specular="0.2" shininess="0.3" reflectance="0.04"/>
        <material name="wood_dark_mat" rgba="0.40 0.24 0.12 1" specular="0.1" shininess="0.2"/>
        <material name="wood_bracket_mat" rgba="0.18 0.19 0.22 1" specular="0.4" shininess="0.6"/>
        <!-- Realistic Mountainous Rock Materials -->
        <material name="granite_rock_mat" rgba="0.38 0.39 0.42 1" specular="0.3" shininess="0.4"/>
        <material name="slate_rock_mat" rgba="0.24 0.25 0.28 1" specular="0.35" shininess="0.5"/>
        <material name="sandstone_rock_mat" rgba="0.64 0.50 0.36 1" specular="0.15" shininess="0.2"/>
        <material name="basalt_rock_mat" rgba="0.18 0.19 0.21 1" specular="0.25" shininess="0.3"/>
        <!-- Transparent Glass Conduit / Pipe Materials -->
        <material name="glass_pipe_mat" rgba="0.25 0.78 0.95 0.28" specular="0.95" shininess="0.95" reflectance="0.25"/>
        <material name="pipe_ring_mat" rgba="0.18 0.20 0.24 1.0" specular="0.8" shininess="0.9"/>
        <!-- Incline Slopes & Staircase Materials -->
        <material name="ramp_mat" rgba="0.45 0.46 0.48 1.0" specular="0.3" shininess="0.4"/>
        <material name="stair_tread_blue_mat" rgba="0.10 0.28 0.58 1.0" specular="0.25" shininess="0.35"/>
        <material name="stair_tread_teal_mat" rgba="0.05 0.48 0.58 1.0" specular="0.25" shininess="0.35"/>
        <material name="stair_nosing_mat" rgba="1.00 0.76 0.00 1.0" emission="0.12" specular="0.5" shininess="0.7"/>
    </asset>

    <worldbody>
        <light pos="{cx_arena:.2f} {cy_arena:.2f} 12" dir="0 0 -1" directional="true"
               diffuse="0.90 0.90 0.90" specular="0.3 0.3 0.3"/>
        <light pos="0 0 8" dir="0 0 -1" directional="false"
               diffuse="0.40 0.40 0.40" specular="0.2 0.2 0.2"/>

        <!-- Floor Plane -->
        {floor_geom_xml}

        <!-- Maze Walls -->
        {''.join(walls_xml)}

        <!-- Goal Object -->
        {goal_xml}

        <!-- Cameras -->
        <camera name="bird_fixed" pos="{cx_arena:.3f} {cy_arena:.3f} {cam_h:.3f}"
                euler="0 0 0" mode="fixed"/>
        <camera name="chase" pos="{chase_cam_x:.3f} {chase_cam_y:.3f} {chase_cam_z:.3f}"
                mode="targetbody" target="core"/>

        <!-- Radial Sphere Robot -->
        <body name="core" pos="{spawn_xy[0]:.4f} {spawn_xy[1]:.4f} {spawn_z:.4f}">
            <freejoint name="root"/>
            <geom name="core_geom" type="sphere" size="{sphere_radius}"
                  material="core_mat" mass="{core_mass}"
                  friction="0.85 0.015 0.005" condim="4"
                  contype="1" conaffinity="2"/>
            <!-- Central Electronics & Avionics Hub (Protected Core Hub) -->
            <geom name="avionics_hub" type="sphere" size="0.045"
                  rgba="0.10 0.75 0.90 0.85" mass="0.10"
                  contype="0" conaffinity="0"/>
            <!-- Central IMU Sensor Site (At CoG) -->
            <site name="imu_site" pos="0 0 0" size="0.01" type="sphere" rgba="0 1 1 0"/>
            {''.join(bars_xml)}
        </body>
    </worldbody>

    {f'<equality>{"".join(equalities_xml)}</equality>' if equalities_xml else ''}

    <actuator>
        {''.join(actuators_xml)}
    </actuator>

    <sensor>
        {''.join(sensors_xml)}
    </sensor>
</mujoco>
"""
    return xml_str, dirs
