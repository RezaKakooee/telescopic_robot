"""Direct MuJoCo MJCF Scene and Robot Builder.

Generates a complete, self-contained MuJoCo XML scene string containing the arena,
floor, maze walls, obstacles, goal marker, lighting, cameras, and the 60-bar radial
sphere robot with physical slide joints, contact feet, and position actuators.
"""
from __future__ import annotations

import colorsys
from typing import Sequence
import numpy as np

from .geometry import fibonacci_sphere
from .mjcf import SLEEVE_STUB, TIP_GAP, FOOT_RADIUS, rolling_radius


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

    # 4b. Ground Step / Rectangular Wooden Planks (passover obstacles)
    steps = getattr(scenario, "wood_planks", None)
    if steps is None or len(steps) == 0:
        steps = getattr(scenario, "steps", None)

    if steps is not None and len(steps) > 0:
        for s_idx, s_def in enumerate(steps):
            sx, sy, shx, shy, sh = float(s_def[0]), float(s_def[1]), float(s_def[2]), float(s_def[3]), float(s_def[4])

            # Main rectangular wooden timber body (primary contact surface)
            walls_xml.append(
                f'<geom name="wood_plank_{s_idx}" type="box" pos="{sx:.4f} {sy:.4f} {sh / 2.0:.4f}" '
                f'size="{shx:.4f} {shy:.4f} {sh / 2.0:.4f}" material="wood_plank_mat" '
                f'friction="1.25 0.01 0.001" condim="4" priority="1" solref="0.008 1" solimp="0.92 0.96 0.002"/>'
            )
            # End-grain caps and anchor brackets only for low ground planks (not tall vertical box pillars)
            if sh <= 0.35:
                cap_th = 0.012
                walls_xml.append(
                    f'<geom name="wood_cap_a_{s_idx}" type="box" pos="{sx:.4f} {sy - shy + cap_th / 2:.4f} {sh / 2.0:.4f}" '
                    f'size="{shx * 1.008:.4f} {cap_th / 2:.4f} {sh / 2.0 * 1.008:.4f}" material="wood_dark_mat" '
                    f'friction="1.2 0.01 0.001" condim="3"/>'
                )
                walls_xml.append(
                    f'<geom name="wood_cap_b_{s_idx}" type="box" pos="{sx:.4f} {sy + shy - cap_th / 2:.4f} {sh / 2.0:.4f}" '
                    f'size="{shx * 1.008:.4f} {cap_th / 2:.4f} {sh / 2.0 * 1.008:.4f}" material="wood_dark_mat" '
                    f'friction="1.2 0.01 0.001" condim="3"/>'
                )
                br_w = 0.025
                walls_xml.append(
                    f'<geom name="wood_bracket_a_{s_idx}" type="box" pos="{sx:.4f} {sy - shy - br_w / 2:.4f} 0.008" '
                    f'size="{shx * 0.55:.4f} {br_w / 2:.4f} 0.008" material="wood_bracket_mat" '
                    f'friction="0.8 0.005 0.0001" condim="3"/>'
                )
                walls_xml.append(
                    f'<geom name="wood_bracket_b_{s_idx}" type="box" pos="{sx:.4f} {sy + shy + br_w / 2:.4f} 0.008" '
                    f'size="{shx * 0.55:.4f} {br_w / 2:.4f} 0.008" material="wood_bracket_mat" '
                    f'friction="0.8 0.005 0.0001" condim="3"/>'
                )


    # 4c. Floor Gaps / Holes / Pits in the ground
    # Each gap: (cx, cy, half_x, half_y, depth)
    gaps = getattr(scenario, "gaps", None)
    if gaps is not None and len(gaps) > 0:
        for g_idx, g_def in enumerate(gaps):
            gx, gy, ghx, ghy = float(g_def[0]), float(g_def[1]), float(g_def[2]), float(g_def[3])
            gdepth = float(g_def[4]) if len(g_def) > 4 else 0.10

            edge_w = 0.025
            edge_h = 0.012

            # Left and Right Steel Hazard Curbs with High-Visibility Edging
            walls_xml.append(
                f'<geom name="gap_curb_left_{g_idx}" type="box" '
                f'pos="{gx - ghx - edge_w / 2:.4f} {gy:.4f} {edge_h / 2:.4f}" '
                f'size="{edge_w / 2:.4f} {ghy:.4f} {edge_h / 2:.4f}" rgba="0.96 0.78 0.08 1" '
                f'friction="1.1 0.01 0.001" condim="4" priority="1"/>'
            )
            walls_xml.append(
                f'<geom name="gap_curb_right_{g_idx}" type="box" '
                f'pos="{gx + ghx + edge_w / 2:.4f} {gy:.4f} {edge_h / 2:.4f}" '
                f'size="{edge_w / 2:.4f} {ghy:.4f} {edge_h / 2:.4f}" rgba="0.96 0.78 0.08 1" '
                f'friction="1.1 0.01 0.001" condim="4" priority="1"/>'
            )
            # Front and Back Steel Boundary Plates
            walls_xml.append(
                f'<geom name="gap_end_a_{g_idx}" type="box" '
                f'pos="{gx:.4f} {gy - ghy - edge_w / 2:.4f} {edge_h / 2:.4f}" '
                f'size="{ghx + edge_w:.4f} {edge_w / 2:.4f} {edge_h / 2:.4f}" rgba="0.22 0.24 0.26 1" '
                f'friction="0.9 0.01 0.001" condim="3"/>'
            )
            walls_xml.append(
                f'<geom name="gap_end_b_{g_idx}" type="box" '
                f'pos="{gx:.4f} {gy + ghy + edge_w / 2:.4f} {edge_h / 2:.4f}" '
                f'size="{ghx + edge_w:.4f} {edge_w / 2:.4f} {edge_h / 2:.4f}" rgba="0.22 0.24 0.26 1" '
                f'friction="0.9 0.01 0.001" condim="3"/>'
            )
            # The Deep Dark Chasm Pit Void Floor
            walls_xml.append(
                f'<geom name="gap_pit_floor_{g_idx}" type="box" '
                f'pos="{gx:.4f} {gy:.4f} {-gdepth:.4f}" '
                f'size="{ghx:.4f} {ghy:.4f} 0.008" rgba="0.05 0.06 0.08 1" '
                f'friction="0.4 0.005 0.0001" condim="3"/>'
            )

    # 4d. Sand Patches — high-friction rough terrain that slows the ball
    # Each sand_patch: (cx, cy, half_x, half_y)
    sand_patches = getattr(scenario, "sand_patches", None)
    if sand_patches is not None and len(sand_patches) > 0:
        for sp_idx, (spx, spy, sphx, sphy) in enumerate(sand_patches):
            # Thin rough-textured slab sitting flush on the floor
            walls_xml.append(
                f'<geom name="sand_{sp_idx}" type="box" '
                f'pos="{spx:.4f} {spy:.4f} 0.003" '
                f'size="{sphx:.4f} {sphy:.4f} 0.003" rgba="0.82 0.72 0.50 1" '
                f'friction="3.5 0.3 0.01" condim="4"/>'
            )
            # Sprinkle small granules on top for visual texture
            rng = np.random.RandomState(42 + sp_idx)
            n_grains = 35
            for gi in range(n_grains):
                ox = spx + rng.uniform(-sphx * 0.9, sphx * 0.9)
                oy = spy + rng.uniform(-sphy * 0.9, sphy * 0.9)
                gr = rng.uniform(0.004, 0.010)
                brightness = rng.uniform(0.55, 0.85)
                walls_xml.append(
                    f'<geom name="grain_{sp_idx}_{gi}" type="sphere" '
                    f'pos="{ox:.4f} {oy:.4f} {gr:.4f}" size="{gr:.4f}" '
                    f'rgba="{brightness:.2f} {brightness * 0.88:.2f} {brightness * 0.62:.2f} 1" '
                    f'friction="2.5 0.2 0.005" condim="3" mass="0.001"/>'
                )

    # 4e. Scattered Mountainous Rocks, Boulders, and Stone Slabs on the floor
    # Each stone_zone: (cx, cy, half_x, half_y, n_stones, max_stone_size)
    stones = getattr(scenario, "stones", None)
    if stones is not None and len(stones) > 0:
        for st_idx, stone_def in enumerate(stones):
            stx, sty, sthx, sthy = float(stone_def[0]), float(stone_def[1]), float(stone_def[2]), float(stone_def[3])
            n_stones = int(stone_def[4]) if len(stone_def) > 4 else 20
            max_sz = float(stone_def[5]) if len(stone_def) > 5 else 0.045
            rng = np.random.RandomState(2026 + st_idx)
            
            rock_materials = ["granite_rock_mat", "slate_rock_mat", "sandstone_rock_mat", "basalt_rock_mat"]

            for si in range(n_stones):
                ox = stx + rng.uniform(-sthx * 0.95, sthx * 0.95)
                oy = sty + rng.uniform(-sthy * 0.95, sthy * 0.95)
                sr = rng.uniform(max_sz * 0.25, max_sz)
                mat = rock_materials[si % len(rock_materials)]

                # Irregular rock geometry: boxes with 3D tilt, ellipsoids, spheres
                geom_choice = rng.choice(["tilted_box", "ellipsoid", "sphere"])

                if geom_choice == "tilted_box":
                    sx2 = rng.uniform(sr * 0.7, sr * 1.4)
                    sy2 = rng.uniform(sr * 0.7, sr * 1.4)
                    sz2 = rng.uniform(sr * 0.4, sr * 0.9)
                    # Random rock angular faceting
                    roll = rng.uniform(-25.0, 25.0)
                    pitch = rng.uniform(-25.0, 25.0)
                    yaw = rng.uniform(0.0, 360.0)
                    walls_xml.append(
                        f'<geom name="rock_{st_idx}_{si}" type="box" '
                        f'pos="{ox:.4f} {oy:.4f} {sz2 * 0.85:.4f}" '
                        f'size="{sx2:.4f} {sy2:.4f} {sz2:.4f}" '
                        f'euler="{roll:.1f} {pitch:.1f} {yaw:.1f}" material="{mat}" '
                        f'friction="1.35 0.02 0.005" condim="4" priority="1" solref="0.008 1"/>'
                    )
                elif geom_choice == "ellipsoid":
                    sx2 = rng.uniform(sr * 0.8, sr * 1.3)
                    sy2 = rng.uniform(sr * 0.8, sr * 1.3)
                    sz2 = rng.uniform(sr * 0.5, sr * 0.8)
                    walls_xml.append(
                        f'<geom name="rock_{st_idx}_{si}" type="ellipsoid" '
                        f'pos="{ox:.4f} {oy:.4f} {sz2:.4f}" '
                        f'size="{sx2:.4f} {sy2:.4f} {sz2:.4f}" material="{mat}" '
                        f'friction="1.35 0.02 0.005" condim="4" priority="1" solref="0.008 1"/>'
                    )
                else:
                    walls_xml.append(
                        f'<geom name="rock_{st_idx}_{si}" type="sphere" '
                        f'pos="{ox:.4f} {oy:.4f} {sr * 0.8:.4f}" size="{sr:.4f}" material="{mat}" '
                        f'friction="1.35 0.02 0.005" condim="4" priority="1" solref="0.008 1"/>'
                    )


    # 4f. Incline Slopes / Ramps (Uphill & Downhill)
    ramps = getattr(scenario, "ramps", None)
    if ramps is not None and len(ramps) > 0:
        for r_idx, r_def in enumerate(ramps):
            rcx, rcy, rlen, rwid, r_h, r_pitch, r_yaw = float(r_def[0]), float(r_def[1]), float(r_def[2]), float(r_def[3]), float(r_def[4]), float(r_def[5]), float(r_def[6])
            slab_th = 0.05
            if abs(r_pitch) < 1e-3:
                # Solid elevated plateau block from floor z=0 to top z=r_h
                block_h = max(r_h, slab_th)
                cz = block_h / 2.0
                pitch_euler = 0.0
                walls_xml.append(
                    f'<geom name="ramp_slab_{r_idx}" type="box" '
                    f'pos="{rcx:.4f} {rcy:.4f} {cz:.4f}" '
                    f'size="{rlen / 2.0 * 1.02:.4f} {rwid / 2.0:.4f} {block_h / 2.0:.4f}" '
                    f'euler="0 0 {r_yaw:.2f}" material="ramp_mat" '
                    f'friction="1.5 0.02 0.005" condim="4" priority="1"/>'
                )
            else:
                # Inclined ramp slab (uphill or downhill)
                cz = abs(r_h) / 2.0
                pitch_euler = -r_pitch
                walls_xml.append(
                    f'<geom name="ramp_slab_{r_idx}" type="box" '
                    f'pos="{rcx:.4f} {rcy:.4f} {cz:.4f}" '
                    f'size="{rlen / 2.0 * 1.02:.4f} {rwid / 2.0:.4f} {slab_th / 2.0:.4f}" '
                    f'euler="0 {pitch_euler:.2f} {r_yaw:.2f}" material="ramp_mat" '
                    f'friction="1.5 0.02 0.005" condim="4" priority="1"/>'
                )
            # High-visibility guide curbs on lateral edges
            curb_h = 0.10
            curb_w = 0.04
            walls_xml.append(
                f'<geom name="ramp_curb_l_{r_idx}" type="box" '
                f'pos="{rcx:.4f} {rcy + rwid/2.0 + curb_w/2.0:.4f} {cz + curb_h/2.0:.4f}" '
                f'size="{rlen / 2.0 * 1.02:.4f} {curb_w / 2.0:.4f} {curb_h / 2.0:.4f}" '
                f'euler="0 {pitch_euler:.2f} {r_yaw:.2f}" rgba="0.96 0.78 0.08 1" '
                f'friction="0.8 0.005 0.0001" condim="3"/>'
            )
            walls_xml.append(
                f'<geom name="ramp_curb_r_{r_idx}" type="box" '
                f'pos="{rcx:.4f} {rcy - rwid/2.0 - curb_w/2.0:.4f} {cz + curb_h/2.0:.4f}" '
                f'size="{rlen / 2.0 * 1.02:.4f} {curb_w / 2.0:.4f} {curb_h / 2.0:.4f}" '
                f'euler="0 {pitch_euler:.2f} {r_yaw:.2f}" rgba="0.96 0.78 0.08 1" '
                f'friction="0.8 0.005 0.0001" condim="3"/>'
            )

    # 4g. Multi-Step Staircases
    staircases = getattr(scenario, "staircases", None)
    if staircases is not None and len(staircases) > 0:
        for st_idx, sc_def in enumerate(staircases):
            start_x, start_y = float(sc_def[0]), float(sc_def[1])
            n_steps = int(sc_def[2])
            rise = float(sc_def[3])
            run = float(sc_def[4])
            wid = float(sc_def[5])
            yaw = float(sc_def[6])
            is_down = bool(sc_def[7]) if len(sc_def) > 7 else False

            for step_i in range(n_steps):
                if is_down:
                    step_h = rise * (n_steps - step_i)
                    step_x = start_x + (step_i + 0.5) * run
                else:
                    step_h = rise * (step_i + 1)
                    step_x = start_x + (step_i + 0.5) * run
                step_y = start_y
                tread_mat = ("stair_tread_blue_mat"
                             if (st_idx + step_i) % 2 == 0
                             else "stair_tread_teal_mat")
                walls_xml.append(
                    f'<geom name="stair_{st_idx}_{step_i}" type="box" '
                    f'pos="{step_x:.4f} {step_y:.4f} {step_h / 2.0:.4f}" '
                    f'size="{run / 2.0:.4f} {wid / 2.0:.4f} {step_h / 2.0:.4f}" '
                    f'material="{tread_mat}" friction="1.35 0.02 0.005" condim="4" priority="1"/>'
                )
                # Preserve the original physical nosing exactly: even this
                # small contact strip affects which rods support a landing.
                walls_xml.append(
                    f'<geom name="stair_nosing_{st_idx}_{step_i}" type="box" '
                    f'pos="{step_x - run/2.0 + 0.015:.4f} {step_y:.4f} {step_h - 0.003:.4f}" '
                    f'size="0.015 {wid / 2.0 * 0.99:.4f} 0.003" '
                    f'material="stair_nosing_mat" friction="1.2 0.01 0.001" condim="3"/>'
                )
                # Add a separate, raised visual-only safety band. The larger
                # width and offset avoid the old coplanar z-fighting, while
                # disabled collision bits leave the calibrated physics alone.
                walls_xml.append(
                    f'<geom name="stair_nosing_visual_{st_idx}_{step_i}" type="box" '
                    f'pos="{step_x - run/2.0 + 0.045:.4f} {step_y:.4f} {step_h + 0.004:.4f}" '
                    f'size="0.045 {wid / 2.0 * 0.99:.4f} 0.004" '
                    f'material="stair_nosing_mat" contype="0" conaffinity="0"/>'
                )
                # Matching vertical band makes the edge readable from the
                # low oblique follow camera as well as from above.
                walls_xml.append(
                    f'<geom name="stair_riser_band_{st_idx}_{step_i}" type="box" '
                    f'pos="{step_x - run/2.0 - 0.004:.4f} {step_y:.4f} {max(step_h - 0.045, 0.045):.4f}" '
                    f'size="0.004 {wid / 2.0 * 0.99:.4f} {min(0.045, step_h / 2.0):.4f}" '
                    f'material="stair_nosing_mat" contype="0" conaffinity="0"/>'
                )

    # 4h. Transparent Glass Pipe / Conduit (In-Pipe Crawling Inspection)
    pipes = getattr(scenario, "pipes", None)
    if pipes is not None and len(pipes) > 0:
        for p_idx, p_def in enumerate(pipes):
            start_x, start_y = float(p_def[0]), float(p_def[1])
            p_len = float(p_def[2])
            in_rad = float(p_def[3])
            out_rad = float(p_def[4]) if len(p_def) > 4 else in_rad + 0.015

            cx = start_x + p_len / 2.0
            cy = start_y
            cz = in_rad + 0.02

            # Regular 16-sided polygonal transparent glass barrel with flat bottom track
            n_facets = 16
            facet_th = out_rad - in_rad
            # Exact facet width so adjacent facets meet with flush tight seams
            facet_w = float(2.0 * in_rad * np.tan(np.pi / n_facets) + 0.002)

            for fi in range(n_facets):
                # Align so fi=8 is exactly at 180 deg (flat horizontal floor at the bottom of the tube)
                angle_rad = fi * (2.0 * np.pi / n_facets)
                angle_deg = float(np.degrees(angle_rad))
                r_mid = in_rad + facet_th / 2.0
                fy = cy + r_mid * np.sin(angle_rad)
                fz = cz + r_mid * np.cos(angle_rad)

                walls_xml.append(
                    f'<geom name="glass_facet_{p_idx}_{fi}" type="box" '
                    f'pos="{cx:.4f} {fy:.4f} {fz:.4f}" '
                    f'size="{p_len / 2.0:.4f} {facet_w / 2.0:.4f} {facet_th / 2.0:.4f}" '
                    f'euler="{-angle_deg:.1f} 0 0" material="glass_pipe_mat" '
                    f'friction="1.2 0.01 0.001" condim="3" priority="1" solref="0.012 1"/>'
                )

            # Chrome metallic reinforcement collar rings (outer perimeter ring only)
            n_rings = max(int(p_len / 2.5) + 1, 2)
            for ri in range(n_rings):
                rx = start_x + ri * (p_len / (n_rings - 1))
                # 16-facet outer collar ring around outer perimeter
                for rfi in range(n_facets):
                    r_angle = rfi * (2.0 * np.pi / n_facets)
                    r_deg = float(np.degrees(r_angle))
                    r_pos_y = cy + (out_rad + 0.01) * np.sin(r_angle)
                    r_pos_z = cz + (out_rad + 0.01) * np.cos(r_angle)
                    walls_xml.append(
                        f'<geom name="pipe_collar_{p_idx}_{ri}_{rfi}" type="box" '
                        f'pos="{rx:.4f} {r_pos_y:.4f} {r_pos_z:.4f}" '
                        f'size="0.025 {facet_w / 2.0 * 1.05:.4f} 0.012" '
                        f'euler="{-r_deg:.1f} 0 0" material="pipe_ring_mat" '
                        f'contype="0" conaffinity="0"/>'
                    )

    # 4i. Athletic Runway Yardlines & Painted Distance Markers (Pure Visual, Zero Friction Obstruction)
    yardlines = getattr(scenario, "yardlines", None)
    if yardlines is not None and len(yardlines) > 0:
        for y_idx, y_def in enumerate(yardlines):
            yx, yy, yhx, yhy, y_rgba = float(y_def[0]), float(y_def[1]), float(y_def[2]), float(y_def[3]), str(y_def[4])
            walls_xml.append(
                f'<geom name="yardline_{y_idx}" type="box" '
                f'pos="{yx:.4f} {yy:.4f} 0.0015" '
                f'size="{yhx:.4f} {yhy:.4f} 0.0015" rgba="{y_rgba}" '
                f'contype="0" conaffinity="0"/>'
            )

    # 4j. Vertical Transparent Cylinders / Silos (Wall of Death / Spiral Vortex Climbing)
    vcyls = getattr(scenario, "vertical_cylinders", None)
    if vcyls is not None and len(vcyls) > 0:
        for vc_idx, vc_def in enumerate(vcyls):
            cx, cy = float(vc_def[0]), float(vc_def[1])
            height = float(vc_def[2])
            in_rad = float(vc_def[3])
            out_rad = float(vc_def[4]) if len(vc_def) > 4 else in_rad + 0.02

            n_facets = 24
            facet_th = out_rad - in_rad
            facet_w = float(2.0 * in_rad * np.tan(np.pi / n_facets) + 0.002)
            r_mid = in_rad + facet_th / 2.0
            cz = height / 2.0

            # 24 vertical transparent glass facets forming the hollow silo
            for fi in range(n_facets):
                angle_rad = fi * (2.0 * np.pi / n_facets)
                angle_deg = float(np.degrees(angle_rad))
                fx = cx + r_mid * np.cos(angle_rad)
                fy = cy + r_mid * np.sin(angle_rad)

                # Check if this facet is at the ramp entrance doorway (x <= 0.1, y <= -in_rad * 0.7)
                is_doorway = (fx < 0.10) and (fy < -in_rad * 0.65)
                if is_doorway:
                    # Doorway arch: open from z=0 to 0.70m, facet stands from z=0.70 to height
                    arch_h = height - 0.70
                    arch_cz = 0.70 + arch_h / 2.0
                    walls_xml.append(
                        f'<geom name="vcyl_facet_{vc_idx}_{fi}" type="box" '
                        f'pos="{fx:.4f} {fy:.4f} {arch_cz:.4f}" '
                        f'size="{facet_th / 2.0:.4f} {facet_w / 2.0:.4f} {arch_h / 2.0:.4f}" '
                        f'euler="0 0 {angle_deg:.1f}" material="glass_pipe_mat" '
                        f'friction="1.2 0.01 0.001" condim="3" priority="1" solref="0.012 1"/>'
                    )
                else:
                    # Full vertical facet
                    walls_xml.append(
                        f'<geom name="vcyl_facet_{vc_idx}_{fi}" type="box" '
                        f'pos="{fx:.4f} {fy:.4f} {cz:.4f}" '
                        f'size="{facet_th / 2.0:.4f} {facet_w / 2.0:.4f} {height / 2.0:.4f}" '
                        f'euler="0 0 {angle_deg:.1f}" material="glass_pipe_mat" '
                        f'friction="1.2 0.01 0.001" condim="3" priority="1" solref="0.012 1"/>'
                    )

            # Banked entry transition curve connecting ramp into the cylinder wall
            n_trans = 8
            for ti in range(n_trans):
                t_frac = (ti + 0.5) / n_trans
                tx = -0.80 * (1.0 - t_frac)
                ty = -in_rad + 0.12 * np.sin(t_frac * np.pi / 2.0)
                tz = 0.24 * (1.0 - t_frac)**2 + 0.03
                t_pitch = -16.7 * (1.0 - t_frac)
                t_roll = 25.0 * t_frac
                walls_xml.append(
                    f'<geom name="vcyl_bank_trans_{vc_idx}_{ti}" type="box" '
                    f'pos="{tx:.4f} {ty:.4f} {tz:.4f}" '
                    f'size="0.055 0.45 0.02" '
                    f'euler="{t_roll:.1f} {t_pitch:.1f} 0" material="ramp_mat" '
                    f'friction="1.4 0.01 0.001" condim="4" priority="1"/>'
                )

            # Perimeter chrome reinforcement rings
            n_rings = max(int(height / 1.0) + 1, 3)
            for ri in range(n_rings):
                rz = ri * (height / (n_rings - 1))
                for rfi in range(n_facets):
                    r_angle = rfi * (2.0 * np.pi / n_facets)
                    r_deg = float(np.degrees(r_angle))
                    rx = cx + (out_rad + 0.008) * np.cos(r_angle)
                    ry = cy + (out_rad + 0.008) * np.sin(r_angle)
                    walls_xml.append(
                        f'<geom name="vcyl_collar_{vc_idx}_{ri}_{rfi}" type="box" '
                        f'pos="{rx:.4f} {ry:.4f} {rz:.4f}" '
                        f'size="0.010 {facet_w / 2.0 * 1.05:.4f} 0.025" '
                        f'euler="0 0 {r_deg:.1f}" material="pipe_ring_mat" '
                        f'contype="0" conaffinity="0"/>'
                    )


    # 4k. Authentic Wall of Death (Motordrome / Silodrome with 45-deg Base Apron Ramp)
    motordromes = getattr(scenario, "motordromes", None)
    if motordromes is not None and len(motordromes) > 0:
        for md_idx, md_def in enumerate(motordromes):
            cx, cy = float(md_def[0]), float(md_def[1])
            floor_r = float(md_def[2])
            wall_r = float(md_def[3])
            apron_h = float(md_def[4])
            total_h = float(md_def[5])
            # Wall-of-death traction. The ball stays up only while
            # mu * v^2 / r >= g, so this number sets the arena's size.
            md_mu = float(md_def[6]) if len(md_def) > 6 else 1.35

            # More facets makes a rounder barrel; more profile rings makes a
            # smoother curve up the bank. Both are just geometry count.
            n_facets = int(md_def[8]) if len(md_def) > 8 and md_def[8] else 32
            # How far each ring's planks reach past their own segment. Some
            # overlap is needed or the rings leave a seam the feet can catch.
            plank_lap = float(md_def[9]) if len(md_def) > 9 and md_def[9] else 1.06
            plank_pad = float(md_def[10]) if len(md_def) > 10 and md_def[10] is not None else 0.02
            plank_thick = 0.015  # half-thickness

            # 1. Banked transition, built from a profile of (radius, height)
            # points. A single pair gives the old straight conical apron; a
            # longer list gives a curved drome bowl, shallow at the bottom and
            # steep at the top.
            #
            # The shape matters more than anything else in this arena. A ball
            # cannot be held on a vertical wall by friction: the friction that
            # holds it also spins it, so it rolls down. It CAN hold a banked
            # circle, where the boards themselves supply the centripetal force.
            # The bank a ball can ride follows from its speed alone,
            # v^2 = g * r * tan(bank), so a curved bowl lets the robot settle
            # at whatever height its current speed has earned. That is how a
            # real drome rider climbs, and it is the only way this one can.
            profile = md_def[7] if len(md_def) > 7 and md_def[7] is not None else None
            if profile is None:
                profile = [(floor_r, 0.0), (wall_r, apron_h)]
            profile = [(float(a), float(b)) for a, b in profile]

            for si in range(len(profile) - 1):
                r1, z1 = profile[si]
                r2, z2 = profile[si + 1]
                seg_dr, seg_dz = r2 - r1, z2 - z1
                seg_slant = float(np.hypot(seg_dr, seg_dz))
                if seg_slant < 1e-4:
                    continue
                bank_pitch = float(np.degrees(np.arctan2(seg_dz, seg_dr)))
                bank_rad = np.radians(bank_pitch)
                # A flat plank must span the chord at its outer radius to tile
                # without a gap. Anything past that overlaps its neighbours,
                # and a foot in the overlap picks up two contacts at once.
                # Measured: with the old fixed 20 mm pad, going from 32 to 48
                # planks cost 0.50 m of sustained ride height.
                seg_w = float(2.0 * max(r1, r2) * np.tan(np.pi / n_facets) + plank_pad)
                # Drop the plank by its own thickness so the riding surface,
                # not the box centre, follows the profile.
                seg_r = (r1 + r2) / 2.0 + plank_thick * np.sin(bank_rad)
                seg_z = (z1 + z2) / 2.0 - plank_thick * np.cos(bank_rad)

                for fi in range(n_facets):
                    ang_rad = fi * (2.0 * np.pi / n_facets)
                    ang_deg = float(np.degrees(ang_rad))
                    ax = cx + seg_r * np.cos(ang_rad)
                    ay = cy + seg_r * np.sin(ang_rad)
                    walls_xml.append(
                        f'<body name="md_apron_body_{md_idx}_{si}_{fi}" '
                        f'pos="{ax:.4f} {ay:.4f} {seg_z:.4f}" euler="0 0 {ang_deg:.1f}">'
                        f'<geom name="md_apron_{md_idx}_{si}_{fi}" type="box" '
                        f'size="{seg_slant / 2.0 * plank_lap:.4f} {seg_w / 2.0:.4f} {plank_thick:.4f}" '
                        f'euler="0 {-bank_pitch:.1f} 0" material="wood_plank_mat" '
                        f'friction="{md_mu:.2f} 0.01 0.001" condim="4" priority="1"/>'
                        f'</body>'
                    )

            # 2. Vertical 90-degree Cylindrical Wooden Wall (z = apron_h to total_h)
            vert_h = total_h - apron_h
            vert_cz = apron_h + vert_h / 2.0
            vert_w = float(2.0 * wall_r * np.tan(np.pi / n_facets) + 0.004)

            for fi in range(n_facets):
                ang_rad = fi * (2.0 * np.pi / n_facets)
                ang_deg = float(np.degrees(ang_rad))
                vx = cx + wall_r * np.cos(ang_rad)
                vy = cy + wall_r * np.sin(ang_rad)

                mat_choice = "wood_plank_mat" if fi % 4 != 0 else "wood_dark_mat"
                walls_xml.append(
                    f'<geom name="md_wall_{md_idx}_{fi}" type="box" '
                    f'pos="{vx:.4f} {vy:.4f} {vert_cz:.4f}" '
                    f'size="0.020 {vert_w / 2.0:.4f} {vert_h / 2.0:.4f}" '
                    f'euler="0 0 {ang_deg:.1f}" material="{mat_choice}" '
                    f'friction="{md_mu:.2f} 0.01 0.001" condim="4" priority="1"/>'
                )

            # 3. Perimeter Steel Tension Bands around Silo
            n_bands = max(int(vert_h / 0.8) + 1, 4)
            for bi in range(n_bands):
                bz = apron_h + bi * (vert_h / (n_bands - 1))
                for fi in range(n_facets):
                    ang_rad = fi * (2.0 * np.pi / n_facets)
                    ang_deg = float(np.degrees(ang_rad))
                    bx = cx + (wall_r + 0.025) * np.cos(ang_rad)
                    by = cy + (wall_r + 0.025) * np.sin(ang_rad)
                    walls_xml.append(
                        f'<geom name="md_band_{md_idx}_{bi}_{fi}" type="box" '
                        f'pos="{bx:.4f} {by:.4f} {bz:.4f}" '
                        f'size="0.008 {vert_w / 2.0 * 1.05:.4f} 0.020" '
                        f'euler="0 0 {ang_deg:.1f}" material="wood_bracket_mat" '
                        f'contype="0" conaffinity="0"/>'
                    )

            # 4. Central Stage Hub (visual only, like the booth in the photo)
            walls_xml.append(
                f'<geom name="md_hub_box_{md_idx}" type="box" '
                f'pos="{cx:.4f} {cy:.4f} 0.20" '
                f'size="0.22 0.22 0.20" material="wood_dark_mat" '
                f'contype="0" conaffinity="0"/>'
            )

    # 4k. Athletic / Traffic Training Cones (Slalom Course)
    cones_raw = getattr(scenario, "cones", None)
    if cones_raw is not None and len(cones_raw) > 0:
        cone_items = np.asarray(cones_raw, dtype=float)
        if cone_items.ndim == 1:
            cone_items = cone_items.reshape(1, -1)
        for c_idx, c_item in enumerate(cone_items):
            cx, cy = float(c_item[0]), float(c_item[1])
            cr = float(c_item[2]) if len(c_item) > 2 else 0.12
            cone_h = 0.35  # 35 cm tall athletic cone
            base_w = cr * 2.3
            base_th = 0.015

            # Weighted square black rubber base plate
            walls_xml.append(
                f'<geom name="cone_base_{c_idx}" type="box" '
                f'pos="{cx:.4f} {cy:.4f} {base_th / 2.0:.4f}" '
                f'size="{base_w / 2.0:.4f} {base_w / 2.0:.4f} {base_th / 2.0:.4f}" '
                f'rgba="0.12 0.12 0.14 1" friction="1.2 0.01 0.001" condim="4"/>'
            )
            # Lower bright orange cone body
            walls_xml.append(
                f'<geom name="cone_lower_{c_idx}" type="cylinder" '
                f'pos="{cx:.4f} {cy:.4f} {cone_h * 0.20:.4f}" '
                f'size="{cr * 0.90:.4f} {cone_h * 0.20:.4f}" '
                f'rgba="1.00 0.35 0.02 1" friction="0.8 0.005 0.0001" condim="4" priority="1"/>'
            )
            # High-visibility reflective white collar
            walls_xml.append(
                f'<geom name="cone_stripe_{c_idx}" type="cylinder" '
                f'pos="{cx:.4f} {cy:.4f} {cone_h * 0.55:.4f}" '
                f'size="{cr * 0.62:.4f} {cone_h * 0.12:.4f}" '
                f'rgba="0.95 0.95 0.98 1" friction="0.8 0.005 0.0001" condim="3"/>'
            )
            # Upper bright orange cone top
            walls_xml.append(
                f'<geom name="cone_upper_{c_idx}" type="cylinder" '
                f'pos="{cx:.4f} {cy:.4f} {cone_h * 0.80:.4f}" '
                f'size="{cr * 0.38:.4f} {cone_h * 0.12:.4f}" '
                f'rgba="1.00 0.35 0.02 1" friction="0.8 0.005 0.0001" condim="4"/>'
            )
            # Smooth rounded top cap
            walls_xml.append(
                f'<geom name="cone_cap_{c_idx}" type="sphere" '
                f'pos="{cx:.4f} {cy:.4f} {cone_h * 0.94:.4f}" '
                f'size="{cr * 0.36:.4f}" '
                f'rgba="1.00 0.35 0.02 1" friction="0.8 0.005 0.0001" condim="3"/>'
            )

    # 5. Goal Marker & Pad
    gx, gy = float(scenario.goal[0]), float(scenario.goal[1])
    goal_xml = f"""
    <geom name="goal_pad" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.004"
          size="0.45 0.004" material="goal_pad_mat" contype="0" conaffinity="0"/>
    <geom name="goal_marker" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.25"
          size="0.25 0.25" material="goal_mat" contype="0" conaffinity="0"/>
    """


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
        <geom name="floor" type="plane" size="{floor_half:.1f} {floor_half:.1f} 0.1" material="grid"
              friction="0.85 0.015 0.005" condim="4"/>

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
