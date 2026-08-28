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

    # Theme colors
    is_realistic = appearance_theme in ["realistic", "carbon_gunmetal"]
    is_white = appearance_theme == "aerospace_white"

    # Physics parameters (Modular Sim-to-Real vs. Baseline)
    s2r = sim2real_cfg or {}
    enable_sim2real = bool(s2r.get("enabled", False))
    
    joint_damping = "0.5" if enable_sim2real else "0"
    joint_frictionloss = "0.8" if enable_sim2real else "0"
    
    f_sl = float(s2r.get("rubber_friction_sliding", 0.85)) if enable_sim2real else 4.0
    f_t = float(s2r.get("rubber_friction_torsional", 0.015)) if enable_sim2real else 0.05
    f_r = float(s2r.get("rubber_friction_rolling", 0.005)) if enable_sim2real else 0.002
    foot_friction = f"{f_sl} {f_t} {f_r}"
    
    sr_time = float(s2r.get("rubber_solref_timeconst", 0.020)) if enable_sim2real else 0.005
    sr_damp = float(s2r.get("rubber_solref_dampratio", 1.20)) if enable_sim2real else 1.0
    foot_solref = f"{sr_time} {sr_damp}"
    
    foot_solimp = "0.90 0.95 0.005" if enable_sim2real else "0.95 0.99 0.001"
    
    max_f = float(s2r.get("actuator_force_limit", 50.0)) if enable_sim2real else 80.0
    actuator_forcerange = f"{-max_f} {max_f}"

    for k, (ux, uy, uz) in enumerate(dirs):
        u = np.array([ux, uy, uz], dtype=float)

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

        sleeve_from = (0.55 * sphere_radius) * u
        sleeve_to = sleeve_mouth * u
        rod_to = (tip0 - FOOT_RADIUS * 0.9) * u
        rod_from = (tip0 - bar_length) * u
        foot = tip0 * u

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
                       range="0 {max_extend}" armature="0.02" damping="{joint_damping}" frictionloss="{joint_frictionloss}"/>
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
            </body>
            """
        )
        actuators_xml.append(
            f'<general name="slide_{k}" joint="slide_{k}" '
            f'gainprm="900 0 0" biasprm="0 -900 -22" biastype="affine" gaintype="fixed" '
            f'ctrlrange="0 {max_extend}" forcerange="{actuator_forcerange}"/>'
        )

    # 2. Spawn Position
    spawn_xy = np.asarray(scenario.spawn_xy, dtype=float)[:2]
    spawn_z = rolling_radius(sphere_radius, 0.15 * max_extend) + 0.005

    # 3. Maze Walls Geometry
    walls_xml: list[str] = []
    half_th = wall_thickness / 2.0
    half_h = wall_height / 2.0
    walls = np.asarray(scenario.walls, dtype=float).reshape(-1, 4)
    for idx, (x1, y1, x2, y2) in enumerate(walls):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        # Add half_th to length so corners overlap cleanly without gaps
        sx = max(dx / 2.0 + half_th if dx > dy else half_th, half_th)
        sy = max(dy / 2.0 + half_th if dy >= dx else half_th, half_th)
        walls_xml.append(
            f'<geom name="wall_{idx}" type="box" pos="{cx:.4f} {cy:.4f} {half_h:.4f}" '
            f'size="{sx:.4f} {sy:.4f} {half_h:.4f}" material="wall_mat" '
            f'friction="0.8 0.005 0.0001" condim="3"/>'
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
            # Dark end-grain caps on both lateral ends
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
            # Heavy steel ground anchor brackets on outer sides
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
                walls_xml.append(
                    f'<geom name="stair_{st_idx}_{step_i}" type="box" '
                    f'pos="{step_x:.4f} {step_y:.4f} {step_h / 2.0:.4f}" '
                    f'size="{run / 2.0:.4f} {wid / 2.0:.4f} {step_h / 2.0:.4f}" '
                    f'material="stair_tread_mat" friction="1.35 0.02 0.005" condim="4" priority="1"/>'
                )
                # Safety nosing stripe on leading edge
                walls_xml.append(
                    f'<geom name="stair_nosing_{st_idx}_{step_i}" type="box" '
                    f'pos="{step_x - run/2.0 + 0.015:.4f} {step_y:.4f} {step_h - 0.003:.4f}" '
                    f'size="0.015 {wid / 2.0 * 0.99:.4f} 0.003" '
                    f'material="stair_nosing_mat" friction="1.2 0.01 0.001" condim="3"/>'
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


    # 5. Goal Marker & Pad
    gx, gy = float(scenario.goal[0]), float(scenario.goal[1])
    goal_xml = f"""
    <geom name="goal_pad" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.004"
          size="0.45 0.004" material="goal_pad_mat" contype="0" conaffinity="0"/>
    <geom name="goal_marker" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.25"
          size="0.25 0.25" material="goal_mat" contype="1" conaffinity="1"/>
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
        <material name="wall_mat" rgba="0.28 0.32 0.38 1" reflectance="0.05"/>
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
        <material name="stair_tread_mat" rgba="0.52 0.38 0.25 1.0" specular="0.2" shininess="0.3"/>
        <material name="stair_nosing_mat" rgba="0.96 0.78 0.08 1.0" specular="0.6" shininess="0.8"/>
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
            {''.join(bars_xml)}
        </body>
    </worldbody>

    <actuator>
        {''.join(actuators_xml)}
    </actuator>
</mujoco>
"""
    return xml_str, dirs
