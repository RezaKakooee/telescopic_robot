"""One XML builder per terrain feature.

``build_mujoco_scene_mjcf`` used to inline every one of these, which made it
eleven hundred lines long and impossible to exercise a single feature in
isolation. The thin-floor-slab bug lived in here unnoticed for exactly that
reason. Each function below takes the scenario and returns the geoms for its
own feature, so a test can build one feature and read the result.

Bodies are unchanged from the original inline blocks: same geom names, same
ordering, same formatting. The generated XML is byte-for-byte what it was.
"""
from __future__ import annotations

import numpy as np

from .terrain import (Cone, Gap, Motordrome, Pipe, Ramp, SandPatch, Staircase,
                      Step, StoneField, VerticalCylinder, Yardline, rows)


def steps_xml(scenario) -> list[str]:
    """4b. Ground Step / Rectangular Wooden Planks (passover obstacles)"""
    walls_xml: list[str] = []
    steps = getattr(scenario, "wood_planks", None)
    if steps is None or len(steps) == 0:
        steps = getattr(scenario, "steps", None)
    steps = rows(Step, steps)

    if steps:
        for s_idx, step in enumerate(steps):
            sx, sy, shx, shy, sh = (float(step.x), float(step.y), float(step.half_x),
                                    float(step.half_y), float(step.height))

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
    return walls_xml


def gaps_xml(scenario) -> list[str]:
    """4c. Floor Gaps / Holes / Pits in the ground"""
    walls_xml: list[str] = []
    # Each gap: (cx, cy, half_x, half_y, depth)
    gaps = rows(Gap, getattr(scenario, "gaps", None))
    if gaps:
        for g_idx, gap in enumerate(gaps):
            gx, gy, ghx, ghy = gap.x, gap.y, gap.half_x, gap.half_y
            gdepth = float(gap.depth)

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
    return walls_xml


def sand_patch_xml(scenario) -> list[str]:
    """4d. Sand Patches — high-friction rough terrain that slows the ball"""
    walls_xml: list[str] = []
    # Each sand_patch: (cx, cy, half_x, half_y)
    sand_patches = rows(SandPatch, getattr(scenario, "sand_patches", None))
    if sand_patches:
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
    return walls_xml


def stone_field_xml(scenario) -> list[str]:
    """4e. Scattered Mountainous Rocks, Boulders, and Stone Slabs on the floor"""
    walls_xml: list[str] = []
    # Each stone_zone: (cx, cy, half_x, half_y, n_stones, max_stone_size)
    stones = rows(StoneField, getattr(scenario, "stones", None))
    if stones:
        for st_idx, field in enumerate(stones):
            stx, sty, sthx, sthy = field.x, field.y, field.half_x, field.half_y
            n_stones = int(field.count)
            max_sz = float(field.max_size)
            is_circle = bool(field.circular)
            rng = np.random.RandomState(2026 + st_idx)
            
            rock_materials = ["granite_rock_mat", "slate_rock_mat", "sandstone_rock_mat", "basalt_rock_mat"]

            for si in range(n_stones):
                if is_circle:
                    r_sample = sthx * np.sqrt(rng.uniform(0.04, 0.95))
                    th_sample = rng.uniform(0.0, 2.0 * np.pi)
                    ox = stx + r_sample * np.cos(th_sample)
                    oy = sty + r_sample * np.sin(th_sample)
                else:
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
    return walls_xml


def ramp_xml(scenario) -> list[str]:
    """4f. Incline Slopes / Ramps (Uphill & Downhill)"""
    walls_xml: list[str] = []
    ramps = rows(Ramp, getattr(scenario, "ramps", None))
    if ramps:
        for r_idx, ramp in enumerate(ramps):
            rcx, rcy, rlen, rwid = (float(ramp.x), float(ramp.y),
                                    float(ramp.length), float(ramp.width))
            r_h, r_pitch, r_yaw = (float(ramp.height_change), float(ramp.pitch_deg),
                                   float(ramp.yaw_deg))
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
    return walls_xml


def staircase_xml(scenario) -> list[str]:
    """4g. Multi-Step Staircases"""
    walls_xml: list[str] = []
    staircases = rows(Staircase, getattr(scenario, "staircases", None))
    if staircases:
        for st_idx, flight in enumerate(staircases):
            start_x, start_y = float(flight.x), float(flight.y)
            n_steps = int(flight.n_steps)
            rise = float(flight.rise)
            run = float(flight.run)
            wid = float(flight.width)
            yaw = float(flight.yaw_deg)
            is_down = bool(flight.descending)

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
    return walls_xml


def pipe_xml(scenario) -> list[str]:
    """4h. Transparent Glass Pipe / Conduit (In-Pipe Crawling Inspection)"""
    walls_xml: list[str] = []
    pipes = rows(Pipe, getattr(scenario, "pipes", None))
    if pipes:
        for p_idx, pipe in enumerate(pipes):
            p_def = pipe
            start_x, start_y = float(p_def[0]), float(p_def[1])
            p_len = float(p_def[2])
            in_rad = float(p_def[3])
            out_rad = float(pipe.outer_radius)

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
    return walls_xml


def yardline_xml(scenario) -> list[str]:
    """4i. Athletic Runway Yardlines & Painted Distance Markers (Pure Visual, Zero Friction Obstruction)"""
    walls_xml: list[str] = []
    yardlines = rows(Yardline, getattr(scenario, "yardlines", None))
    if yardlines:
        for y_idx, line in enumerate(yardlines):
            yx, yy, yhx, yhy, y_rgba = (float(line.x), float(line.y), float(line.half_x),
                                        float(line.half_y), str(line.rgba))
            euler_str = ('' if line.yaw_deg is None
                         else f'euler="0 0 {float(line.yaw_deg):.2f}" ')
            walls_xml.append(
                f'<geom name="yardline_{y_idx}" type="box" '
                f'pos="{yx:.4f} {yy:.4f} 0.0015" {euler_str}'
                f'size="{yhx:.4f} {yhy:.4f} 0.0015" rgba="{y_rgba}" '
                f'contype="0" conaffinity="0"/>'
            )
    return walls_xml


def vertical_cylinder_xml(scenario) -> list[str]:
    """4j. Vertical Transparent Cylinders / Silos (Wall of Death / Spiral Vortex Climbing)"""
    walls_xml: list[str] = []
    vcyls = rows(VerticalCylinder, getattr(scenario, "vertical_cylinders", None))
    if vcyls:
        for vc_idx, cylinder in enumerate(vcyls):
            vc_def = cylinder
            cx, cy = float(vc_def[0]), float(vc_def[1])
            height = float(vc_def[2])
            in_rad = float(vc_def[3])
            out_rad = float(cylinder.outer_radius)

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
    return walls_xml


def motordrome_xml(scenario) -> list[str]:
    """4k. Authentic Wall of Death (Motordrome / Silodrome with 45-deg Base Apron Ramp)"""
    walls_xml: list[str] = []
    motordromes = rows(Motordrome, getattr(scenario, "motordromes", None))
    if motordromes:
        for md_idx, bowl in enumerate(motordromes):
            cx, cy = float(bowl.x), float(bowl.y)
            floor_r = float(bowl.floor_radius)
            wall_r = float(bowl.wall_radius)
            apron_h = float(bowl.apron_height)
            total_h = float(bowl.total_height)
            # Wall-of-death traction. The ball stays up only while
            # mu * v^2 / r >= g, so this number sets the arena's size.
            md_mu = float(bowl.friction)

            # More facets makes a rounder barrel; more profile rings makes a
            # smoother curve up the bank. Both are just geometry count.
            n_facets = int(bowl.facets) if bowl.facets else 32
            # How far each ring's planks reach past their own segment. Some
            # overlap is needed or the rings leave a seam the feet can catch.
            plank_lap = float(bowl.plank_overlap) if bowl.plank_overlap else 1.06
            plank_pad = float(bowl.plank_pad) if bowl.plank_pad is not None else 0.02
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
            profile = bowl.profile
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
    return walls_xml


def cone_xml(scenario) -> list[str]:
    """4k. Athletic / Traffic Training Cones (Slalom Course)"""
    walls_xml: list[str] = []
    cones_raw = getattr(scenario, "cones", None)
    if cones_raw is not None and len(cones_raw) > 0:
        cone_items = np.asarray(cones_raw, dtype=float)
        if cone_items.ndim == 1:
            cone_items = cone_items.reshape(1, -1)
        for c_idx, c_item in enumerate(cone_items):
            cone = Cone(*c_item) if len(c_item) > 2 else Cone(c_item[0], c_item[1])
            cx, cy, cr = float(cone.x), float(cone.y), float(cone.radius)
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
    return walls_xml

