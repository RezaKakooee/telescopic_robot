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

from . import playground_course as PC
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

            def placed(x, y):
                """Position rotated by the flight yaw about its start; plus the euler attribute."""
                if abs(yaw) < 1e-9:
                    return f'pos="{x:.4f} {y:.4f}', ""
                c, si = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
                dx, dy = x - start_x, y - start_y
                return f'pos="{start_x + c * dx - si * dy:.4f} {start_y + si * dx + c * dy:.4f}', f'euler="0 0 {yaw:.2f}" '

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
                    f'{placed(step_x, step_y)[0]} {step_h / 2.0:.4f}" {placed(step_x, step_y)[1]}'
                    f'size="{run / 2.0:.4f} {wid / 2.0:.4f} {step_h / 2.0:.4f}" '
                    f'material="{tread_mat}" friction="1.35 0.02 0.005" condim="4" priority="1"/>'
                )
                # Preserve the original physical nosing exactly: even this
                # small contact strip affects which rods support a landing.
                walls_xml.append(
                    f'<geom name="stair_nosing_{st_idx}_{step_i}" type="box" '
                    f'{placed(step_x - run/2.0 + 0.015, step_y)[0]} {step_h - 0.003:.4f}" {placed(step_x, step_y)[1]}'
                    f'size="0.015 {wid / 2.0 * 0.99:.4f} 0.003" '
                    f'material="stair_nosing_mat" friction="1.2 0.01 0.001" condim="3"/>'
                )
                # Add a separate, raised visual-only safety band. The larger
                # width and offset avoid the old coplanar z-fighting, while
                # disabled collision bits leave the calibrated physics alone.
                walls_xml.append(
                    f'<geom name="stair_nosing_visual_{st_idx}_{step_i}" type="box" '
                    f'{placed(step_x - run/2.0 + 0.045, step_y)[0]} {step_h + 0.004:.4f}" {placed(step_x, step_y)[1]}'
                    f'size="0.045 {wid / 2.0 * 0.99:.4f} 0.004" '
                    f'material="stair_nosing_mat" contype="0" conaffinity="0"/>'
                )
                # Matching vertical band makes the edge readable from the
                # low oblique follow camera as well as from above.
                walls_xml.append(
                    f'<geom name="stair_riser_band_{st_idx}_{step_i}" type="box" '
                    f'{placed(step_x - run/2.0 - 0.004, step_y)[0]} {max(step_h - 0.045, 0.045):.4f}" {placed(step_x, step_y)[1]}'
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
            yaw_deg = float(pipe.yaw_deg) if getattr(pipe, "yaw_deg", None) is not None else 0.0

            def placed(x, y, roll_deg):
                """Position and orientation attributes for a pipe part, rotated by the pipe yaw."""
                if abs(yaw_deg) < 1e-9:
                    return f'pos="{x:.4f} {y:.4f}"', f'euler="{roll_deg:.1f} 0 0"'
                yaw = np.radians(yaw_deg)
                dx, dy = x - start_x, y - start_y
                rx = start_x + dx * np.cos(yaw) - dy * np.sin(yaw)
                ry = start_y + dx * np.sin(yaw) + dy * np.cos(yaw)
                # quat = Rz(yaw) * Rx(roll)
                cz_, sz_ = np.cos(yaw / 2), np.sin(yaw / 2)
                cx_, sx_ = np.cos(np.radians(roll_deg) / 2), np.sin(np.radians(roll_deg) / 2)
                q = (cz_ * cx_, cz_ * sx_, sz_ * sx_, sz_ * cx_)  # (w, x, y, z)
                return f'pos="{rx:.4f} {ry:.4f}"', f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"'

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

                f_pos, f_rot = placed(cx, fy, -angle_deg)
                walls_xml.append(
                    f'<geom name="glass_facet_{p_idx}_{fi}" type="box" '
                    f'{f_pos[:-1]} {fz:.4f}" '
                    f'size="{p_len / 2.0:.4f} {facet_w / 2.0:.4f} {facet_th / 2.0:.4f}" '
                    f'{f_rot} material="glass_pipe_mat" '
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
                    c_pos, c_rot = placed(rx, r_pos_y, -r_deg)
                    walls_xml.append(
                        f'<geom name="pipe_collar_{p_idx}_{ri}_{rfi}" type="box" '
                        f'{c_pos[:-1]} {r_pos_z:.4f}" '
                        f'size="0.025 {facet_w / 2.0 * 1.05:.4f} 0.012" '
                        f'{c_rot} material="pipe_ring_mat" '
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


def campus_features_xml(scenario) -> list[str]:
    """Realistic University Campus architectural assets and landscape features.

    Clean North-South Central Boulevard:
      - 3.0m wide continuous paved asphalt promenade from South (0, -8.5) to North (0, 8.5)
      - Beveled granite curb edgings along both sides (x = -1.5 and x = +1.5)
      - Paved paver entrance quads at South spawn and North goal
      - Solid 3D Science & Engineering Hall (West) with ribbon glass and entrance portico
      - Solid 3D University Library (East) with grand glass atrium and mullion fins
      - Avenue shade trees planted along lawns at x = +/- 2.8m
      - Minimalist streetlamps sited strictly along curb edges at x = +/- 1.65m (zero path collisions)
      - Wooden park benches on lawns facing the promenade (x = +/- 2.2m)
      - Contiguous stainless-steel handrails along the entire stairs, terrace, and ramp flight
      - Perpendicular utility pipe hazard line & curb mounting brackets
    """
    if getattr(scenario, "kind", "") not in ("campus", "university_campus"):
        return []

    xml: list[str] = []

    # =========================================================================
    # 1. Main Promenade Pavement & Curbs (Clean North-South Axis)
    # =========================================================================
    # 1a. Central Asphalt Promenade (x in [-1.5, 1.5], y in [-8.5, 8.5], 3.0m wide)
    xml.append(
        '<geom name="walkway_promenade" type="box" pos="0.0000 0.0000 0.0030" '
        'size="1.5000 8.5000 0.0030" material="campus_asphalt_mat" '
        'friction="0.95 0.015 0.005" condim="4"/>'
    )
    # 1b. Granite Curb Edges along the promenade flanks
    xml.append(
        '<geom name="curb_west" type="box" pos="-1.5200 0.0000 0.0100" '
        'size="0.0400 8.5000 0.0100" material="campus_curb_mat" condim="3"/>'
    )
    xml.append(
        '<geom name="curb_east" type="box" pos="1.5200 0.0000 0.0100" '
        'size="0.0400 8.5000 0.0100" material="campus_curb_mat" condim="3"/>'
    )

    # 1c. South Spawn Entrance Quad (Paved Granite Pavers)
    xml.append(
        '<geom name="plaza_south" type="box" pos="0.0000 -8.0000 0.0040" '
        'size="2.2000 1.5000 0.0040" material="campus_paver_mat" '
        'friction="0.95 0.015 0.005" condim="4"/>'
    )
    # 1d. North Goal Finish Quad (Paved Granite Pavers)
    xml.append(
        '<geom name="plaza_north" type="box" pos="0.0000 8.0000 0.0040" '
        'size="2.2000 1.5000 0.0040" material="campus_paver_mat" '
        'friction="0.95 0.015 0.005" condim="4"/>'
    )

    # =========================================================================
    # 2. 3D Architectural Buildings (Solid Masses with Windows, Facades, Canopies)
    # =========================================================================
    # -------------------------------------------------------------------------
    # Building 1: Science & Engineering Hall (West: x in [-9.0, -3.5], y in [-2.0, 7.0])
    # Center: (-6.25, 2.50), size: (2.75, 4.50, 2.40), height: 4.8m
    # -------------------------------------------------------------------------
    xml.append(
        '<geom name="bldg1_plinth" type="box" pos="-6.2500 2.5000 0.2500" '
        'size="2.8000 4.5500 0.2500" material="stone_facade_mat" condim="3"/>'
    )
    xml.append(
        '<geom name="bldg1_body" type="box" pos="-6.2500 2.5000 2.4000" '
        'size="2.7500 4.5000 1.9000" material="brick_wall_mat" condim="3"/>'
    )
    xml.append(
        '<geom name="bldg1_roof" type="box" pos="-6.2500 2.5000 4.3500" '
        'size="2.8200 4.5800 0.0800" material="roof_trim_mat" condim="3"/>'
    )
    # Window ribbons facing the central boulevard
    xml.append(
        '<geom name="bldg1_win_1f" type="box" pos="-3.4800 2.5000 1.4000" '
        'size="0.0200 3.8000 0.4000" material="glass_window_mat" contype="0" conaffinity="0"/>'
    )
    xml.append(
        '<geom name="bldg1_win_2f" type="box" pos="-3.4800 2.5000 3.0000" '
        'size="0.0200 3.8000 0.4000" material="glass_window_mat" contype="0" conaffinity="0"/>'
    )
    # Entrance canopy portico
    xml.append(
        '<geom name="bldg1_canopy" type="box" pos="-3.0000 2.5000 2.2000" '
        'size="0.6000 1.2000 0.0300" material="glass_window_mat" contype="0" conaffinity="0"/>'
    )
    xml.append(
        '<geom name="bldg1_col_a" type="cylinder" pos="-2.5000 1.4000 1.1000" '
        'size="0.0400 1.1000" material="lamp_pole_mat" condim="3"/>'
    )
    xml.append(
        '<geom name="bldg1_col_b" type="cylinder" pos="-2.5000 3.6000 1.1000" '
        'size="0.0400 1.1000" material="lamp_pole_mat" condim="3"/>'
    )

    # -------------------------------------------------------------------------
    # Building 2: University Library & Student Center (East: x in [3.5, 9.0], y in [-7.0, 2.0])
    # Center: (6.25, -2.50), size: (2.75, 4.50, 2.40), height: 4.8m
    # -------------------------------------------------------------------------
    xml.append(
        '<geom name="bldg2_plinth" type="box" pos="6.2500 -2.5000 0.2500" '
        'size="2.8000 4.5500 0.2500" material="roof_trim_mat" condim="3"/>'
    )
    xml.append(
        '<geom name="bldg2_body" type="box" pos="6.2500 -2.5000 2.4000" '
        'size="2.7500 4.5000 1.9000" material="stone_facade_mat" condim="3"/>'
    )
    xml.append(
        '<geom name="bldg2_roof" type="box" pos="6.2500 -2.5000 4.3500" '
        'size="2.8200 4.5800 0.0800" material="roof_trim_mat" condim="3"/>'
    )
    # Grand glass atrium wall facing the boulevard
    xml.append(
        '<geom name="bldg2_atrium_glass" type="box" pos="3.4800 -2.5000 2.4000" '
        'size="0.0200 3.8000 1.8000" material="glass_window_mat" contype="0" conaffinity="0"/>'
    )
    # Vertical bronze architectural mullion fins
    for m_i, m_y in enumerate([-5.5, -4.0, -2.5, -1.0, 0.5]):
        xml.append(
            f'<geom name="bldg2_mullion_{m_i}" type="box" pos="3.4700 {m_y:.4f} 2.4000" '
            f'size="0.0350 0.0350 1.8000" material="window_frame_mat" contype="0" conaffinity="0"/>'
        )
    # Library entrance canopy
    xml.append(
        '<geom name="bldg2_canopy" type="box" pos="3.0000 -2.5000 2.2000" '
        'size="0.6000 1.2000 0.0300" material="glass_window_mat" contype="0" conaffinity="0"/>'
    )
    xml.append(
        '<geom name="bldg2_col_a" type="cylinder" pos="2.5000 -1.4000 1.1000" '
        'size="0.0400 1.1000" material="lamp_pole_mat" condim="3"/>'
    )
    xml.append(
        '<geom name="bldg2_col_b" type="cylinder" pos="2.5000 -3.6000 1.1000" '
        'size="0.0400 1.1000" material="lamp_pole_mat" condim="3"/>'
    )

    # =========================================================================
    # 3. Avenue Shade Trees (Neatly lined along lawns at x = +/- 2.8m)
    # =========================================================================
    for t_i, ty in enumerate([-6.5, -3.5, 0.0, 3.5, 6.5]):
        # West tree
        xml.append(
            f'<geom name="tree_trunk_w_{t_i}" type="cylinder" pos="-2.8000 {ty:.4f} 0.7500" '
            f'size="0.1300 0.7500" material="tree_trunk_mat" friction="0.9 0.01 0.001" condim="3"/>'
        )
        xml.append(
            f'<geom name="tree_crown_lower_w_{t_i}" type="sphere" pos="-2.8000 {ty:.4f} 2.1000" '
            f'size="1.2000" material="tree_foliage_mat" contype="0" conaffinity="0"/>'
        )
        xml.append(
            f'<geom name="tree_crown_upper_w_{t_i}" type="sphere" pos="-2.8000 {ty:.4f} 2.8000" '
            f'size="0.9000" material="tree_foliage_light_mat" contype="0" conaffinity="0"/>'
        )
        # East tree
        xml.append(
            f'<geom name="tree_trunk_e_{t_i}" type="cylinder" pos="2.8000 {ty:.4f} 0.7500" '
            f'size="0.1300 0.7500" material="tree_trunk_mat" friction="0.9 0.01 0.001" condim="3"/>'
        )
        xml.append(
            f'<geom name="tree_crown_lower_e_{t_i}" type="sphere" pos="2.8000 {ty:.4f} 2.1000" '
            f'size="1.2000" material="tree_foliage_mat" contype="0" conaffinity="0"/>'
        )
        xml.append(
            f'<geom name="tree_crown_upper_e_{t_i}" type="sphere" pos="2.8000 {ty:.4f} 2.8000" '
            f'size="0.9000" material="tree_foliage_light_mat" contype="0" conaffinity="0"/>'
        )

    # =========================================================================
    # 4. Pathway Streetlamps (Strictly Sited on Curb Edge at x = +/- 1.65m)
    # =========================================================================
    for l_i, ly in enumerate([-5.0, -1.0, 3.0, 7.0]):
        # West curb streetlamp
        xml.append(
            f'<geom name="lamp_pole_w_{l_i}" type="cylinder" pos="-1.6500 {ly:.4f} 1.2500" '
            f'size="0.0350 1.2500" material="lamp_pole_mat" condim="3"/>'
        )
        xml.append(
            f'<geom name="lamp_arm_w_{l_i}" type="capsule" fromto="-1.6500 {ly:.4f} 2.5000 -1.3500 {ly:.4f} 2.5000" '
            f'size="0.0220" material="lamp_pole_mat" contype="0" conaffinity="0"/>'
        )
        xml.append(
            f'<geom name="lamp_glow_w_{l_i}" type="sphere" pos="-1.3500 {ly:.4f} 2.4400" '
            f'size="0.0750" material="lamp_glow_mat" contype="0" conaffinity="0"/>'
        )
        # East curb streetlamp
        xml.append(
            f'<geom name="lamp_pole_e_{l_i}" type="cylinder" pos="1.6500 {ly:.4f} 1.2500" '
            f'size="0.0350 1.2500" material="lamp_pole_mat" condim="3"/>'
        )
        xml.append(
            f'<geom name="lamp_arm_e_{l_i}" type="capsule" fromto="1.6500 {ly:.4f} 2.5000 1.3500 {ly:.4f} 2.5000" '
            f'size="0.0220" material="lamp_pole_mat" contype="0" conaffinity="0"/>'
        )
        xml.append(
            f'<geom name="lamp_glow_e_{l_i}" type="sphere" pos="1.3500 {ly:.4f} 2.4400" '
            f'size="0.0750" material="lamp_glow_mat" contype="0" conaffinity="0"/>'
        )

    # =========================================================================
    # 5. Teak Wood Park Benches (Sited on Lawns at x = +/- 2.2m Facing Promenade)
    # =========================================================================
    for b_i, by in enumerate([-4.5, 4.5]):
        # West bench (faces East toward promenade)
        xml.append(
            f'<geom name="bench_w_seat_{b_i}" type="box" pos="-2.2000 {by:.4f} 0.2200" '
            f'size="0.2200 0.6500 0.0250" material="bench_slats_mat" condim="3"/>'
        )
        xml.append(
            f'<geom name="bench_w_back_{b_i}" type="box" pos="-2.4000 {by:.4f} 0.4200" '
            f'size="0.0250 0.6500 0.1600" material="bench_slats_mat" condim="3"/>'
        )
        # East bench (faces West toward promenade)
        xml.append(
            f'<geom name="bench_e_seat_{b_i}" type="box" pos="2.2000 {by:.4f} 0.2200" '
            f'size="0.2200 0.6500 0.0250" material="bench_slats_mat" condim="3"/>'
        )
        xml.append(
            f'<geom name="bench_e_back_{b_i}" type="box" pos="2.4000 {by:.4f} 0.4200" '
            f'size="0.0250 0.6500 0.1600" material="bench_slats_mat" condim="3"/>'
        )

    # =========================================================================
    # 5. Grand Terrace Station: 3-Step Stairs, Elevated Deck & Descent Ramp
    # =========================================================================
    # 5a. 3-Step Stairs (Full 3.0m promenade width, ascending along +Y from y=1.0 to 1.75)
    # Step 1: y in [1.00, 1.25], height 0.05m
    xml.append(
        '<geom name="campus_stair_0" type="box" pos="0.0000 1.1250 0.0250" '
        'size="1.5000 0.1250 0.0250" material="stair_tread_blue_mat" '
        'friction="1.35 0.02 0.005" condim="4" priority="1"/>'
    )
    xml.append(
        '<geom name="campus_stair_nosing_0" type="box" pos="0.0000 1.0150 0.0480" '
        'size="1.4900 0.0150 0.0030" material="stair_nosing_mat" friction="1.2 0.01 0.001" condim="3"/>'
    )
    # Step 2: y in [1.25, 1.50], height 0.10m
    xml.append(
        '<geom name="campus_stair_1" type="box" pos="0.0000 1.3750 0.0500" '
        'size="1.5000 0.1250 0.0500" material="stair_tread_teal_mat" '
        'friction="1.35 0.02 0.005" condim="4" priority="1"/>'
    )
    xml.append(
        '<geom name="campus_stair_nosing_1" type="box" pos="0.0000 1.2650 0.0980" '
        'size="1.4900 0.0150 0.0030" material="stair_nosing_mat" friction="1.2 0.01 0.001" condim="3"/>'
    )
    # Step 3: y in [1.50, 1.75], height 0.15m
    xml.append(
        '<geom name="campus_stair_2" type="box" pos="0.0000 1.6250 0.0750" '
        'size="1.5000 0.1250 0.0750" material="stair_tread_blue_mat" '
        'friction="1.35 0.02 0.005" condim="4" priority="1"/>'
    )
    xml.append(
        '<geom name="campus_stair_nosing_2" type="box" pos="0.0000 1.5150 0.1480" '
        'size="1.4900 0.0150 0.0030" material="stair_nosing_mat" friction="1.2 0.01 0.001" condim="3"/>'
    )

    # 5b. Elevated Terrace Deck (z = +0.15m, y in [1.75, 3.50], 3.0m wide)
    xml.append(
        '<geom name="campus_terrace_deck" type="box" pos="0.0000 2.6250 0.0750" '
        'size="1.5000 0.8750 0.0750" material="campus_paver_mat" '
        'friction="1.25 0.02 0.005" condim="4" priority="1"/>'
    )

    # 5c. Smooth Descent Ramp (y in [3.50, 4.25], sloping down from z=0.15 to z=0.0)
    xml.append(
        '<geom name="campus_descent_ramp" type="box" pos="0.0000 3.8750 0.0750" '
        'size="1.5000 0.3824 0.0200" euler="11.31 0 0" material="campus_paver_mat" '
        'friction="1.25 0.02 0.005" condim="4" priority="1"/>'
    )

    # =========================================================================
    # 6. Contiguous Stainless-Steel Handrails (Spanning Stairs, Terrace, and Ramp)
    # =========================================================================
    # Flanks at x = -1.52 and x = +1.52, covering y from 1.0 to 4.25
    for s_side, sx in [("west", -1.52), ("east", 1.52)]:
        # Segment 1: Slanted handrail over 3-step stairs (y: 1.0 -> 1.75, z: 0.85 -> 1.00)
        xml.append(
            f'<geom name="handrail_stair_{s_side}" type="capsule" '
            f'fromto="{sx:.4f} 1.0000 0.8500 {sx:.4f} 1.7500 1.0000" '
            f'size="0.0250" material="handrail_steel_mat" condim="3"/>'
        )
        # Segment 2: Level handrail over terrace platform (y: 1.75 -> 3.50, z: 1.00)
        xml.append(
            f'<geom name="handrail_deck_{s_side}" type="capsule" '
            f'fromto="{sx:.4f} 1.7500 1.0000 {sx:.4f} 3.5000 1.0000" '
            f'size="0.0250" material="handrail_steel_mat" condim="3"/>'
        )
        # Segment 3: Slanted handrail over descent ramp (y: 3.50 -> 4.25, z: 1.00 -> 0.85)
        xml.append(
            f'<geom name="handrail_ramp_{s_side}" type="capsule" '
            f'fromto="{sx:.4f} 3.5000 1.0000 {sx:.4f} 4.2500 0.8500" '
            f'size="0.0250" material="handrail_steel_mat" condim="3"/>'
        )
        # Vertical stanchion posts
        for p_i, py in enumerate([1.0, 1.75, 2.625, 3.5, 4.25]):
            pz = 1.0 if (1.75 <= py <= 3.5) else (0.85 if (py == 1.0 or py == 4.25) else 0.925)
            xml.append(
                f'<geom name="rail_post_{s_side}_{p_i}" type="cylinder" '
                f'pos="{sx:.4f} {py:.4f} {pz / 2:.4f}" size="0.0220 {pz / 2:.4f}" '
                f'material="handrail_steel_mat" condim="3"/>'
            )

    # =========================================================================
    # 7. Perpendicular Utility Pipe Hazard Line & Flanges (at y = -3.8)
    # =========================================================================
    # Clean, straight yellow warning stripe painted across the promenade
    xml.append(
        '<geom name="pipe_hazard_line" type="box" pos="0.0000 -4.1000 0.0050" '
        'size="1.5000 0.1200 0.0030" material="stair_nosing_mat" contype="0" conaffinity="0"/>'
    )
    # Flanged mounting brackets on curb ends
    xml.append(
        '<geom name="pipe_bracket_west" type="box" pos="-1.5500 -3.8000 0.0450" '
        'size="0.0800 0.0600 0.0450" material="pipe_ring_mat" condim="3"/>'
    )
    xml.append(
        '<geom name="pipe_bracket_east" type="box" pos="1.5500 -3.8000 0.0450" '
        'size="0.0800 0.0600 0.0450" material="pipe_ring_mat" condim="3"/>'
    )

    # =========================================================================
    # 8. Recessed Maintenance Pit Frame (at x = 0.55, y = -1.5)
    # =========================================================================
    xml.append(
        '<geom name="pit_frame" type="box" pos="0.5500 -1.5000 0.0040" '
        'size="0.3800 0.4800 0.0030" material="pipe_ring_mat" contype="0" conaffinity="0"/>'
    )

    # =========================================================================
    # 9. Wooden Delivery Crates (Off-center near Loading Bay at y = 5.5, West side)
    # =========================================================================
    xml.append(
        '<geom name="crate_1" type="box" pos="-0.8500 5.5000 0.2500" '
        'size="0.3500 0.3500 0.2500" material="wood_crate_mat" '
        'friction="1.0 0.01 0.001" condim="4" priority="1"/>'
    )
    xml.append(
        '<geom name="crate_2" type="box" pos="-0.8500 6.2500 0.1800" '
        'size="0.2500 0.2500 0.1800" material="wood_crate_mat" '
        'friction="1.0 0.01 0.001" condim="4" priority="1"/>'
    )

    return xml


def playground_features_xml(scenario) -> list[str]:
    """Robotics Testing Playground: Multi-Turn Street Course with 3-Box Parkour & Deep Valleys.

    Features:
      - 2.4m wide street corridors with 3 dedicated 90-degree corner turns
      - Station 1: Launch sprint runway (Leg 1, East)
      - Station 2: Hurdle leap at x=4.5m, height 0.18m
      - Station 3: 3-Box Platform Parkour across 2 deep valleys (Leg 2, North, y in [2.2, 6.7])
                   Box 1 (+0.24m) -> Valley 1 (-0.5m) -> Box 2 (+0.24m) -> Valley 2 (-0.5m) -> Box 3 (+0.24m)
      - Station 4: 3-Step Stairs & Elevated Deck (Leg 3, West, x in [7.0, 3.5])
      - Station 5: Rough cobblestone suspension bed (Leg 4, North, y in [10.5, 12.8])
      - Station 6: 0.20 m jump wall across the corridor (Leg 4, North, y = 14.2)
      - Station 7: Deceleration target bullseye and finish goal beacon at (0.0, 16.5)
      - Perimeter acrylic guide walls with high-visibility safety top rails
    """
    if getattr(scenario, "kind", "") not in ("playground", "robotics_playground", "proving_ground"):
        return []

    xml: list[str] = []
    hw = 1.20  # corridor half-width

    # =========================================================================
    # 1. Asphalt Street Pavement Corridors (Legs 1, 2, 3, 4)
    # =========================================================================
    # Leg 1: East corridor (x in [-1.2, 10.2], y in [-1.2, 1.2])
    xml.append(
        '<geom name="play_pave_leg1" type="box" pos="4.5000 0.0000 0.0020" '
        'size="5.7000 1.2000 0.0020" material="playground_runway_mat" '
        'friction="1.2 0.015 0.005" condim="4"/>'
    )
    # Leg 2: North corridor (x in [7.8, 10.2], y in [-1.2, 10.2])
    xml.append(
        '<geom name="play_pave_leg2" type="box" pos="9.0000 4.5000 0.0020" '
        'size="1.2000 5.7000 0.0020" material="playground_runway_mat" '
        'friction="1.2 0.015 0.005" condim="4"/>'
    )
    # Leg 3: West corridor (x in [-1.2, 10.2], y in [7.8, 10.2])
    xml.append(
        '<geom name="play_pave_leg3" type="box" pos="4.5000 9.0000 0.0020" '
        'size="5.7000 1.2000 0.0020" material="playground_runway_mat" '
        'friction="1.2 0.015 0.005" condim="4"/>'
    )
    # Leg 4: North finish corridor (x in [-1.2, 1.2], y in [7.8, 17.5])
    xml.append(
        '<geom name="play_pave_leg4" type="box" pos="0.0000 13.0000 0.0020" '
        'size="1.2000 4.8000 0.0020" material="playground_runway_mat" '
        'friction="1.2 0.015 0.005" condim="4"/>'
    )

    # =========================================================================
    # 2. Station 1: Launch Pad on Leg 1 (x in [0.0, 0.6])
    # =========================================================================
    xml.append(
        '<geom name="play_launch_pad" type="box" pos="0.3000 0.0000 0.0040" '
        'size="0.3000 1.1000 0.0020" material="launch_pad_mat" contype="0" conaffinity="0"/>'
    )

    # =========================================================================
    # 3. Station 2: Hurdle Jump Station (x = 4.5)
    # =========================================================================
    # Takeoff pad
    xml.append(
        '<geom name="hurdle_takeoff_pad" type="box" pos="4.0000 0.0000 0.0040" '
        'size="0.3000 1.1000 0.0020" material="launch_pad_mat" contype="0" conaffinity="0"/>'
    )
    # Upright support posts at street flanks
    for h_side, hy in [("north", hw), ("south", -hw)]:
        xml.append(
            f'<geom name="hurdle_base_{h_side}" type="cylinder" pos="4.5000 {hy:.4f} 0.0150" '
            f'size="0.0900 0.0150" material="bench_iron_mat" condim="3"/>'
        )
        xml.append(
            f'<geom name="hurdle_post_{h_side}" type="cylinder" pos="4.5000 {hy:.4f} 0.1200" '
            f'size="0.0300 0.1200" material="stair_nosing_mat" condim="3"/>'
        )
    # Horizontal cross-bar spanning across the lane at height 0.18m
    xml.append(
        '<geom name="hurdle_crossbar" type="cylinder" pos="4.5000 0.0000 0.1800" '
        'size="0.0350 1.1800" euler="90 0 0" material="stair_nosing_mat" '
        'friction="0.8 0.005 0.0001" condim="4" priority="1"/>'
    )

    # Turn 1 Warning Chevron at Corner 1 (9.0, 0.0)
    xml.append(
        '<geom name="turn1_chevron" type="box" pos="8.5000 0.0000 0.0040" '
        'size="0.0400 1.0000 0.0020" material="stair_nosing_mat" contype="0" conaffinity="0"/>'
    )

    # =========================================================================
    # 4. Station 3: The 3-Box Platform Parkour & Deep Valleys (Leg 2, x=9.0)
    # =========================================================================
    # Three boxes with valleys between them; sizes from playground_course.
    for bi in range(3):
        y0, y1, cy, bh = PC.BOX_Y0[bi], PC.BOX_Y1[bi], PC.BOX_CENTER_Y[bi], PC.BOX_HEIGHTS[bi]
        xml.append(
            f'<geom name="parkour_box_{bi + 1}" type="box" pos="{PC.LEG2_X:.4f} {cy:.4f} {bh / 2:.4f}" '
            f'size="{PC.BOX_HALF_W:.4f} {PC.BOX_LEN / 2:.4f} {bh / 2:.4f}" material="ramp_mat" '
            'friction="1.4 0.02 0.005" condim="4" priority="1"/>'
        )
        # Safety nosing strips on the front and rear top edges
        for tag, ny in (("front", y0 + 0.015), ("rear", y1 - 0.015)):
            xml.append(
                f'<geom name="box{bi + 1}_nosing_{tag}" type="box" pos="{PC.LEG2_X:.4f} {ny:.4f} {bh - 0.002:.4f}" '
                f'size="{PC.BOX_HALF_W - 0.02:.4f} 0.0150 0.0040" material="stair_nosing_mat" '
                'friction="1.2 0.01 0.001" condim="3"/>'
            )

    # Padded Landing Zone after Box 3
    xml.append(
        f'<geom name="box3_landing_pad" type="box" pos="{PC.LEG2_X:.4f} {PC.BOX_Y1[2] + 0.5:.4f} 0.0040" '
        'size="1.1000 0.4000 0.0020" material="launch_pad_mat" contype="0" conaffinity="0"/>'
    )

    # Turn 2 Warning Chevron at Corner 2 (9.0, 9.0)
    xml.append(
        '<geom name="turn2_chevron" type="box" pos="9.0000 8.5000 0.0040" '
        'size="1.0000 0.0400 0.0020" material="stair_nosing_mat" contype="0" conaffinity="0"/>'
    )

    # =========================================================================
    # =========================================================================
    # 5. Station 4: High 3-Step Stairs & Elevated Deck (Leg 3, y=9.0, heading West)
    # =========================================================================
    # Steps ascending West from the ground to STAIR_TOP; sizes from playground_course.
    for si in range(PC.STAIR_N):
        sx = PC.STAIR_X0 - si * PC.STAIR_RUN
        sz = (si + 1) * PC.STAIR_RISE
        t_mat = "stair_tread_blue_mat" if si % 2 == 0 else "stair_tread_teal_mat"
        xml.append(
            f'<geom name="play3_stair_{si}" type="box" pos="{sx - PC.STAIR_RUN / 2:.4f} {PC.LEG3_Y:.4f} {sz / 2:.4f}" '
            f'size="{PC.STAIR_RUN / 2:.4f} 1.2000 {sz / 2:.4f}" material="{t_mat}" '
            f'friction="1.35 0.02 0.005" condim="4" priority="1"/>'
        )
        xml.append(
            f'<geom name="play3_nosing_{si}" type="box" pos="{sx - 0.0150:.4f} {PC.LEG3_Y:.4f} {sz - 0.003:.4f}" '
            f'size="0.0150 1.1800 0.0030" material="stair_nosing_mat" friction="1.2 0.01 0.001" condim="3"/>'
        )
    # Elevated deck at the stair top height
    deck_c = (PC.DECK_X0 + PC.DECK_X1) / 2
    xml.append(
        f'<geom name="play3_deck" type="box" pos="{deck_c:.4f} {PC.LEG3_Y:.4f} {PC.STAIR_TOP / 2:.4f}" '
        f'size="{(PC.DECK_X1 - PC.DECK_X0) / 2:.4f} 1.2000 {PC.STAIR_TOP / 2:.4f}" material="campus_paver_mat" '
        'friction="1.25 0.02 0.005" condim="4" priority="1"/>'
    )
    # Descent ramp from the deck down to the ground
    ramp_len = float(np.hypot(PC.RAMP_X1 - PC.RAMP_X0, PC.STAIR_TOP))
    xml.append(
        f'<geom name="play3_ramp" type="box" pos="{(PC.RAMP_X0 + PC.RAMP_X1) / 2:.4f} {PC.LEG3_Y:.4f} {PC.STAIR_TOP / 2:.4f}" '
        f'size="{ramp_len / 2:.4f} 1.2000 0.0200" euler="0 {-PC.RAMP_PITCH_DEG:.2f} 0" material="campus_paver_mat" '
        'friction="1.25 0.02 0.005" condim="4" priority="1"/>'
    )
    # Handrails flanking Leg 3 stairs & deck at y = 9.0 +/- 1.22m, elevated for 0.30m deck
    for s_side, sy in [("north", 9.0 + hw), ("south", 9.0 - hw)]:
        xml.append(
            f'<geom name="play3_rail_{s_side}" type="capsule" '
            f'fromto="7.0000 {sy:.4f} 1.0000 3.5000 {sy:.4f} 1.0000" '
            f'size="0.0250" material="handrail_steel_mat" condim="3"/>'
        )

    # Turn 3 Warning Chevron at Corner 3 (0.0, 9.0)
    xml.append(
        '<geom name="turn3_chevron" type="box" pos="0.5000 9.0000 0.0040" '
        'size="0.0400 1.0000 0.0020" material="stair_nosing_mat" contype="0" conaffinity="0"/>'
    )

    # =========================================================================
    # 6. Station 5 & 6: Rough Bed & Low Conduit (Leg 4, x=0.0, heading North)
    # =========================================================================
    # Rough Cobblestone Bed: y in [10.5, 12.8]
    xml.append(
        '<geom name="gravel_border_s" type="box" pos="0.0000 10.4500 0.0150" '
        'size="1.1500 0.0400 0.0150" material="pipe_ring_mat" condim="3"/>'
    )
    xml.append(
        '<geom name="gravel_border_n" type="box" pos="0.0000 12.8500 0.0080" '
        'size="1.1500 0.0400 0.0080" material="pipe_ring_mat" condim="3"/>'
    )

    # Station 6: Jump Wall (Leg 4, x=0). A solid wall across the
    # corridor. The only way past is a forward jump; touching it counts as a hit
    # (the "barrier_" prefix, see MujocoRadialSphereEnv.HIT_GEOM_PREFIXES).
    xml.append(
        f'<geom name="barrier_wall_s6" type="box" pos="0.0000 {PC.WALL_Y:.4f} {PC.WALL_H / 2:.4f}" '
        f'size="1.1800 {PC.WALL_HALF_T:.4f} {PC.WALL_H / 2:.4f}" material="stair_nosing_mat" condim="3"/>'
    )
    # Takeoff pad in front of the wall
    xml.append(
        f'<geom name="wall_takeoff_pad" type="box" pos="0.0000 {PC.WALL_Y - 0.5:.4f} 0.0040" '
        'size="1.1000 0.3000 0.0020" material="launch_pad_mat" contype="0" conaffinity="0"/>'
    )

    # =========================================================================
    # 7. Station 7: Deceleration Zone & Pure Green Goal Target Pad (0.0, 16.5)
    # =========================================================================
    xml.append(
        '<geom name="brake_pad" type="box" pos="0.0000 15.5000 0.0035" '
        'size="1.1000 0.8000 0.0020" material="brake_checker_mat" contype="0" conaffinity="0"/>'
    )
    # Clean emerald-green goal bullseye and pad (no red)
    xml.append(
        '<geom name="target_green_ring" type="cylinder" pos="0.0000 16.5000 0.0040" '
        'size="0.7500 0.0020" material="goal_pad_mat" contype="0" conaffinity="0"/>'
    )
    xml.append(
        '<geom name="target_bullseye" type="cylinder" pos="0.0000 16.5000 0.0050" '
        'size="0.3500 0.0020" material="goal_mat" contype="0" conaffinity="0"/>'
    )

    # =========================================================================
    # 8. Perimeter Guide Walls (h = 0.40m, transparent acrylic + yellow rail)
    # =========================================================================
    # Define bounding wall segments around the 2.4m corridor
    wall_segs = [
        # Outer Boundary Segments
        ((-1.2, -hw), (9.0 + hw, -hw)),
        ((9.0 + hw, -hw), (9.0 + hw, 9.0 + hw)),
        ((9.0 + hw, 9.0 + hw), (0.0 + hw, 9.0 + hw)),
        ((0.0 + hw, 9.0 + hw), (0.0 + hw, 17.5)),
        ((0.0 + hw, 17.5), (-hw, 17.5)),
        ((-hw, 17.5), (-hw, 9.0 - hw)),
        ((-hw, 9.0 - hw), (-1.2, 9.0 - hw)),
        ((-1.2, -hw), (-1.2, hw)),
        # Inner Corner Segments
        ((-1.2, hw), (9.0 - hw, hw)),
        ((9.0 - hw, hw), (9.0 - hw, 9.0 - hw)),
        ((9.0 - hw, 9.0 - hw), (-hw, 9.0 - hw)),
    ]

    for w_i, ((wx1, wy1), (wx2, wy2)) in enumerate(wall_segs):
        wcx, wcy = (wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0
        w_len = float(np.hypot(wx2 - wx1, wy2 - wy1))
        w_yaw = float(np.degrees(np.arctan2(wy2 - wy1, wx2 - wx1)))
        if w_len < 0.05:
            continue
        # Transparent acrylic panel
        xml.append(
            f'<geom name="wall_panel_{w_i}" type="box" pos="{wcx:.4f} {wcy:.4f} 0.2000" '
            f'size="{w_len / 2:.4f} 0.0150 0.2000" euler="0 0 {w_yaw:.2f}" '
            f'material="acrylic_wall_mat" condim="4"/>'
        )
        # Safety yellow top capping rail
        xml.append(
            f'<geom name="wall_rail_{w_i}" type="capsule" '
            f'fromto="{wx1:.4f} {wy1:.4f} 0.4100 {wx2:.4f} {wy2:.4f} 0.4100" '
            f'size="0.0220" material="stair_nosing_mat" condim="3"/>'
        )

    return xml



