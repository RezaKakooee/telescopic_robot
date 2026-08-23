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
        rod_to = tip0 * u
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
                      friction="{foot_friction}" condim="4" priority="1"
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

    # 4. Obstacle Pillars (if any)
    pillars = getattr(scenario, "pillars", None)
    if pillars is not None and len(pillars) > 0:
        pillars = np.asarray(pillars, dtype=float).reshape(-1, 3)  # x, y, radius
        for p_idx, (px, py, pr) in enumerate(pillars):
            walls_xml.append(
                f'<geom name="pillar_{p_idx}" type="cylinder" pos="{px:.4f} {py:.4f} {half_h:.4f}" '
                f'size="{pr:.4f} {half_h:.4f}" rgba="0.85 0.25 0.25 1" '
                f'friction="0.8 0.005 0.0001" condim="3"/>'
            )

    # 4b. Ground Step / Obstacle Blocks (if any)
    steps = getattr(scenario, "steps", None)
    if steps is not None and len(steps) > 0:
        for s_idx, (sx, sy, shx, shy, sh) in enumerate(steps):
            walls_xml.append(
                f'<geom name="step_{s_idx}" type="box" pos="{sx:.4f} {sy:.4f} {sh / 2.0:.4f}" '
                f'size="{shx:.4f} {shy:.4f} {sh / 2.0:.4f}" rgba="0.88 0.45 0.15 1" '
                f'friction="1.2 0.005 0.0001" condim="3"/>'
            )

    # 4c. Floor Gaps / Cracks (شکاف) — recessed trench the ball must cross
    # Each gap: (cx, cy, half_x, half_y, depth)
    gaps = getattr(scenario, "gaps", None)
    if gaps is not None and len(gaps) > 0:
        for g_idx, (gx, gy, ghx, ghy, gdepth) in enumerate(gaps):
            # Two raised floor platforms on either side with a trench between them
            # The "gap" is simply a void below the floor plane, so we build raised
            # edges that the ball can fall between.
            edge_h = 0.008  # thin raised lip around the gap
            walls_xml.append(
                f'<geom name="gap_edge_a_{g_idx}" type="box" '
                f'pos="{gx - ghx - 0.02:.4f} {gy:.4f} {edge_h / 2:.4f}" '
                f'size="0.02 {ghy:.4f} {edge_h / 2:.4f}" rgba="0.25 0.22 0.20 1" '
                f'friction="0.6 0.005 0.0001" condim="3"/>'
            )
            walls_xml.append(
                f'<geom name="gap_edge_b_{g_idx}" type="box" '
                f'pos="{gx + ghx + 0.02:.4f} {gy:.4f} {edge_h / 2:.4f}" '
                f'size="0.02 {ghy:.4f} {edge_h / 2:.4f}" rgba="0.25 0.22 0.20 1" '
                f'friction="0.6 0.005 0.0001" condim="3"/>'
            )
            # The gap trench itself (a recessed box below floor level)
            walls_xml.append(
                f'<geom name="gap_trench_{g_idx}" type="box" '
                f'pos="{gx:.4f} {gy:.4f} {-gdepth / 2:.4f}" '
                f'size="{ghx:.4f} {ghy:.4f} {gdepth / 2:.4f}" rgba="0.12 0.10 0.08 1" '
                f'friction="0.3 0.005 0.0001" condim="3"/>'
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

    # 4e. Scattered Stones / Pebbles on the floor
    # Each stone_zone: (cx, cy, half_x, half_y, n_stones, max_stone_size)
    stones = getattr(scenario, "stones", None)
    if stones is not None and len(stones) > 0:
        for st_idx, stone_def in enumerate(stones):
            stx, sty, sthx, sthy = stone_def[:4]
            n_stones = int(stone_def[4]) if len(stone_def) > 4 else 20
            max_sz = float(stone_def[5]) if len(stone_def) > 5 else 0.025
            rng = np.random.RandomState(1337 + st_idx)
            for si in range(n_stones):
                ox = stx + rng.uniform(-sthx * 0.9, sthx * 0.9)
                oy = sty + rng.uniform(-sthy * 0.9, sthy * 0.9)
                sr = rng.uniform(max_sz * 0.3, max_sz)
                # Irregular shape: randomly choose box or sphere
                shape = rng.choice(["sphere", "box"])
                grey = rng.uniform(0.35, 0.70)
                if shape == "sphere":
                    walls_xml.append(
                        f'<geom name="stone_{st_idx}_{si}" type="sphere" '
                        f'pos="{ox:.4f} {oy:.4f} {sr:.4f}" size="{sr:.4f}" '
                        f'rgba="{grey:.2f} {grey * 0.95:.2f} {grey * 0.90:.2f} 1" '
                        f'friction="1.5 0.05 0.002" condim="3" mass="0.005"/>'
                    )
                else:
                    sx2 = rng.uniform(sr * 0.6, sr * 1.2)
                    sy2 = rng.uniform(sr * 0.6, sr * 1.2)
                    sz2 = rng.uniform(sr * 0.4, sr * 0.8)
                    walls_xml.append(
                        f'<geom name="stone_{st_idx}_{si}" type="box" '
                        f'pos="{ox:.4f} {oy:.4f} {sz2:.4f}" '
                        f'size="{sx2:.4f} {sy2:.4f} {sz2:.4f}" '
                        f'rgba="{grey:.2f} {grey * 0.95:.2f} {grey * 0.90:.2f} 1" '
                        f'friction="1.5 0.05 0.002" condim="3" mass="0.008"/>'
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

    xml_str = f"""<mujoco model="radial_sphere_arena">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="{timestep:.5f}" gravity="0 0 -9.81" integrator="implicitfast"/>

    <visual>
        <headlight ambient="0.45 0.45 0.45" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
        <rgba haze="0.12 0.20 0.30 1"/>
        <global azimuth="140" elevation="-30"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.94 0.94 0.95" rgb2="0.86 0.86 0.88"/>
        <texture name="skybox" type="skybox" builtin="gradient"
                 rgb1="0.20 0.35 0.55" rgb2="0.04 0.07 0.12" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="35 35" reflectance="0.08" texuniform="true"/>
        <material name="wall_mat" rgba="0.28 0.32 0.38 1" reflectance="0.05"/>
        <material name="goal_mat" rgba="0.0 0.85 0.90 0.60" reflectance="0.1"/>
        <material name="goal_pad_mat" rgba="0.0 0.85 0.90 0.35" reflectance="0.05"/>
        <material name="core_mat" rgba="{core_rgba}" specular="0.6" shininess="0.8" reflectance="0.12"/>
    </asset>

    <worldbody>
        <light pos="{cx_arena:.2f} {cy_arena:.2f} 12" dir="0 0 -1" directional="true"
               diffuse="0.90 0.90 0.90" specular="0.3 0.3 0.3"/>
        <light pos="0 0 8" dir="0 0 -1" directional="false"
               diffuse="0.40 0.40 0.40" specular="0.2 0.2 0.2"/>

        <!-- Floor Plane -->
        <geom name="floor" type="plane" size="50 50 0.1" material="grid"
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
                  friction="0.85 0.015 0.005" condim="4"/>
            {''.join(bars_xml)}
        </body>
    </worldbody>

    <actuator>
        {''.join(actuators_xml)}
    </actuator>
</mujoco>
"""
    return xml_str, dirs
