"""Demonstrate transparent core sphere: see inside the ball as rods extend and compress dynamically!

Renders:
1. Macro 3D close-up view showing the internal guide sleeves, telescopic shafts, and pneumatic extension/compression waves.
2. Motion sequence:
   - In-place radial breathing: all rods extending out and compressing in.
   - Peristaltic drive: rolling across the floor, showing internal rods cyclically pushing against the ground and retracting into the hub.
   - High jump & landing cushion: explosive rocket leap, airborne tuck, landing gear deployment, and internal compression absorbing the impact.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import shutil
from pathlib import Path
import imageio
import mujoco
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario, generate_scenario
from skills import execute_skill
from skills.overlay import annotate
from skills.locomotion import move, stop, go_fast
from skills.jumping import jump_up

FORWARD = np.array([1.0, 0.0], dtype=np.float32)


def render_transparent_core_demo(out_path: Path, still_out_path: Path):
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = True

    sc = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=sc, randomize=False, max_steps=600)
    env.reset(seed=42)

    # Make the core sphere transparent: translucent amber/yellow glass
    core_mat_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_MATERIAL, "core_mat")
    if core_mat_id >= 0:
        env.model.mat_rgba[core_mat_id] = [1.0, 0.85, 0.20, 0.25]

    core_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "core_geom")
    if core_geom_id >= 0:
        env.model.geom_rgba[core_geom_id] = [1.0, 0.85, 0.20, 0.25]

    # Increase sleeve visibility so internal hub guides stand out clearly
    for k in range(env.n_bars):
        sleeve_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"sleeve_{k}")
        if sleeve_id >= 0:
            env.model.geom_rgba[sleeve_id] = [0.22, 0.26, 0.35, 0.95] # dark gunmetal sleeves

    w = imageio.get_writer(str(out_path), fps=25, codec="libx264", quality=9)

    # Initialize renderer with native render_size
    if env.renderer is None:
        env.renderer = mujoco.Renderer(env.model, height=env.render_size[0], width=env.render_size[1])

    def get_frame(title, lines, distance=1.40, elevation=-22.0, azimuth=45.0):
        env._update_dynamic_colors()
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = env.core_body_id
        cam.distance = distance
        cam.elevation = elevation
        cam.azimuth = azimuth
        env.renderer.update_scene(env.data, camera=cam)
        raw = env.renderer.render()
        return np.array(annotate(raw, title, lines, margin=14), copy=True)

    key_stills = []

    # Sequence 1: Radial Pulse (Extension & Compression inside transparent shell) (steps 0 - 80)
    print("  Phase 1: Radial Pulse (Extension & Compression inside transparent shell)...")
    for step in range(80):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[:3].copy()
        # Sine wave stroke across all rods
        ext = 0.02 + 0.13 * (0.5 + 0.5 * np.sin(step * 0.14))
        targets = np.full(env.n_bars, ext, dtype=np.float32)
        env.step(targets)

        if step % 2 == 0:
            lines = [
                "Inspection: Transparent Core Shell (Alpha=0.25)",
                f"Internal Stroke: {ext*100:5.1f} cm / {env.max_extend*100:.1f} cm",
                f"Action: Symmetric Radial Breath & Compression",
                f"Internal Sleeves & Rod Pistons Fully Visible",
            ]
            f = get_frame("Transparent Core: Internal Rod Piston Action", lines, distance=1.35, elevation=-20.0, azimuth=step * 1.5)
            w.append_data(f)
            if step == 16:
                key_stills.append(f)
            elif step == 38:
                key_stills.append(f)

    # Sequence 2: Peristaltic Rolling (Internal Rod Cycling) (steps 80 - 200)
    print("  Phase 2: Peristaltic Rolling (Internal Rod Cycling)...")
    for step in range(120):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        targets = move(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, speed=1.2)
        env.step(targets)

        if step % 2 == 0:
            lines = [
                "Mode: Peristaltic Rolling Gait",
                f"Underbelly: Internal pistons compress on ground",
                f"Flanks & Rear: Rods extend outward to propel",
                f"Speed: {float(np.linalg.norm(vel[:2])):5.2f} m/s",
            ]
            f = get_frame("Transparent Core: Rolling Wave Mechanics", lines, distance=1.45, elevation=-22.0, azimuth=35.0)
            w.append_data(f)
            if step == 40:
                key_stills.append(f)

    # Sequence 3: High Jump & Suspension Compression (steps 200 - 300)
    print("  Phase 3: High Jump & Suspension Compression...")
    for step in range(100):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        z = float(pos[2])

        if step < 20:
            phase = "crouch (internal rods retract deep into core)"
            p_code = "crouch"
        elif step < 32:
            phase = "takeoff (explosive rod extension outwards)"
            p_code = "takeoff"
        elif z > 0.28:
            phase = "airborne (all rods tucked to core)"
            p_code = "airborne"
        elif z > 0.22:
            phase = "landing gear deployed"
            p_code = "landing"
        else:
            phase = "suspension compression (impact absorbed)"
            p_code = "landing"

        targets = execute_skill("jump_up", quat, env.dirs_body, env.max_extend, phase=p_code)
        env.step(targets)

        if step % 2 == 0:
            lines = [
                f"Jump Phase: {phase}",
                f"Height: z = {z:5.2f} m  (vz = {vel[2]:+5.2f} m/s)",
                f"Internal Mechanism: Telescopic shafts sliding",
                f"Shell Alpha: 0.25 Glassmorphic",
            ]
            f = get_frame("Transparent Core: Jump & Landing Cushion", lines, distance=1.55, elevation=-18.0, azimuth=50.0)
            w.append_data(f)
            if step == 26: # takeoff
                key_stills.append(f)
            elif step == 54: # landing compression
                key_stills.append(f)

    w.close()
    env.close()

    if len(key_stills) >= 4:
        top = np.concatenate([key_stills[0], key_stills[1]], axis=1)
        bot = np.concatenate([key_stills[2], key_stills[3]], axis=1)
        collage = np.concatenate([top, bot], axis=0)
        imageio.imwrite(str(still_out_path), collage)
        print(f"  --> Saved collage: {still_out_path}")

    print(f"  --> Saved video: {out_path}")


if __name__ == "__main__":
    out_dir = Path("storage_local/transparent_core_showcase")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "transparent_core_mechanism.mp4"
    grid_path = out_dir / "transparent_core_mechanism_grid.png"
    render_transparent_core_demo(video_path, grid_path)

    # Copy to brain artifact directory
    art_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/all_skills_showcase")
    art_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(video_path), str(art_dir / "transparent_core_mechanism.mp4"))
    shutil.copy2(str(grid_path), str(art_dir / "transparent_core_mechanism_grid.png"))
    print(f"  --> Copied to artifact directory: {art_dir}")
