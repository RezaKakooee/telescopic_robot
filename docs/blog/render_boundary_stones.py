"""Render script for Wall-less Circular Boundary Containment on Rocky Stone Terrain.

Skill: stay_in_boundary with active suspension & terrain conformance on rough stones.
Arena: Wall-less circular boundary (R = 2.0m) filled with procedural stones/boulders.
Duration: 12,000 simulation steps @ stride 4 = 3,000 frames @ 25 fps = 120.0s (2 minutes).

Features:
  - 130 multi-faceted stones/boulders distributed uniformly inside the circular arena.
  - Active suspension: underneath bars adjust to stones (retract on rocks, extend into dips).
  - Skyhook heave damper keeps core ride height stable while navigating boulders.
  - Multi-skill roaming: forward cruise, curved meander, in-place pivots, and boundary deflection.
  - Simplified HUD header modeled after docs/blog/assets/jump-to-precision.mp4.
  - Faststart web-streamable MP4 encoding.

Outputs:
  - docs/blog/assets/boundary-stones.mp4
  - docs/blog/assets/boundary-stones-preview.png
  - docs/blog/assets/boundary-stones.csv
  - docs/blog/assets/boundary-stones-results.json
"""
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

os.environ.setdefault("MUJOCO_GL", "egl")
import csv
import json
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.mid_level.navigation import stay_in_boundary


def main():
    assets = Path(__file__).resolve().parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/rl/standing_jump_showcase.yaml")
    cfg.camera.enabled = False

    boundary_radius = 2.0
    n_stones = 130
    max_stone_size = 0.055

    scenario = generate_scenario(
        "boundary",
        cfg,
        radius=boundary_radius,
        n_segments=64,
        n_stones=n_stones,
        max_stone_size=max_stone_size,
    )

    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=15000)
    env.reset(seed=42)

    # Initial settling steps
    for _ in range(25):
        targets = np.full(len(env.dirs_body), 0.025, dtype=np.float32)
        env.step(targets)

    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.core_body_id
    camera.elevation = -26.0
    camera.distance = 3.50

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 17)
    small = ImageFont.truetype(font_path, 13)
    badge_font = ImageFont.truetype(font_bold_path, 12)
    large = ImageFont.truetype(font_bold_path, 20)

    dt = float(env.model.opt.timestep * env.action_repeat)
    video_stride = 4  # 12,000 steps / 4 = 3,000 frames @ 25 fps = 120.0s (2 minutes)

    # 4 distinct state badges modeled after jump-to-precision
    badges = [
        ("FORWARD ROAM", 135, (56, 189, 248)),     # Sky Blue
        ("CURVED MEANDER", 145, (192, 132, 252)),  # Purple
        ("IN-PLACE TURN", 135, (74, 222, 128)),    # Green
        ("BOUNDARY DEFLECT", 155, (250, 204, 21)), # Amber Yellow
    ]
    badge_xs = [16, 160, 314, 458]

    video_path = assets / "boundary-stones.mp4"
    preview_path = assets / "boundary-stones-preview.png"
    csv_path = assets / "boundary-stones.csv"
    json_path = assets / "boundary-stones-results.json"

    writer = imageio.get_writer(
        str(video_path),
        fps=25,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart", "-crf", "23", "-preset", "medium"],
    )

    csv_rows = []
    saved_preview = False
    sim_step = 0
    t_sim = 0.0

    r_history = []
    speed_history = []
    core_z_history = []
    sub_skills_seen = set()

    total_sim_steps = 12000  # 120.0s simulation -> 2 minutes video
    print(f"Recording 2-minute boundary containment with stones ({total_sim_steps} steps, 3,000 frames) to {video_path}...")
    start_wall_time = time.time()

    while sim_step < total_sim_steps:
        pos = env.data.qpos[:3].copy()
        quat = env.data.qpos[3:7].copy()
        vel = env.data.qvel[:3].copy()

        r = float(np.linalg.norm(pos[:2]))
        speed_mag = float(np.linalg.norm(vel[:2]))
        core_z = float(pos[2])
        r_history.append(r)
        speed_history.append(speed_mag)
        core_z_history.append(core_z)

        tc = env.get_terrain_clearances() if hasattr(env, "get_terrain_clearances") else None
        cf = env.get_rod_contact_forces() if hasattr(env, "get_rod_contact_forces") else None

        targets, meta = stay_in_boundary(
            quat,
            env.dirs_body,
            env.max_extend,
            ball_xy=pos[:2],
            lin_vel=vel[:2],
            boundary_radius=boundary_radius,
            speed=0.75,
            safety_margin=0.65,
            step_count=sim_step,
            core_z=core_z,
            core_vz=vel[2],
            contact_forces=cf,
            terrain_clearances=tc,
            enable_suspension=True,
            return_metadata=True,
        )

        sub_skills_seen.add(meta["sub_skill"])
        env.step(targets)
        sim_step += 1
        t_sim += dt

        # Record CSV row every 10 steps to keep file compact
        if sim_step % 10 == 0:
            csv_rows.append({
                "step": sim_step,
                "time_s": round(t_sim, 3),
                "ball_x": round(float(pos[0]), 4),
                "ball_y": round(float(pos[1]), 4),
                "radius_m": round(r, 4),
                "core_z_m": round(core_z, 4),
                "distance_to_edge_m": round(meta["distance_to_edge"], 4),
                "speed_mps": round(speed_mag, 4),
                "v_radial_mps": round(meta["v_radial"], 4),
                "sub_skill": meta["sub_skill"],
                "action_name": meta["action_name"],
                "is_boundary_active": int(meta["is_boundary_active"]),
            })

        # Render composite frame
        if sim_step % video_stride == 0:
            camera.azimuth = 65.0 + 12.0 * np.sin(0.08 * t_sim)
            renderer.update_scene(env.data, camera)
            viewport = renderer.render()

            canvas = Image.new("RGB", (800, 608), (15, 23, 42))
            canvas.paste(Image.fromarray(viewport), (0, 160))
            draw = ImageDraw.Draw(canvas)

            # Determine active mode index:
            if meta["is_boundary_active"]:
                active_idx = 3  # BOUNDARY DEFLECT
            elif meta["sub_skill"] == "curve":
                active_idx = 1  # CURVED MEANDER
            elif meta["sub_skill"] == "turn":
                active_idx = 2  # IN-PLACE TURN
            else:
                active_idx = 0  # FORWARD ROAM

            # 1. Simplified badge pills
            for b_i, (b_name, b_w, b_col) in enumerate(badges):
                bx = badge_xs[b_i]
                is_act = (b_i == active_idx)
                bg = b_col if is_act else (30, 41, 59)
                tx = (15, 23, 42) if is_act else (180, 195, 215)
                border = b_col if is_act else (60, 75, 100)
                draw.rounded_rectangle([(bx, 16), (bx + b_w, 48)], radius=6, fill=bg, outline=border, width=1)
                draw.text((bx + 10, 24), b_name, font=badge_font, fill=tx)

            # 2. Time info and boundary on top right
            draw.text((675, 16), f"{t_sim:5.1f} s", font=large, fill=(255, 255, 255))
            draw.text((645, 42), "Boundary: 2.0 m", font=small, fill=(56, 189, 248))

            # 3. Clean status title
            if meta["is_boundary_active"]:
                draw.text((16, 62), "Active boundary inward deflection & speed regulation", font=large, fill=(255, 255, 255))
                rad_col = (250, 204, 21)  # Amber
            else:
                draw.text((16, 62), "Rough-terrain active suspension roaming inside boundary", font=large, fill=(255, 255, 255))
                rad_col = (56, 189, 248)  # Cyan

            # 4. Telemetry metrics line
            draw.text(
                (16, 94),
                f"Radius: {r:.2f} m / {boundary_radius:.2f} m   Speed: {speed_mag:.2f} m/s   Core z: {core_z:.2f} m   Edge dist: {meta['distance_to_edge']:.2f} m",
                font=font,
                fill=rad_col,
            )

            # 5. Explanatory subtitle line
            draw.text(
                (16, 124),
                f"Active Primitive: [{meta['sub_skill'].upper()}] ({meta['action_name']}) • Active Suspension Conformance • 130 Stones",
                font=small,
                fill=(203, 213, 225),
            )

            # Clean separator line between HUD and 3D view
            draw.line([(0, 159), (800, 159)], fill=(51, 65, 85), width=2)

            frame_arr = np.array(canvas)
            writer.append_data(frame_arr)

            # Save preview when robot is actively deflecting near the line over stones
            if not saved_preview and meta["is_boundary_active"] and sim_step > 300:
                canvas.save(preview_path)
                saved_preview = True
                print(f"Saved preview image to {preview_path} at step {sim_step}")

        if sim_step % 1000 == 0:
            elapsed = time.time() - start_wall_time
            print(f"Step {sim_step}/{total_sim_steps} ({t_sim:.1f}s sim) - r={r:.2f}m, speed={speed_mag:.2f}m/s - elapsed {elapsed:.1f}s")

    if not saved_preview:
        canvas.save(preview_path)
        print(f"Saved fallback preview image to {preview_path}")

    writer.close()
    env.close()

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["step", "time_s", "ball_x", "ball_y", "radius_m", "core_z_m", "distance_to_edge_m",
                      "speed_mps", "v_radial_mps", "sub_skill", "action_name", "is_boundary_active"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    # Save JSON summary
    max_radius_reached = float(np.max(r_history))
    mean_speed = float(np.mean(speed_history))
    results = {
        "scenario": "wall_less_circular_boundary_stones",
        "boundary_radius_m": boundary_radius,
        "n_stones": n_stones,
        "max_stone_size": max_stone_size,
        "max_radius_reached_m": round(max_radius_reached, 4),
        "containment_margin_m": round(boundary_radius - max_radius_reached, 4),
        "containment_success": bool(max_radius_reached < boundary_radius),
        "mean_speed_mps": round(mean_speed, 4),
        "max_speed_mps": round(float(np.max(speed_history)), 4),
        "mean_core_z_m": round(float(np.mean(core_z_history)), 4),
        "sub_skills_observed": sorted(list(sub_skills_seen)),
        "sim_steps": sim_step,
        "sim_duration_s": round(t_sim, 2),
        "video_duration_s": round(sim_step / (video_stride * 25.0), 2),
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    total_wall_time = time.time() - start_wall_time
    print(f"\nRender complete in {total_wall_time:.1f}s!")
    print(f"Max radius: {max_radius_reached:.3f}m / {boundary_radius:.2f}m boundary (Margin: {results['containment_margin_m']:.3f}m)")
    print(f"Mean speed: {mean_speed:.3f}m/s, Max speed: {results['max_speed_mps']:.3f}m/s")
    print(f"Mean core ride height: {results['mean_core_z_m']:.3f}m")
    print(f"Video saved to: {video_path} (Duration: {results['video_duration_s']}s)")
    print(f"Results JSON: {json_path}")


if __name__ == "__main__":
    main()
