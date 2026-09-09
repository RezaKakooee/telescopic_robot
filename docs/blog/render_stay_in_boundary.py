"""Render script for Wall-less Circular Boundary Containment (stay_in_boundary).

Simplified HUD header modeled after docs/blog/assets/jump-to-precision.mp4:
  - Dynamic 4-badge mode pill bar highlighting the currently active state:
    [FORWARD ROAM], [CURVED MEANDER], [IN-PLACE TURN], [BOUNDARY DEFLECT]
  - Clean two-line telemetry: large status header, live metrics (radius, speed,
    radial velocity, distance to edge), and active sub-skill annotation.
  - No cluttered nested boxes or tiny cards.

Performance & Timing:
  - Commanded cruise speed: 0.80 m/s (fast, dynamic, agile locomotion).
  - Duration: 12,000 simulation steps @ stride 4 = 3,000 frames @ 25 fps = 120.0s (2 minutes).
  - Strict containment guarantee: ball remains strictly inside r < 2.0m for all 2 minutes.

Outputs:
  - docs/blog/assets/stay-in-boundary.mp4
  - docs/blog/assets/stay-in-boundary-preview.png
  - docs/blog/assets/stay-in-boundary.csv
  - docs/blog/assets/stay-in-boundary-results.json
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

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from radial_sphere.config import load_config, script_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.mid_level.navigation import stay_in_boundary


def main():
    assets = Path(__file__).resolve().parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    args = script_config("render_stay_in_boundary")

    cfg = load_config("configs/rl/standing_jump_showcase.yaml")
    cfg.camera.enabled = False

    boundary_radius = 2.0
    n_stones = 130 if args.stones else 0
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
    for _ in range(25 if args.stones else 20):
        targets = np.full(len(env.dirs_body), 0.025, dtype=np.float32)
        env.step(targets)

    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.core_body_id
    camera.elevation = -26.0
    camera.distance = 3.50 if args.stones else 3.40

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

    file_prefix = "boundary-stones" if args.stones else "stay-in-boundary"
    video_path = assets / f"{file_prefix}.mp4"
    preview_path = assets / f"{file_prefix}-preview.png"
    csv_path = assets / f"{file_prefix}.csv"
    json_path = assets / f"{file_prefix}-results.json"

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
    print(f"Recording 2-minute boundary containment (stones={args.stones}) ({total_sim_steps} steps, 3,000 frames) to {video_path}...")

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

        tc = env.get_terrain_clearances() if (args.stones and hasattr(env, "get_terrain_clearances")) else None
        cf = env.get_rod_contact_forces() if (args.stones and hasattr(env, "get_rod_contact_forces")) else None

        targets, meta = stay_in_boundary(
            quat,
            env.dirs_body,
            env.max_extend,
            ball_xy=pos[:2],
            lin_vel=vel[:2],
            boundary_radius=boundary_radius,
            speed=0.75 if args.stones else 0.80,
            safety_margin=0.65,
            step_count=sim_step,
            core_z=core_z if args.stones else None,
            core_vz=vel[2] if args.stones else None,
            contact_forces=cf,
            terrain_clearances=tc,
            enable_suspension=args.stones,
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

            # 1. Simplified badge pills (like jump-to-precision)
            for b_i, (b_name, b_w, b_col) in enumerate(badges):
                bx = badge_xs[b_i]
                is_act = (b_i == active_idx)
                bg = b_col if is_act else (30, 41, 59)
                tx = (15, 23, 42) if is_act else (180, 195, 215)
                border = b_col if is_act else (60, 75, 100)
                draw.rounded_rectangle([(bx, 16), (bx + b_w, 48)], radius=6, fill=bg, outline=border, width=1)
                draw.text((bx + 10, 24), b_name, font=badge_font, fill=tx)

            # 2. Time info on top right
            draw.text((675, 16), f"{t_sim:5.1f} s", font=large, fill=(255, 255, 255))
            draw.text((645, 42), "Boundary: 2.0 m", font=small, fill=(56, 189, 248))

            # 3. Clean status title
            if meta["is_boundary_active"]:
                draw.text((16, 62), "Active boundary inward deflection & speed regulation", font=large, fill=(255, 255, 255))
                rad_col = (250, 204, 21)  # Amber
            elif args.stones:
                draw.text((16, 62), "Rough-terrain active suspension roaming inside boundary", font=large, fill=(255, 255, 255))
                rad_col = (56, 189, 248)  # Cyan
            else:
                draw.text((16, 62), "Autonomous multi-skill roaming inside circular arena", font=large, fill=(255, 255, 255))
                rad_col = (56, 189, 248)  # Cyan

            # 4. Telemetry metrics line
            if args.stones:
                metrics_str = f"Radius: {r:.2f} m / {boundary_radius:.2f} m   Speed: {speed_mag:.2f} m/s   Core z: {core_z:.2f} m   Edge dist: {meta['distance_to_edge']:.2f} m"
            else:
                metrics_str = f"Radius: {r:.2f} m / {boundary_radius:.2f} m   Speed: {speed_mag:.2f} m/s   vr: {meta['v_radial']:+.2f} m/s   Edge dist: {meta['distance_to_edge']:.2f} m"

            draw.text(
                (16, 94),
                metrics_str,
                font=font,
                fill=rad_col,
            )

            # 5. Explanatory subtitle line
            sub_tag = "Active Suspension Conformance • 130 Stones" if args.stones else "Zero physical walls"
            draw.text(
                (16, 124),
                f"Active Primitive: [{meta['sub_skill'].upper()}] ({meta['action_name']}) • {sub_tag}",
                font=small,
                fill=(203, 213, 225),
            )

            # Clean separator line between HUD and 3D view
            draw.line([(0, 159), (800, 159)], fill=(51, 65, 85), width=2)

            frame_arr = np.array(canvas)
            writer.append_data(frame_arr)

            # Save preview when robot is actively deflecting near the line
            if not saved_preview and meta["is_boundary_active"] and sim_step > 200:
                canvas.save(preview_path)
                saved_preview = True
                print(f"Saved preview image to {preview_path} at step {sim_step}")

    if not saved_preview:
        canvas.save(preview_path)
        print(f"Saved fallback preview image to {preview_path}")

    writer.close()
    env.close()

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["step", "time_s", "ball_x", "ball_y", "radius_m", "distance_to_edge_m",
                      "speed_mps", "v_radial_mps", "sub_skill", "action_name", "is_boundary_active"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    # Save JSON summary
    max_radius_reached = float(np.max(r_history))
    mean_speed = float(np.mean(speed_history))
    results = {
        "scenario": "wall_less_circular_boundary",
        "boundary_radius_m": boundary_radius,
        "max_radius_reached_m": round(max_radius_reached, 4),
        "containment_margin_m": round(boundary_radius - max_radius_reached, 4),
        "containment_success": bool(max_radius_reached < boundary_radius),
        "mean_speed_mps": round(mean_speed, 4),
        "max_speed_mps": round(float(np.max(speed_history)), 4),
        "sub_skills_observed": sorted(list(sub_skills_seen)),
        "sim_steps": sim_step,
        "sim_duration_s": round(t_sim, 2),
        "video_duration_s": round(sim_step / (video_stride * 25.0), 2),
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Render complete! Max radius: {max_radius_reached:.3f}m / {boundary_radius:.2f}m boundary")
    print(f"Mean speed: {mean_speed:.3f}m/s, Max speed: {results['max_speed_mps']:.3f}m/s")
    print(f"Video saved to: {video_path} (Duration: {results['video_duration_s']}s)")
    print(f"Results JSON: {json_path}")


if __name__ == "__main__":
    main()
