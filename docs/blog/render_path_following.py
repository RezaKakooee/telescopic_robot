"""Render script for closed-loop ground path following (follow_path).
Demonstrates RoboBall tracking a visual painted ground path across straightaways,
curves, and corners by dynamically orchestrating existing skills (move, curve, turn, stop).

Canvas: 800x608 (Top HUD: 800x160, 3D viewport: 800x448).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario
from skills import execute_skill


def generate_curved_path():
    """Generate a multi-feature 7.8m path with straight, curved, and corner sections."""
    xs = np.linspace(0.0, 7.8, 80)
    ys = np.zeros_like(xs)
    for i, x in enumerate(xs):
        if x < 1.6:
            ys[i] = 0.0
        elif x < 4.8:
            # S-curve chicane
            ys[i] = 0.50 * np.sin((x - 1.6) / 3.2 * np.pi)
        elif x < 6.4:
            # Banked return curve
            ys[i] = -0.38 * np.sin((x - 4.8) / 1.6 * np.pi)
        else:
            ys[i] = 0.0
    return np.stack([xs, ys], axis=1).astype(np.float32)


def main():
    assets = Path(__file__).resolve().parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False

    path_pts = generate_curved_path()

    # Visual road ribbon painted directly onto the arena floor
    yardlines = []
    for pt in path_pts[::2]:
        yardlines.append([float(pt[0]), float(pt[1]), 0.06, 0.08, "0.15 0.85 0.95 0.90"])

    scenario = Scenario(
        kind="goal",
        name="follow_path_demo",
        spawn_xy=np.array([0.0, 0.0], dtype=np.float32),
        goal=path_pts[-1],
        path_pts=path_pts,
        markers=np.empty((0, 2), dtype=np.float32),
        path_length=7.8,
    )
    scenario.yardlines = yardlines

    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=2000)
    env.reset(seed=42)

    # Settle stance
    for _ in range(35):
        env.step(execute_skill("jump_up", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, phase="stand"))

    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.core_body_id
    camera.azimuth = 68.0
    camera.elevation = -25.0
    camera.distance = 2.4

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 15)
    badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)

    dt = float(env.model.opt.timestep * env.action_repeat)
    # Record every 2 simulation steps at 25 fps
    video_stride = max(1, round(1.0 / (25 * dt * 2)))

    badges = [
        ("MOVE (STRAIGHT)", 155, (56, 189, 248)),
        ("CURVE (ARC)", 140, (250, 204, 21)),
        ("TURN (CORNER)", 145, (192, 132, 252)),
        ("STOP (GOAL)", 135, (74, 222, 128)),
    ]

    rows = []
    preview_saved = False
    sub_skill_counts = {"move": 0, "curve": 0, "turn": 0, "stop": 0}
    max_cte = 0.0
    sum_cte = 0.0
    total_samples = 0

    total_steps = 620
    print("=== Rendering Ground Path Following Video (Canvas 800x608, Orchestrated Skills) ===")

    with imageio.get_writer(assets / "follow-path-ground.mp4", fps=25) as writer:
        for step in range(total_steps):
            sim_t = step * dt
            pos = env.data.qpos[:3].copy()
            vel = env.data.qvel[:3].copy()
            quat = env.data.qpos[3:7].copy()

            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            vx, vy = float(vel[0]), float(vel[1])
            speed_now = float(np.hypot(vx, vy))

            targets, meta = execute_skill(
                "follow_path", quat, env.dirs_body, env.max_extend,
                ball_xy=pos[:2], path_pts=path_pts, lin_vel=vel,
                lookahead=0.85, speed=1.2, goal_tolerance=0.35,
                return_metadata=True,
            )
            env.step(targets)

            sub_skill = meta["sub_skill"]
            sub_skill_counts[sub_skill] = sub_skill_counts.get(sub_skill, 0) + 1
            cte = abs(meta["cross_track_error"]) * 100.0  # in cm
            max_cte = max(max_cte, cte)
            sum_cte += cte
            total_samples += 1

            if sub_skill == "move":
                phase_idx = 0
            elif sub_skill == "curve":
                phase_idx = 1
            elif sub_skill == "turn":
                phase_idx = 2
            else:
                phase_idx = 3

            dist_to_goal = meta["dist_to_goal"]
            progress_pct = min(100.0, (x / 7.8) * 100.0)

            rows.append({
                "time_s": sim_t,
                "step": step,
                "sub_skill": sub_skill,
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "vx_m_s": vx,
                "vy_m_s": vy,
                "speed_m_s": speed_now,
                "cross_track_error_cm": cte,
                "curvature_1_m": meta.get("curvature", 0.0),
                "heading_err_deg": meta.get("heading_error_deg", 0.0),
                "dist_to_goal_m": dist_to_goal,
            })

            if step % video_stride == 0:
                renderer.update_scene(env.data, camera)
                render_frame = renderer.render()

                canvas = Image.new("RGB", (800, 608), (15, 23, 42))
                canvas.paste(Image.fromarray(render_frame), (0, 160))
                draw = ImageDraw.Draw(canvas)

                # Draw badges
                badge_xs = [20, 185, 335, 490]
                for b_i, (b_name, b_w, b_col) in enumerate(badges):
                    bx = badge_xs[b_i]
                    is_act = (b_i == phase_idx)
                    bg = b_col if is_act else (40, 50, 70)
                    tx = (15, 23, 42) if is_act else (200, 210, 230)
                    draw.rounded_rectangle([(bx, 18), (bx + b_w, 50)], radius=5, fill=bg)
                    draw.text((bx + 10, 26), b_name, font=badge_font, fill=tx)

                # Telemetry header & details
                if phase_idx == 0:
                    draw.text((20, 65), "Straightaway cruise via `move` primitive", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Cross-Track Err: {cte:.1f} cm   Speed: {speed_now:.2f} m/s   Curvature: {meta['curvature']:.2f} 1/m", font=font, fill=(56, 189, 248))
                    draw.text((20, 126), "Pure pursuit lookahead drives nominal peristaltic wave along road ribbon", font=small, fill=(203, 213, 225))
                elif phase_idx == 1:
                    draw.text((20, 65), "Continuous road arc via `curve` primitive", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Cross-Track Err: {cte:.1f} cm   Speed: {speed_now:.2f} m/s   Curvature: {meta['curvature']:.2f} 1/m", font=font, fill=(250, 204, 21))
                    draw.text((20, 126), "Asymmetric lateral push wave sustains continuous radius with centripetal balance", font=small, fill=(226, 232, 240))
                elif phase_idx == 2:
                    draw.text((20, 65), "Sharp cornering via `turn` primitive", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Cross-Track Err: {cte:.1f} cm   Heading Err: {meta['heading_error_deg']:+.1f}°   Speed: {speed_now:.2f} m/s", font=font, fill=(192, 132, 252))
                    draw.text((20, 126), "High-authority differential turning vector snaps heading into upcoming bend", font=small, fill=(226, 232, 240))
                else:
                    draw.text((20, 65), "Terminal arrival via `stop` primitive", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Goal Distance: {dist_to_goal*100:.1f} cm   Residual Speed: {speed_now:.2f} m/s", font=font, fill=(74, 222, 128))
                    draw.text((20, 126), "Active counter-torque kickstand braking brings sphere to rest on target pad", font=small, fill=(203, 213, 225))

                # Time & progress
                draw.text((690, 20), f"{sim_t:4.2f} s", font=large, fill=(255, 255, 255))
                draw.text((670, 52), f"Progress: {progress_pct:.0f}%", font=badge_font, fill=(56, 189, 248))

                frame_np = np.asarray(canvas)
                writer.append_data(frame_np)

                # Capture preview frame during chicane curve
                if not preview_saved and phase_idx == 1 and x > 3.0:
                    canvas.save(assets / "follow-path-ground-preview.png")
                    preview_saved = True

    if not preview_saved:
        canvas.save(assets / "follow-path-ground-preview.png")

    # Save CSV
    with open(assets / "follow-path-ground.csv", "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    # Save JSON summary
    end_x = float(rows[-1]["x_m"])
    end_y = float(rows[-1]["y_m"])
    final_dist = float(rows[-1]["dist_to_goal_m"])
    mean_cte = sum_cte / max(total_samples, 1)

    summary = {
        "skill": "follow_path",
        "total_path_length_m": 7.8,
        "final_position": [round(end_x, 2), round(end_y, 2)],
        "final_goal_distance_cm": round(final_dist * 100, 1),
        "mean_cross_track_error_cm": round(mean_cte, 1),
        "max_cross_track_error_cm": round(max_cte, 1),
        "sub_skills_dispatched": sub_skill_counts,
        "total_duration_s": round(total_steps * dt, 2),
        "success": bool(final_dist < 0.35 and mean_cte < 15.0),
    }
    with open(assets / "follow-path-ground-results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Generated follow_path video: final goal dist={summary['final_goal_distance_cm']:.1f} cm, mean CTE={mean_cte:.1f} cm, sub-skills={sub_skill_counts}")


if __name__ == "__main__":
    main()
