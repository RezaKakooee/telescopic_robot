"""Render script for the trench and gap crossing skill (straddle_gap).
Demonstrates RoboBall bridging an open 22 cm central trench between two elevated
platforms using coordinated dual-flank outrigger peristalsis and active centerline servoing.

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

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _assets import assets_dir  # noqa: E402

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill


def main():
    assets = assets_dir()
    assets.mkdir(parents=True, exist_ok=True)
    
    cfg = load_config("configs/rl/gap_bridge.yaml")
    cfg.camera.enabled = False
    scenario = generate_scenario("gap_bridge", cfg, seed=42)
    # Remove enclosing outer walls for an open, clean traversal view
    scenario.walls = np.zeros((0, 4), dtype=np.float32)
    
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10000)
    env.reset(seed=42)
    
    deck_h = 0.25
    gap_w = 0.22
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = deck_h + 0.19
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    
    # Settle onto Box 1 and Box 2
    for _ in range(25):
        t = execute_skill("straddle_gap", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend)
        env.step(t)
        
    start_x = float(env.data.qpos[0])
    start_z = float(env.data.qpos[2])
    d_fwd = np.array([1.0, 0.0])
    
    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.core_body_id
    camera.azimuth = 148.0
    camera.elevation = -22.0
    camera.distance = 2.25
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 15)
    badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    
    dt = float(env.model.opt.timestep * env.action_repeat)
    # Target 25 fps, 1x real-time (stride = round(1 / (25 * dt)))
    video_stride = max(1, round(1.0 / (25 * dt)))
    
    badges = [
        ("DUAL-FLANK OUTRIGGER", 175, (56, 189, 248)),
        ("UNDERBELLY TUCKED", 165, (250, 204, 21)),
        ("CENTERLINE SERVO", 155, (192, 132, 252)),
        ("DECK TRACTION", 140, (74, 222, 128)),
    ]
    
    rows = []
    max_y_dev = 0.0
    preview_saved = False
    
    total_steps = 580
    
    print(f"=== Rendering Trench Straddle Video (Canvas 800x608) ===")
    with imageio.get_writer(assets / "move-straddle-gap.mp4", fps=25) as writer:
        for step in range(total_steps):
            sim_t = step * dt
            pos = env.data.qpos[:3].copy()
            vel = env.data.qvel[:3].copy()
            quat = env.data.qpos[3:7].copy()
            
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            vx = float(vel[0])
            max_y_dev = max(max_y_dev, abs(y))
            
            targets = execute_skill(
                "straddle_gap", quat, env.dirs_body, env.max_extend,
                d_hat=d_fwd, lateral_offset=y, centering_gain=2.2,
            )
            env.step(targets)
            
            rows.append({
                "time_s": sim_t,
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "vx_m_s": vx,
                "deck_clearance_cm": (z - deck_h - 0.15) * 100,
            })
            
            if step % video_stride == 0:
                renderer.update_scene(env.data, camera)
                render_frame = renderer.render()
                
                canvas = Image.new("RGB", (800, 608), (15, 23, 42))
                canvas.paste(Image.fromarray(render_frame), (0, 160))
                draw = ImageDraw.Draw(canvas)
                
                # Badges
                badge_xs = [20, 205, 380, 545]
                for b_i, (b_name, b_w, b_col) in enumerate(badges):
                    bx = badge_xs[b_i]
                    draw.rounded_rectangle([(bx, 18), (bx + b_w, 50)], radius=5, fill=b_col)
                    draw.text((bx + 10, 26), b_name, font=badge_font, fill=(15, 23, 42))
                    
                # Title and Live Telemetry
                draw.text((20, 65), f"Bridging open {gap_w*100:.0f} cm trench on dual platforms", font=large, fill=(255, 255, 255))
                deck_margin = (z - deck_h - 0.15) * 100
                telemetry_str = f"Progress: {x:.2f} m / 4.80 m   Speed: {vx:.2f} m/s   Offset: {y*100:+.1f} cm   Margin: {deck_margin:+.1f} cm"
                draw.text((20, 98), telemetry_str, font=font, fill=(56, 189, 248))
                draw.text((20, 126), "Central rods tucked to clear void; lateral flanks drive both ledges synchronously", font=small, fill=(203, 213, 225))
                
                # Right aligned info
                draw.text((690, 20), f"{sim_t:4.2f} s", font=large, fill=(255, 255, 255))
                draw.text((680, 52), "Telemetry HUD", font=small, fill=(148, 163, 184))
                
                frame_np = np.asarray(canvas)
                writer.append_data(frame_np)
                
                # Capture preview frame at mid-course (e.g. x around 2.2 m)
                if not preview_saved and x >= 2.2:
                    canvas.save(assets / "move-straddle-gap-preview.png")
                    preview_saved = True
                    
            if x >= 4.85:
                print(f"Reached finish line at step {step} (t = {sim_t:.2f}s, x = {x:.2f}m)")
                break
                    
    # Save CSV
    with open(assets / "move-straddle-gap.csv", "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
        
    # Save JSON summary
    end_x = float(rows[-1]["x_m"])
    end_y = float(rows[-1]["y_m"])
    end_z = float(rows[-1]["z_m"])
    avg_speed = float(np.mean([r["vx_m_s"] for r in rows[50:]]))
    
    summary = {
        "skill": "straddle_gap",
        "gap_width_cm": gap_w * 100,
        "platform_height_cm": deck_h * 100,
        "initial_pos": [start_x, 0.0, start_z],
        "final_pos": [end_x, end_y, end_z],
        "total_distance_m": end_x - start_x,
        "max_centerline_deviation_cm": max_y_dev * 100,
        "average_cruise_speed_m_s": avg_speed,
        "min_core_z_m": float(np.min([r["z_m"] for r in rows])),
        "success": bool(end_x >= 4.0 and end_z > deck_h + 0.10),
    }
    with open(assets / "move-straddle-gap-results.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"✅ Generated straddle video: traversed {summary['total_distance_m']:.2f} m, max y-dev = {summary['max_centerline_deviation_cm']:.1f} cm, avg speed = {avg_speed:.2f} m/s")


if __name__ == "__main__":
    main()
