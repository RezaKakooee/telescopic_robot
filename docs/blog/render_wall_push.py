"""Render script for wall contact and obstacle pushing (push_against_wall).
Demonstrates RoboBall approaching a vertical wall, bracing rods against the surface,
and actively shoving itself away into open space.

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


def main():
    assets = Path(__file__).resolve().parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    
    wall_y = -0.45
    scenario = Scenario(
        kind="goal",
        name="wall_push_demo",
        spawn_xy=np.array([0.0, -0.15], dtype=np.float32),
        goal=np.array([30.0, 0.0], dtype=np.float32),
        path_pts=np.array([[0.0, -0.15], [30.0, -0.15]], dtype=np.float32),
        markers=np.empty((0, 2), dtype=np.float32),
        path_length=30.0,
        walls=np.array([[-1.0, wall_y, 35.0, wall_y]], dtype=np.float32),
    )
    
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=3000)
    env.reset(seed=42)
    
    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.core_body_id
    camera.azimuth = -52.0
    camera.elevation = -18.0
    camera.distance = 2.2
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 15)
    badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    
    dt = float(env.model.opt.timestep * env.action_repeat)
    video_stride = max(1, round(1.0 / (25 * dt * 2)))  # record every 2 steps
    
    badges = [
        ("APPROACH & ANGLE", 155, (56, 189, 248)),
        ("CONTACT & BRACE", 155, (250, 204, 21)),
        ("LATERAL SHOVE", 145, (192, 132, 252)),
        ("LANE RECOVERY", 145, (74, 222, 128)),
    ]
    
    rows = []
    wall_normal = np.array([0.0, 1.0])
    y_min = 0.0
    max_vy = 0.0
    preview_saved = False
    
    state = "approach"
    state_timer = 0
    contact_count = 0
    max_contacts = 3
    cycle_records = []
    curr_min_gap = 999.0
    curr_max_vy = -999.0
    
    total_steps = 640
    print("=== Rendering Extended Multi-Contact Wall Push Video (Canvas 800x608, 3 Contacts) ===")
    
    with imageio.get_writer(assets / "wall-push-shove.mp4", fps=25) as writer:
        for step in range(total_steps):
            sim_t = step * dt
            pos = env.data.qpos[:3].copy()
            vel = env.data.qvel[:3].copy()
            quat = env.data.qpos[3:7].copy()
            
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            vx, vy = float(vel[0]), float(vel[1])
            y_min = min(y_min, y)
            max_vy = max(max_vy, vy)
            gap_to_wall = (y - wall_y - 0.15) * 100
            
            curr_min_gap = min(curr_min_gap, gap_to_wall)
            curr_max_vy = max(curr_max_vy, vy)
            
            # Dynamic state transitions for repeated multi-contact bounces
            if state == "approach":
                phase_idx = 0
                phase_name = "approach"
                d_y = -1.2 if gap_to_wall > 30.0 else -0.8
                d_dir = np.array([0.5, d_y])
                targets = execute_skill("move", quat, env.dirs_body, env.max_extend,
                                        d_hat=d_dir / np.linalg.norm(d_dir), speed=1.2)
                if gap_to_wall <= 9.0:
                    state = "brace"
                    state_timer = 0
                    contact_count += 1
            elif state == "brace":
                phase_idx = 1
                phase_name = "brace"
                targets = execute_skill("push_against_wall", quat, env.dirs_body, env.max_extend,
                                        wall_normal=wall_normal, push_strength=0.80)
                state_timer += 1
                if state_timer >= 16:
                    state = "shove"
                    state_timer = 0
            elif state == "shove":
                phase_idx = 2
                phase_name = "shove"
                targets = execute_skill("push_against_wall", quat, env.dirs_body, env.max_extend,
                                        wall_normal=wall_normal, push_strength=0.98)
                state_timer += 1
                if state_timer >= 40:
                    cycle_records.append({
                        "contact_index": contact_count,
                        "min_gap_cm": round(curr_min_gap, 2),
                        "peak_vy_m_s": round(curr_max_vy, 2),
                    })
                    curr_min_gap = 999.0
                    curr_max_vy = -999.0
                    if contact_count >= max_contacts:
                        state = "final_cruise"
                        state_timer = 0
                    else:
                        state = "cruise_back"
                        state_timer = 0
            elif state == "cruise_back":
                phase_idx = 3
                phase_name = "recovery"
                targets = execute_skill("move", quat, env.dirs_body, env.max_extend,
                                        d_hat=np.array([1.0, 0.0]), speed=1.1)
                state_timer += 1
                if state_timer >= 25:
                    state = "approach"
                    state_timer = 0
            else:  # final_cruise
                phase_idx = 3
                phase_name = "cruise"
                targets = execute_skill("move_forward", quat, env.dirs_body, env.max_extend,
                                        d_hat=np.array([1.0, 0.0]), speed=1.2)
                state_timer += 1
                
            env.step(targets)
            
            rows.append({
                "time_s": sim_t,
                "step": step,
                "contact_number": contact_count,
                "phase": phase_name,
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "vx_m_s": vx,
                "vy_m_s": vy,
                "gap_to_wall_cm": gap_to_wall,
            })
            
            if step % video_stride == 0:
                renderer.update_scene(env.data, camera)
                render_frame = renderer.render()
                
                canvas = Image.new("RGB", (800, 608), (15, 23, 42))
                canvas.paste(Image.fromarray(render_frame), (0, 160))
                draw = ImageDraw.Draw(canvas)
                
                # Draw badges
                badge_xs = [20, 185, 350, 505]
                for b_i, (b_name, b_w, b_col) in enumerate(badges):
                    bx = badge_xs[b_i]
                    is_act = (b_i == phase_idx)
                    bg = b_col if is_act else (40, 50, 70)
                    tx = (15, 23, 42) if is_act else (200, 210, 230)
                    draw.rounded_rectangle([(bx, 18), (bx + b_w, 50)], radius=5, fill=bg)
                    draw.text((bx + 10, 26), b_name, font=badge_font, fill=tx)
                    
                # Header & live telemetry
                contact_str = f"CONTACT {max(1, contact_count)} OF {max_contacts}"
                if phase_idx == 0:
                    draw.text((20, 65), f"Approaching wall barrier [{contact_str}]", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Wall Gap: {gap_to_wall:.1f} cm   vy: {vy:+.2f} m/s   vx: {vx:.2f} m/s", font=font, fill=(56, 189, 248))
                    draw.text((20, 126), "Steering obliquely toward wall face to establish controlled surface engagement", font=small, fill=(203, 213, 225))
                elif phase_idx == 1:
                    draw.text((20, 65), f"Surface contact & lateral rod bracing [{contact_str}]", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Wall Gap: {gap_to_wall:.1f} cm   Surface State: FIRM BRACE", font=font, fill=(250, 204, 21))
                    draw.text((20, 126), "Rods facing wall extend selectively while underbelly maintains ground stance", font=small, fill=(226, 232, 240))
                elif phase_idx == 2:
                    draw.text((20, 65), f"Active lateral shove & repulsive thrust [{contact_str}]", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Wall Gap: {gap_to_wall:.1f} cm   Repulsion Speed vy: {vy:+.2f} m/s", font=font, fill=(192, 132, 252))
                    draw.text((20, 126), "High-force rod thrust pushes off the wall, repelling the core into free space", font=small, fill=(226, 232, 240))
                else:
                    draw.text((20, 65), "Disengaged into open lane & cruising", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Wall Gap: {gap_to_wall:.1f} cm   Forward Speed: {vx:.2f} m/s", font=font, fill=(74, 222, 128))
                    draw.text((20, 126), "Barrier cleared; ball re-aligns with travel axis before next maneuver", font=small, fill=(203, 213, 225))
                    
                # Time & contact status info
                draw.text((690, 20), f"{sim_t:4.2f} s", font=large, fill=(255, 255, 255))
                draw.text((680, 52), f"Hits: {contact_count}/{max_contacts}", font=badge_font, fill=(250, 204, 21))
                
                frame_np = np.asarray(canvas)
                writer.append_data(frame_np)
                
                # Capture preview frame during Contact 2 active shove
                if not preview_saved and contact_count == 2 and phase_idx == 2 and 12 <= state_timer <= 28:
                    canvas.save(assets / "wall-push-shove-preview.png")
                    preview_saved = True
                    
    # If preview wasn't captured, save final
    if not preview_saved:
        canvas.save(assets / "wall-push-shove-preview.png")
        
    # Save CSV
    with open(assets / "wall-push-shove.csv", "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
        
    # Save JSON summary
    end_x = float(rows[-1]["x_m"])
    end_y = float(rows[-1]["y_m"])
    
    summary = {
        "skill": "push_against_wall",
        "wall_y_m": wall_y,
        "total_contacts": contact_count,
        "total_duration_s": round(total_steps * dt, 2),
        "total_forward_progress_m": round(end_x, 2),
        "cycle_records": cycle_records,
        "max_repulsion_vy_m_s": round(max_vy, 2),
        "final_position": [round(end_x, 2), round(end_y, 2)],
        "success": bool(contact_count >= 3 and max_vy > 1.5),
    }
    with open(assets / "wall-push-shove-results.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"✅ Generated extended multi-contact wall push video: {contact_count} contacts completed, max vy={max_vy:.2f} m/s, forward dist={end_x:.2f} m")


if __name__ == "__main__":
    main()
