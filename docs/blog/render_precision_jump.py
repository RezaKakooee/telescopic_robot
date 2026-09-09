"""Render script for precision target jumping (jump_to).
Demonstrates RoboBall hopping onto an elevated target platform with closed-loop
takeoff velocity servoing, eliminating orientation lottery landing scatter.

Canvas: 800x608 (Top HUD: 800x160, 3D viewport: 800x448).
Video speed: 2x slow motion.
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
    
    target_x = 0.50
    target_h = 0.12
    step_half_x = 0.25
    step_half_y = 0.40
    scenario = Scenario(
        kind="goal",
        name="precision_jump_demo",
        spawn_xy=np.array([0.0, 0.0], dtype=np.float32),
        goal=np.array([5.0, 0.0], dtype=np.float32),
        path_pts=np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32),
        markers=np.empty((0, 2), dtype=np.float32),
        path_length=5.0,
        steps=[[target_x, 0.0, step_half_x, step_half_y, target_h]],
    )
    
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=1000)
    env.reset(seed=42)
    
    # Settle naturally in stable upright stance
    for _ in range(40):
        env.step(execute_skill("jump_to", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                               d_hat=np.array([1.0, 0.0]), phase="stand"))
        
    origin = env.data.qpos[:3].copy()
    start_x = float(origin[0])
    start_z = float(origin[2])
    d_hat = np.array([1.0, 0.0])
    vx_tgt = 0.70
    vz_tgt = 2.60
    
    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.azimuth = 90.0
    camera.elevation = -14.0
    camera.distance = 1.80
    camera.lookat = [0.26, 0.0, 0.24]
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 15)
    badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    
    dt = float(env.model.opt.timestep * env.action_repeat)
    # 2x slow motion at 25 fps
    video_stride = max(1, round(1.0 / (25 * 2 * dt)))
    
    badges = [
        ("STAND & CROUCH", 145, (56, 189, 248)),
        ("VELOCITY SERVO BURN", 175, (250, 204, 21)),
        ("BALLISTIC FLIGHT", 155, (192, 132, 252)),
        ("STICK LANDING", 140, (74, 222, 128)),
    ]
    
    phase = "stand"
    peak_z = start_z
    vz_launch = 0.0
    vx_launch = 0.0
    preview_saved = False
    rows = []
    
    total_steps = 150
    print("=== Rendering Precision Jump-To Video (Canvas 800x608, 2x Slow Motion) ===")
    
    with imageio.get_writer(assets / "jump-to-precision.mp4", fps=25) as writer:
        for step in range(total_steps):
            sim_t = step * dt
            pos = env.data.qpos[:3].copy()
            vel = env.data.qvel[:3].copy()
            quat = env.data.qpos[3:7].copy()
            
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
            peak_z = max(peak_z, z)
            
            # Phase transitions
            if sim_t < 0.12:
                phase = "stand"
            elif sim_t < 0.24:
                phase = "crouch"
            elif sim_t < 0.36:
                phase = "takeoff"
                if vz > vz_launch:
                    vz_launch = vz
                    vx_launch = vx
            elif vz < 0.0 and z < target_h + 0.22:
                phase = "landing"
            else:
                phase = "airborne"
                    
            if phase == "stand" or phase == "crouch":
                phase_idx = 0
            elif phase == "takeoff":
                phase_idx = 1
            elif phase == "airborne":
                phase_idx = 2
            else:
                phase_idx = 3
                
            targets = execute_skill(
                "jump_to", quat, env.dirs_body, env.max_extend,
                d_hat=d_hat, phase=phase,
                vel=vel, vx_target=vx_tgt, vz_target=vz_tgt, drop_height=target_h,
            )
            env.step(targets)
            
            rows.append({
                "time_s": sim_t,
                "step": step,
                "phase": phase,
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "vx_m_s": vx,
                "vz_m_s": vz,
                "vy_m_s": vy,
            })
            
            if step % video_stride == 0:
                renderer.update_scene(env.data, camera)
                render_frame = renderer.render()
                
                canvas = Image.new("RGB", (800, 608), (15, 23, 42))
                canvas.paste(Image.fromarray(render_frame), (0, 160))
                draw = ImageDraw.Draw(canvas)
                
                # Draw badges
                badge_xs = [20, 180, 375, 550]
                for b_i, (b_name, b_w, b_col) in enumerate(badges):
                    bx = badge_xs[b_i]
                    is_act = (b_i == phase_idx)
                    bg = b_col if is_act else (40, 50, 70)
                    tx = (15, 23, 42) if is_act else (200, 210, 230)
                    draw.rounded_rectangle([(bx, 18), (bx + b_w, 50)], radius=5, fill=bg)
                    draw.text((bx + 10, 26), b_name, font=badge_font, fill=tx)
                    
                # Header & live telemetry
                if phase_idx == 0:
                    draw.text((20, 65), "Pre-jump crouch & stance stabilization", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Target: x = {target_x:.2f} m, h = {target_h*100:.0f} cm   Command: vx*={vx_tgt:.2f}, vz*={vz_tgt:.2f}", font=font, fill=(56, 189, 248))
                    draw.text((20, 126), "Rods retract to ground level to prepare explosive takeoff impulse", font=small, fill=(203, 213, 225))
                elif phase_idx == 1:
                    draw.text((20, 65), "Closed-loop takeoff velocity servoing", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Burn vz: {vz:+.2f} / {vz_tgt:.2f} m/s   vx: {vx:.2f} / {vx_tgt:.2f} m/s   vy: {vy:+.2f}", font=font, fill=(250, 204, 21))
                    draw.text((20, 126), "Real-time feedback trims horizontal thrust and cancels lateral drift", font=small, fill=(226, 232, 240))
                elif phase_idx == 2:
                    draw.text((20, 65), "Ballistic parabolic trajectory toward platform", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Apex Height: {peak_z*100:.1f} cm   x: {x:.2f} m   Target: {target_x:.2f} m", font=font, fill=(192, 132, 252))
                    draw.text((20, 126), "Core flies on calibrated arc; orientation lottery eliminated", font=small, fill=(226, 232, 240))
                else:
                    draw.text((20, 65), "Compliant stick landing on elevated platform", font=large, fill=(255, 255, 255))
                    draw.text((20, 98), f"Final Deck Height: {z*100:.1f} cm   Target Center Margin: {abs(x - target_x)*100:.1f} cm", font=font, fill=(74, 222, 128))
                    draw.text((20, 126), "Touchdown damping absorbs shock with zero rollout drive", font=small, fill=(203, 213, 225))
                    
                # Time info
                draw.text((690, 20), f"{sim_t:4.2f} s", font=large, fill=(255, 255, 255))
                draw.text((680, 52), "2x Slow-Motion", font=small, fill=(148, 163, 184))
                
                frame_np = np.asarray(canvas)
                writer.append_data(frame_np)
                
                # Capture preview frame at apex
                if not preview_saved and phase_idx == 2 and abs(vz) < 0.35:
                    canvas.save(assets / "jump-to-precision-preview.png")
                    preview_saved = True
                    
    # If preview wasn't saved yet, save current canvas
    if not preview_saved:
        canvas.save(assets / "jump-to-precision-preview.png")
        
    # Save CSV
    with open(assets / "jump-to-precision.csv", "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
        
    # Save JSON summary
    end_x = float(rows[-1]["x_m"])
    end_y = float(rows[-1]["y_m"])
    end_z = float(rows[-1]["z_m"])
    pad_error = abs(end_x - target_x)
    
    summary = {
        "skill": "jump_to",
        "target_x_m": target_x,
        "target_height_m": target_h,
        "commanded_vx_m_s": vx_tgt,
        "commanded_vz_m_s": vz_tgt,
        "achieved_launch_vx_m_s": vx_launch,
        "achieved_launch_vz_m_s": vz_launch,
        "apex_height_m": peak_z,
        "landing_x_m": end_x,
        "landing_y_m": end_y,
        "landing_z_m": end_z,
        "landing_position_error_cm": round(pad_error * 100, 2),
        "success": bool(pad_error < 0.15 and end_z > target_h + 0.15),
    }
    with open(assets / "jump-to-precision-results.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"✅ Generated jump_to video: launch vz={vz_launch:.2f} m/s, landing x={end_x:.2f} m (pad error {summary['landing_position_error_cm']:.1f} cm)")


if __name__ == "__main__":
    main()
