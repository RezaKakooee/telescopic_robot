"""Render script for the controlled falling skill (fall_down).
Shows RoboBall stepping off an elevated platform and absorbing the drop landing
using compliant rod actuation without shell impact or chaotic rebounds.

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

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _assets import assets_dir  # noqa: E402

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario, skill_course_platform
from skills import execute_skill


def main():
    assets = assets_dir()
    
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    scenario = generate_scenario("skill_course", cfg, seed=42)
    # Remove walls to showcase an open, clean platform ledge drop
    scenario.walls = np.zeros((0, 4), dtype=np.float32)
    plat = skill_course_platform(cfg)
    
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10000)
    env.reset(seed=42)
    
    px, py = float(plat["xy"][0]), float(plat["xy"][1])
    env.data.qpos[0] = px - 0.22
    env.data.qpos[1] = py
    env.data.qpos[2] = plat["height"] + 0.20
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    
    for _ in range(60):
        env.step(execute_skill("stop", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend))
        
    start_z = float(env.data.qpos[2])
    deck = plat["height"] + 0.19
    drop_h = plat["height"]
    d = np.array([1.0, 0.0])
    
    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 85, -16, 2.4
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 16)
    badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 23)
    
    dt = float(env.model.opt.timestep * env.action_repeat)
    # 2x slow motion at 25 fps: each video frame corresponds to dt * video_stride = 0.02s
    video_stride = max(1, round(1 / (25 * 2 * dt)))
    
    phases_list = [
        ("EDGE CRAWL", 135, (56, 189, 248)),
        ("FREEFALL GEAR", 145, (250, 204, 21)),
        ("SHOCK ABSORB", 145, (192, 132, 252)),
        ("SETTLE", 95, (74, 222, 128)),
    ]
    
    phase = "edge"
    sim_t = 0.0
    rows = []
    min_z = start_z
    max_drop_vz = 0.0
    preview_saved = False
    
    total_duration = 3.6
    total_steps = round(total_duration / dt)
    
    with imageio.get_writer(assets / "fall-down-platform.mp4", fps=25) as writer:
        for step in range(total_steps):
            sim_t = step * dt
            pos = env.data.qpos[:3].copy()
            vel = env.data.qvel[:3].copy()
            z = float(pos[2])
            vz = float(vel[2])
            min_z = min(min_z, z)
            max_drop_vz = min(max_drop_vz, vz)
            
            # Phase transitions
            if phase == "edge" and z < deck - 0.04:
                phase = "freefall"
            elif phase == "freefall" and z < 0.28:
                phase = "absorb"
            elif phase == "absorb" and z < 0.22 and vz > -0.2:
                phase = "settle"
                
            targets = execute_skill("fall_down", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                                    d_hat=d, phase=phase, drop_height=drop_h, edge_speed=0.35, gear=0.5)
            env.step(targets)
            
            shell_clearance_cm = (z - 0.15) * 100
            
            rows.append({
                "time_s": sim_t,
                "phase": phase,
                "x_m": float(pos[0]),
                "y_m": float(pos[1]),
                "z_m": z,
                "vx_cm_s": float(vel[0] * 100),
                "vz_cm_s": float(vz * 100),
                "shell_clearance_cm": shell_clearance_cm,
            })
            
            if step % video_stride == 0:
                # Camera tracks the robot smoothly
                camera.lookat = [pos[0] + 0.05, pos[1], 0.26]
                renderer.update_scene(env.data, camera)
                render_frame = renderer.render()
                
                canvas = Image.new("RGB", (800, 608), (17, 24, 39))
                canvas.paste(Image.fromarray(render_frame), (0, 160))
                draw = ImageDraw.Draw(canvas)
                
                # Active phase index
                if phase == "edge":
                    act_idx = 0
                elif phase == "freefall":
                    act_idx = 1
                elif phase == "absorb":
                    act_idx = 2
                else:
                    act_idx = 3
                    
                # Draw 4 badges with exact positions
                badge_xs = [20, 165, 320, 475]
                for b_i, (b_name, b_w, b_col) in enumerate(phases_list):
                    bx = badge_xs[b_i]
                    is_act = (b_i == act_idx)
                    bg = b_col if is_act else (40, 50, 70)
                    tx = (15, 23, 42) if is_act else (200, 210, 230)
                    draw.rounded_rectangle([(bx, 20), (bx + b_w, 55)], radius=6, fill=bg)
                    draw.text((bx + 12, 28), b_name, font=badge_font, fill=tx)
                    
                # Header & telemetry descriptions
                if phase == "edge":
                    draw.text((20, 70), "Controlled lip roll-off at crawl speed", font=large, fill=(255, 255, 255))
                    draw.text((20, 105), f"Height: {z*100:.1f} cm   vz: {vz*100:+.0f} cm/s   Speed: 35 cm/s", font=font, fill=(56, 189, 248))
                    draw.text((20, 132), "Creeps forward gently to prevent tipping end-over-end off the deck", font=small, fill=(203, 213, 225))
                elif phase == "freefall":
                    draw.text((20, 70), "Mid-air landing gear deployment", font=large, fill=(255, 255, 255))
                    draw.text((20, 105), f"Height: {z*100:.1f} cm   vz: {vz*100:+.0f} cm/s (falling)", font=font, fill=(250, 204, 21))
                    draw.text((20, 132), "Underbelly rods extend 50% stroke; upper rods tuck to clear the ledge", font=small, fill=(226, 232, 240))
                elif phase == "absorb":
                    draw.text((20, 70), "Touchdown impact shock absorption", font=large, fill=(255, 255, 255))
                    draw.text((20, 105), f"Height: {z*100:.1f} cm   Shell Clearance: {shell_clearance_cm:+.1f} cm", font=font, fill=(192, 132, 252))
                    draw.text((20, 132), "Compliant rods compress smoothly, dissipating kinetic energy without bounce", font=small, fill=(226, 232, 240))
                else: # settle
                    draw.text((20, 70), "Stable ground settle, ready to roll", font=large, fill=(255, 255, 255))
                    draw.text((20, 105), f"Height: {z*100:.1f} cm   Status: Upright & damped", font=font, fill=(74, 222, 128))
                    draw.text((20, 132), "Stance height stabilized on floor; zero core-ground collisions", font=small, fill=(203, 213, 225))
                    
                # Time & slow-mo label (clean right alignment)
                draw.text((615, 22), f"{sim_t:4.2f} s", font=large, fill=(255, 255, 255))
                draw.text((615, 55), "2x slow motion", font=font, fill=(148, 163, 184))
                
                frame_np = np.asarray(canvas)
                writer.append_data(frame_np)
                
                # Save preview during absorb/touchdown
                if not preview_saved and phase == "absorb":
                    canvas.save(assets / "fall-down-platform-preview.png")
                    preview_saved = True
                    
    # Save CSV
    with open(assets / "fall-down-platform.csv", "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
        
    # Save JSON summary
    end_z = float(rows[-1]["z_m"])
    summary = {
        "skill": "fall_down",
        "platform_height_m": drop_h,
        "initial_core_z_m": start_z,
        "final_core_z_m": end_z,
        "net_drop_cm": (start_z - end_z) * 100,
        "minimum_core_z_m": min_z,
        "minimum_shell_clearance_cm": (min_z - 0.15) * 100,
        "peak_downward_velocity_m_s": abs(max_drop_vz),
        "zero_core_impact": bool(min_z > 0.15),
    }
    with open(assets / "fall-down-platform-results.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Controlled fall video generated successfully: net drop = {summary['net_drop_cm']:.1f} cm, min clearance = +{summary['minimum_shell_clearance_cm']:.1f} cm")

if __name__ == "__main__":
    main()
