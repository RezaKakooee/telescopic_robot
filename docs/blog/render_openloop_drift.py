"""Render script for Open-Loop Centrifugal Drift demonstration.
Shows RoboBall attempting to follow an R = 2.0 m curve by merely rotating
the commanded push heading at constant angular velocity omega = v / R.
Illustrates severe outward centrifugal drift and understeer compared to the intended arc.

Canvas: 1120x608
- Left panel (800x448): 3D tracking perspective.
- Right panel (320x448): Overhead minimap showing target circular arc vs actual open-loop spiral.
- Top HUD (1120x160): Telemetry, drift distance, warning badges.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill


def main():
    assets = Path(__file__).resolve().parent / "assets"
    
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range, cfg.scenario.goal.y_range = [0., 0.], [-400., -400.]
    env = MujocoRadialSphereEnv(cfg, scenario=generate_scenario("goal", cfg, seed=42),
                                randomize=False, max_steps=10000)
    env.reset(seed=42)
    
    for _ in range(60):
        env.step(execute_skill("jump_up", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, phase="stand"))
        
    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90, -45, 2.2
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 16)
    badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    
    dt = float(env.model.opt.timestep * env.action_repeat)
    stride = round(0.04 / dt)
    
    # Coordinates for overhead minimap: panel is [800..1120] in x, [160..608] in y.
    map_cx, map_cy = 840, 240
    map_scale = 32.0 # pixels per meter
    
    origin = env.data.qpos[:2].copy()
    
    def map_point(xy):
        px = float(map_cx + (xy[0] - origin[0]) * map_scale)
        py = float(map_cy - (xy[1] - origin[1]) * map_scale)
        return (px, py)
        
    # Phase 1: Straight cruise (1.5s)
    # Phase 2: Open-loop turn (4.5s) rotating heading at omega = v / R
    R = 2.0
    v = 1.2
    omega = v / R # 0.6 rad/s
    
    straight_dur = 1.5
    turn_dur = 4.5
    total_time = straight_dur + turn_dur
    
    print("Simulating straight phase...")
    for _ in range(round(straight_dur / dt)):
        targets = execute_skill("move", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, d_hat=[1, 0], speed=v)
        env.step(targets)
        
    curve_start_xy = env.data.qpos[:2].copy()
    target_center = np.array([curve_start_xy[0], curve_start_xy[1] - R])
    
    # Reset and run full recording
    env.reset(seed=42)
    for _ in range(60):
        env.step(execute_skill("jump_up", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, phase="stand"))
    origin = env.data.qpos[:2].copy()
    
    rows = []
    trail_actual = []
    preview_saved = False
    
    sim_t = 0.0
    turn_step = 0
    curve_start_xy = None
    target_center = None
    
    with imageio.get_writer(assets / "move-curve-openloop-drift.mp4", fps=25) as writer:
        total_steps = round(total_time / dt)
        for step in range(total_steps):
            sim_t = step * dt
            pos = env.data.qpos[:2].copy()
            vel = env.data.qvel[:2].copy()
            speed_cm_s = float(np.linalg.norm(vel) * 100)
            
            if sim_t < straight_dur:
                phase_name = "STRAIGHT CRUISE"
                d_cmd = np.array([1.0, 0.0])
                targets = execute_skill("move", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, d_hat=d_cmd, speed=v)
                env.step(targets)
                dist_from_center = None
                drift_m = 0.0
            else:
                phase_name = "OPEN-LOOP ROTATING HEADING"
                if curve_start_xy is None:
                    curve_start_xy = pos.copy()
                    target_center = np.array([curve_start_xy[0], curve_start_xy[1] - R])
                
                t_in_turn = (step - round(straight_dur / dt)) * dt
                angle = -omega * t_in_turn
                d_cmd = np.array([np.cos(angle), np.sin(angle)])
                targets = execute_skill("move", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, d_hat=d_cmd, speed=v)
                env.step(targets)
                
                dist_from_center = float(np.linalg.norm(pos - target_center))
                drift_m = float(dist_from_center - R)
                
            pt = map_point(pos)
            trail_actual.append(pt)
            
            rows.append({
                "time_s": sim_t,
                "x_m": float(pos[0]),
                "y_m": float(pos[1]),
                "vx_cm_s": float(vel[0] * 100),
                "vy_cm_s": float(vel[1] * 100),
                "speed_cm_s": speed_cm_s,
                "phase": phase_name,
                "drift_m": drift_m,
            })
            
            if step % stride == 0:
                camera.lookat = [pos[0], pos[1], float(env.data.qpos[2])]
                renderer.update_scene(env.data, camera)
                render_frame = renderer.render()
                
                canvas = Image.new("RGB", (1120, 608), (17, 24, 39))
                canvas.paste(Image.fromarray(render_frame), (0, 160))
                draw = ImageDraw.Draw(canvas)
                
                # Right panel background
                draw.rectangle([(800, 160), (1120, 608)], fill=(15, 23, 42))
                draw.line([(800, 160), (800, 608)], fill=(51, 65, 85), width=2)
                
                # Minimap header
                draw.text((820, 175), "PATH FROM ABOVE", font=large, fill=(255, 255, 255))
                
                # Draw intended target circle arc if initiated
                if target_center is not None:
                    # Draw target arc as dashed/dotted cyan line
                    c_pt = map_point(target_center)
                    r_px = R * map_scale
                    # Draw arc from top (angle pi/2) clockwise
                    bbox = [c_pt[0] - r_px, c_pt[1] - r_px, c_pt[0] + r_px, c_pt[1] + r_px]
                    # In PIL, arc angles are clockwise degrees from 3 oclock: 90 deg down, 270 deg up
                    draw.arc(bbox, start=-90, end=70, fill=(56, 189, 248), width=3)
                    draw.text((820, 210), f"Intended Arc (R = {R:.1f} m)", font=small, fill=(56, 189, 248))
                
                # Draw actual trail
                if len(trail_actual) > 1:
                    draw.line(trail_actual, fill=(248, 113, 113), width=4)
                
                # Current position dot
                cur_pt = map_point(pos)
                draw.ellipse([cur_pt[0]-4, cur_pt[1]-4, cur_pt[0]+4, cur_pt[1]+4], fill=(255, 255, 255), outline=(239, 68, 68), width=2)
                
                # Minimap legend & labels
                draw.text((820, 520), f"Actual: Outward Understeer", font=badge_font, fill=(248, 113, 113))
                if sim_t >= straight_dur:
                    draw.text((820, 545), f"Max Drift: +{drift_m*100:.0f} cm", font=large, fill=(239, 68, 68))
                    draw.text((820, 575), f"Radius: {dist_from_center:.2f} m (vs 2.0 m)", font=small, fill=(203, 213, 225))
                else:
                    draw.text((820, 550), f"+x east / +y north", font=small, fill=(148, 163, 184))
                    draw.text((820, 575), f"Open-loop heading rotation", font=small, fill=(148, 163, 184))

                # Top HUD:
                # Badges: [STRAIGHT CRUISE] [OPEN-LOOP ROTATING HEADING (NO FEEDBACK)]
                is_open = (sim_t >= straight_dur)
                b1_fill = (40, 50, 70) if is_open else (56, 189, 248)
                b1_txt = (200, 210, 230) if is_open else (15, 23, 42)
                b2_fill = (239, 68, 68) if is_open else (40, 50, 70)
                b2_txt = (255, 255, 255) if is_open else (200, 210, 230)
                
                draw.rounded_rectangle([(20, 20), (220, 55)], radius=6, fill=b1_fill)
                draw.text((35, 27), "STRAIGHT CRUISE", font=badge_font, fill=b1_txt)
                
                draw.rounded_rectangle([(235, 20), (590, 55)], radius=6, fill=b2_fill)
                draw.text((250, 27), "OPEN-LOOP ROTATING HEADING", font=badge_font, fill=b2_txt)
                
                if is_open:
                    draw.text((20, 70), f"Centrifugal Drift: understeers outward by +{drift_m*100:.0f} cm", font=large, fill=(255, 255, 255))
                    draw.text((20, 105), f"Speed: {speed_cm_s:.0f} cm/s   Push rotation: omega = {omega:.2f} rad/s (No radial feedback)", font=font, fill=(248, 113, 113))
                    draw.text((20, 132), f"Trajectory fails to hold arc: spiraling out to R = {dist_from_center:.2f} m", font=small, fill=(226, 232, 240))
                else:
                    draw.text((20, 70), "Straight cruise at nominal speed", font=large, fill=(255, 255, 255))
                    draw.text((20, 105), f"Speed: {speed_cm_s:.0f} cm/s   Heading: [1, 0]", font=font, fill=(56, 189, 248))
                    draw.text((20, 132), "Preparing open-loop steering test (omega = v / R = 0.6 rad/s)", font=small, fill=(203, 213, 225))
                
                # Elapsed time
                draw.text((800, 45), f"{sim_t:4.1f} / {total_time:.1f} s", font=large, fill=(255, 255, 255))
                draw.text((800, 80), "Real time", font=font, fill=(148, 163, 184))
                
                frame_np = np.asarray(canvas)
                writer.append_data(frame_np)
                
                # Save preview at 5.0s (showing dramatic drift)
                if not preview_saved and sim_t >= 5.0:
                    canvas.save(assets / "move-curve-openloop-drift-preview.png")
                    preview_saved = True
                    
    # Save CSV
    import csv
    with open(assets / "move-curve-openloop-drift.csv", "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
        
    # Save JSON summary
    summary = {
        "scenario": "open_loop_rotating_heading",
        "requested_speed_m_s": v,
        "target_radius_m": R,
        "omega_rad_s": omega,
        "straight_duration_s": straight_dur,
        "turn_duration_s": turn_dur,
        "final_drift_m": rows[-1]["drift_m"],
        "max_drift_m": max(r["drift_m"] for r in rows),
        "final_effective_radius_m": R + rows[-1]["drift_m"],
    }
    with open(assets / "move-curve-openloop-drift-results.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Open-loop drift video generated successfully: max drift = +{summary['max_drift_m']*100:.1f} cm")

if __name__ == "__main__":
    main()
