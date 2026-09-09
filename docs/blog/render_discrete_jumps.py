"""Render script for Discrete Heading Jumps on a Curved Street demonstration.
Shows RoboBall attempting to navigate a curved street corner via discrete heading jumps
(e.g., +30 deg, +60 deg, +90 deg), causing a jagged polygonal trajectory,
momentum conflicts, lateral slip, and lane deviations.

Canvas: 1120x608
- Left panel (800x448): 3D tracking perspective.
- Right panel (320x448): Overhead minimap showing curved street lane vs polygonal trajectory.
- Top HUD (1120x160): Badges, telemetry, lateral error, and velocity drops.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import csv
import json
import math
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
    
    # 5-phase discrete jump sequence to turn 90 degrees right:
    # 1. Straight approach along +x (1.5s)
    # 2. Discrete Jump 1: +30 deg right (1.2s)
    # 3. Discrete Jump 2: +60 deg right (1.2s)
    # 4. Discrete Jump 3: +90 deg right (1.2s)
    # 5. Straight exit along -y (1.5s)
    phases = [
        ("STRAIGHT", 1.5, 0.0, "#48d8bb"),
        ("JUMP +30°", 1.2, 30.0, "#f97316"),
        ("JUMP +60°", 1.2, 60.0, "#eab308"),
        ("JUMP +90°", 1.2, 90.0, "#ef4444"),
        ("STRAIGHT", 1.5, 90.0, "#48d8bb"),
    ]
    
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
    badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 23)
    
    dt = float(env.model.opt.timestep * env.action_repeat)
    stride = round(0.04 / dt)
    
    origin = env.data.qpos[:2].copy()
    
    # Coordinates for overhead minimap: panel is [800..1120] in x, [160..608] in y.
    map_cx, map_cy = 850, 240
    map_scale = 35.0 # pixels per meter
    
    def map_point(xy):
        px = float(map_cx + (xy[0] - origin[0]) * map_scale)
        py = float(map_cy - (xy[1] - origin[1]) * map_scale)
        return (px, py)
        
    R_ref = 2.2 # Nominal curved corner radius
    speed = 1.2
    
    rows = []
    trails = []
    total_time = sum(p[1] for p in phases)
    sim_t = 0.0
    preview_saved = False
    
    # Store reference curved road landmark once curve begins
    curve_start_xy = None
    ref_center = None
    
    phase_metrics = []
    
    with imageio.get_writer(assets / "move-curve-discrete-jumps.mp4", fps=25) as writer:
        for p_idx, (p_label, p_dur, p_angle_deg, p_color) in enumerate(phases):
            n_steps = round(p_dur / dt)
            rad = np.radians(-p_angle_deg) # rightward turn
            d_cmd = np.array([np.cos(rad), np.sin(rad)])
            
            p_trail = []
            trails.append((p_trail, p_color, p_label))
            p_speeds = []
            p_errors = []
            
            for step in range(n_steps):
                pos = env.data.qpos[:2].copy()
                vel = env.data.qvel[:2].copy()
                current_speed = float(np.linalg.norm(vel) * 100)
                p_speeds.append(current_speed)
                
                # Check reference center when first turn jump occurs
                if p_idx == 1 and curve_start_xy is None:
                    curve_start_xy = pos.copy()
                    ref_center = np.array([curve_start_xy[0], curve_start_xy[1] - R_ref])
                
                if ref_center is not None and p_idx in (1, 2, 3):
                    cur_r = float(np.linalg.norm(pos - ref_center))
                    dev_error_m = cur_r - R_ref
                else:
                    dev_error_m = 0.0
                p_errors.append(dev_error_m)
                
                targets = execute_skill("move", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                                        d_hat=d_cmd, speed=speed)
                env.step(targets)
                
                pt = map_point(pos)
                p_trail.append(pt)
                
                rows.append({
                    "time_s": sim_t,
                    "phase": p_label,
                    "angle_deg": p_angle_deg,
                    "x_m": float(pos[0]),
                    "y_m": float(pos[1]),
                    "vx_cm_s": float(vel[0] * 100),
                    "vy_cm_s": float(vel[1] * 100),
                    "speed_cm_s": current_speed,
                    "lane_error_cm": dev_error_m * 100,
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
                    
                    # Draw reference smooth curved street lane if started
                    if ref_center is not None:
                        c_pt = map_point(ref_center)
                        r_px = R_ref * map_scale
                        bbox = [c_pt[0] - r_px, c_pt[1] - r_px, c_pt[0] + r_px, c_pt[1] + r_px]
                        draw.arc(bbox, start=-90, end=0, fill=(56, 189, 248), width=3)
                        draw.text((820, 205), f"Smooth Road Arc (R = {R_ref:.1f} m)", font=small, fill=(56, 189, 248))
                    
                    # Draw all trail segments
                    for tr, col, _ in trails:
                        if len(tr) > 1:
                            draw.line(tr, fill=col, width=4)
                            
                    # Current position
                    cur_pt = map_point(pos)
                    draw.ellipse([cur_pt[0]-4, cur_pt[1]-4, cur_pt[0]+4, cur_pt[1]+4],
                                 fill=(255, 255, 255), outline=(239, 68, 68), width=2)
                                 
                    # Minimap labels
                    draw.text((820, 520), "Jagged Polygonal Path", font=badge_font, fill=(249, 115, 22))
                    if p_idx in (1, 2, 3):
                        draw.text((820, 545), f"Lane Dev: {dev_error_m*100:+.0f} cm", font=large, fill=(239, 68, 68))
                        draw.text((820, 575), f"Heading Jump: +{p_angle_deg:.0f}°", font=small, fill=(203, 213, 225))
                    else:
                        draw.text((820, 550), "+x east / +y north", font=small, fill=(148, 163, 184))
                        draw.text((820, 575), "Discrete 30° turn jumps", font=small, fill=(148, 163, 184))
                        
                    # Top HUD: 5 badges
                    badge_xs = [20, 150, 295, 440, 585]
                    for b_i, (b_name, _, _, b_col) in enumerate(phases):
                        bx = badge_xs[b_i]
                        bw = 120 if "JUMP" in b_name else 115
                        is_act = (b_i == p_idx)
                        bg = b_col if is_act else (40, 50, 70)
                        tx = (15, 23, 42) if is_act else (200, 210, 230)
                        draw.rounded_rectangle([(bx, 20), (bx + bw, 55)], radius=6, fill=bg)
                        draw.text((bx + 10, 28), b_name, font=badge_font, fill=tx)
                        
                    if "JUMP" in p_label:
                        draw.text((20, 70), f"Sharp Heading Switch to +{p_angle_deg:.0f}°: fighting momentum", font=large, fill=(255, 255, 255))
                        draw.text((20, 105), f"Speed: {current_speed:.0f} cm/s   Lane Deviation: {dev_error_m*100:+.0f} cm", font=font, fill=(249, 115, 22))
                        draw.text((20, 132), "Ground contact forces abruptly redirect, causing lateral slip and corner overshoots", font=small, fill=(226, 232, 240))
                    else:
                        draw.text((20, 70), "Straight cruise along street corridor", font=large, fill=(255, 255, 255))
                        draw.text((20, 105), f"Speed: {current_speed:.0f} cm/s   Command: straight", font=font, fill=(56, 189, 248))
                        draw.text((20, 132), "Approaching curved bend via piecewise discrete heading turns", font=small, fill=(203, 213, 225))

                    # Elapsed time
                    draw.text((800, 45), f"{sim_t:4.1f} / {total_time:.1f} s", font=large, fill=(255, 255, 255))
                    draw.text((800, 80), "Real time", font=font, fill=(148, 163, 184))
                    
                    frame_np = np.asarray(canvas)
                    writer.append_data(frame_np)
                    
                    if not preview_saved and sim_t >= 4.6: # inside JUMP +90° showing lane deviation telemetry
                        canvas.save(assets / "move-curve-discrete-jumps-preview.png")
                        preview_saved = True
                        
                sim_t += dt
                
            phase_metrics.append({
                "phase": p_label,
                "angle_deg": p_angle_deg,
                "duration_s": p_dur,
                "mean_speed_cm_s": float(np.mean(p_speeds)),
                "min_speed_cm_s": float(np.min(p_speeds)),
                "max_speed_cm_s": float(np.max(p_speeds)),
                "end_pos_m": [float(pos[0]), float(pos[1])],
            })
            
    # Save CSV
    with open(assets / "move-curve-discrete-jumps.csv", "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
        
    # Save JSON summary
    summary = {
        "scenario": "discrete_heading_jumps",
        "commanded_speed_m_s": speed,
        "total_duration_s": total_time,
        "nominal_curve_radius_m": R_ref,
        "phases": phase_metrics,
        "max_lane_deviation_cm": float(max(abs(r["lane_error_cm"]) for r in rows)),
    }
    with open(assets / "move-curve-discrete-jumps-results.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Discrete jumps video generated successfully: max lane deviation = {summary['max_lane_deviation_cm']:.1f} cm")

if __name__ == "__main__":
    main()
