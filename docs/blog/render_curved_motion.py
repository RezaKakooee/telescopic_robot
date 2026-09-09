"""Record continuous curvy street demonstration: straight -> curve right -> straight -> curve left -> straight.

Renders side-by-side: 3D tracking perspective on the left (800x448) + overhead path trail on the right (320x448).
Canvas size: 1120x608.

Run:
PYTHONPATH=. MUJOCO_GL=egl /home/azureuser/miniconda3/envs/roboverse/bin/python docs/blog/render_curved_motion.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import json
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
    
    # 5-phase winding street sequence:
    # 1. Straight road along +x
    # 2. Right curve (R = 2.0 m)
    # 3. Intermediate straight road
    # 4. Left curve (R = 2.0 m)
    # 5. Exit straight road
    phases_cfg = [
        ("STRAIGHT", "straight", 1.8, None, None, "#48d8bb"),
        ("CURVE RIGHT", "curve", 3.0, 2.0, "right", "#ffbd59"),
        ("STRAIGHT", "straight", 1.5, None, None, "#48d8bb"),
        ("CURVE LEFT", "curve", 3.0, 2.0, "left", "#c49aff"),
        ("STRAIGHT", "straight", 1.8, None, None, "#48d8bb"),
    ]
    
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range, cfg.scenario.goal.y_range = [0., 0.], [-400., -400.]
    env = MujocoRadialSphereEnv(cfg, scenario=generate_scenario("goal", cfg, seed=42),
                               randomize=False, max_steps=10000)
    env.reset(seed=42)
    
    # Settle stance
    for _ in range(100):
        env.step(execute_skill("jump_up", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, phase="stand"))
        
    origin = env.data.qpos[:2].copy()
    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90, -45, 2.2
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 17)
    badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    
    dt = float(env.model.opt.timestep * env.action_repeat)
    stride = round(0.04 / dt)
    
    # Trail mapping coordinates for the right panel: x in [800, 1120], y in [160, 608]
    # Center map origin around (960, 230)
    # Scale: ~24 pixels per meter
    map_cx, map_cy = 920, 210
    map_scale = 23.0
    
    def map_point(xy):
        # x is rightward in world -> rightward on map
        # y is forward/backward -> -y is downward on map
        px = float(map_cx + (xy[0] - origin[0]) * map_scale)
        py = float(map_cy - (xy[1] - origin[1]) * map_scale)
        return (px, py)
        
    d_curr = np.array([1.0, 0.0], dtype=np.float64)
    center = None
    trails = []
    rows = []
    summaries = []
    total_time = 0.0
    preview_saved = False
    
    try:
        with imageio.get_writer(assets / "move-curved-street.mp4", fps=25) as writer:
            for p_idx, (label, mode, duration, radius, direction, color) in enumerate(phases_cfg):
                n_steps = round(duration / dt)
                phase_origin = env.data.qpos[:2].copy()
                trail = [map_point(phase_origin)]
                trails.append((trail, color))
                velocities = []
                center = None
                
                for step in range(n_steps):
                    pos = env.data.qpos[:2].copy()
                    quat = env.data.qpos[3:7].copy()
                    vel = env.data.qvel[:2].copy()
                    speed_now = float(np.linalg.norm(vel))
                    velocities.append(speed_now)
                    
                    if mode == "straight":
                        targets = execute_skill(
                            "move", quat, env.dirs_body, env.max_extend,
                            d_hat=d_curr, speed=1.2, rod_mechanism="multi_stage"
                        )
                    else:  # curve
                        targets = execute_skill(
                            "curve", quat, env.dirs_body, env.max_extend,
                            d_hat=d_curr, radius=radius, direction=direction,
                            speed=1.2, ball_xy=pos, center_xy=center,
                            rod_mechanism="multi_stage"
                        )
                        # Establish instantaneous center if not set
                        if center is None:
                            n_in = np.array([d_curr[1], -d_curr[0]]) if direction == "right" else np.array([-d_curr[1], d_curr[0]])
                            center = pos + n_in * radius
                        # Update current heading tangent
                        rel = pos - center
                        rot_dir = -1.0 if direction == "right" else +1.0
                        th = np.arctan2(rel[1], rel[0]) + rot_dir * (np.pi / 2.0)
                        d_curr = np.array([np.cos(th), np.sin(th)])
                        
                    _, _, term, trunc, _ = env.step(targets)
                    if term or trunc:
                        raise RuntimeError("Curved street recording ended early")
                        
                    total_time += dt
                    elapsed_phase = (step + 1) * dt
                    
                    rows.append([
                        total_time, p_idx + 1, label, pos[0] - origin[0], pos[1] - origin[1],
                        vel[0] * 100.0, vel[1] * 100.0, speed_now * 100.0
                    ])
                    
                    if step % stride != 0:
                        continue
                        
                    trail.append(map_point(pos))
                    
                    # Update camera tracking
                    camera.lookat[0] = env.data.qpos[0]
                    camera.lookat[1] = env.data.qpos[1]
                    camera.lookat[2] = 0.20
                    renderer.update_scene(env.data, camera=camera)
                    
                    # Composite canvas: 1120x608
                    canvas = Image.new("RGB", (1120, 608), "#142032")
                    canvas.paste(Image.fromarray(renderer.render()), (0, 160))
                    draw = ImageDraw.Draw(canvas)
                    
                    # Top badges (5 phases)
                    for k, (b_name, _, _, _, _, b_col) in enumerate(phases_cfg):
                        bx = 12 + k * 156
                        is_act = (k == p_idx)
                        draw.rounded_rectangle(
                            (bx, 12, bx + 146, 52), radius=7,
                            fill=b_col if is_act else "#30425b"
                        )
                        draw.text(
                            (bx + 10, 20), b_name, font=badge_font,
                            fill="#142032" if is_act else "#ffffff"
                        )
                        
                    # Telemetry text on left
                    if mode == "straight":
                        desc = "Straight road: cruise along street centerline"
                    elif direction == "right":
                        desc = f"Right curve: carving R = {radius:.1f} m bend"
                    else:
                        desc = f"Left curve: carving R = {radius:.1f} m counter-bend"
                        
                    draw.text((18, 66), desc, font=large, fill="white")
                    mean_v = np.mean(velocities[-20:]) * 100.0
                    draw.text((18, 108), f"Speed: {mean_v:.0f} cm/s   Path: SMOOTH CURVED LINE", font=font, fill="#8ee5db")
                    
                    # Telemetry on right
                    draw.text((820, 24), f"{total_time:4.1f} / 11.1 s", font=large, fill="white")
                    draw.text((820, 70), "Real time", font=font, fill="white")
                    
                    # Minimap / overhead panel header
                    draw.line([(800, 160), (800, 608)], fill="#223348", width=2)
                    draw.text((820, 172), "PATH FROM ABOVE", font=large, fill="white")
                    
                    # Draw path trail
                    for tr_pts, tr_col in trails:
                        if len(tr_pts) > 1:
                            draw.line(tr_pts, fill=tr_col, width=4)
                            
                    # Current ball marker on minimap
                    cur_px, cur_py = map_point(pos)
                    draw.ellipse((cur_px - 6, cur_py - 6, cur_px + 6, cur_py + 6), fill="white", outline="#ffbd59", width=2)
                    
                    # Heading arrow on minimap
                    arr_len = 22.0
                    tip_x = cur_px + d_curr[0] * arr_len
                    tip_y = cur_py - d_curr[1] * arr_len
                    draw.line((cur_px, cur_py, tip_x, tip_y), fill="#ff5252", width=3)
                    
                    # Minimap annotations
                    draw.text((820, 520), "+x east / +y north", font=small, fill="#8ee5db")
                    draw.text((820, 546), "Winding city street S-curve", font=small, fill="white")
                    draw.text((820, 572), "Continuous run (no resets)", font=small, fill="#ffbd59")
                    
                    writer.append_data(np.asarray(canvas))
                    
                    # Capture preview image during the left curve
                    if p_idx == 3 and step == round(2.0 / dt) and not preview_saved:
                        canvas.save(assets / "move-curved-street-preview.png")
                        preview_saved = True
                        
                summaries.append({
                    "phase": label,
                    "mode": mode,
                    "duration_s": duration,
                    "radius_m": radius,
                    "direction": direction,
                    "mean_speed_cm_s": float(np.mean(velocities[-round(0.5/dt):]) * 100.0),
                    "end_position_m": (env.data.qpos[:2] - origin).tolist(),
                })
                
        if not preview_saved:
            canvas.save(assets / "move-curved-street-preview.png")
            
        np.savetxt(
            assets / "move-curved-street.csv", rows, delimiter=",", fmt="%s", comments="",
            header="time_s,phase_idx,phase_name,x_m,y_m,vx_cm_s,vy_cm_s,speed_cm_s"
        )
        report = {
            "skill": "curve",
            "seed": 42,
            "duration_s": total_time,
            "phases": summaries,
        }
        (assets / "move-curved-street-results.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Curved street demonstration completed successfully:")
        print(json.dumps(report, indent=2))
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
