"""Record low, medium, and high jumps for both standstill and moving jumps at 2x slow motion.

Generates:
- docs/blog/assets/jump-height-standing.mp4 (+ preview, csv, json)
- docs/blog/assets/jump-height-moving.mp4   (+ preview, csv, json)

Run:
PYTHONPATH=. MUJOCO_GL=egl /home/azureuser/miniconda3/envs/roboverse/bin/python docs/blog/render_jump_height.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
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
from skills.runner import skill_targets


def render_standing_jump_comparison():
    assets = assets_dir()
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range, cfg.scenario.goal.y_range = [0., 0.], [-400., -400.]
    env = MujocoRadialSphereEnv(cfg, scenario=generate_scenario("goal", cfg, seed=42),
                               randomize=False, max_steps=10000)
    env.reset(seed=42)

    # Stand settle
    for _ in range(100):
        env.step(execute_skill("jump_up", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, phase="stand"))

    origin = env.data.qpos[:3].copy()
    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90, -10, 2.3
    camera.lookat[:] = [origin[0], origin[1], 0.65]

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

    jump_levels = [
        ("LOW (20 cm)", 20.0),
        ("MEDIUM (35 cm)", 35.0),
        ("HIGH (50 cm)", 50.0),
    ]

    dt = float(env.model.opt.timestep * env.action_repeat)
    video_stride = max(1, round(1 / (25 * 2 * dt)))  # 2x slow motion

    rows = []
    results_summary = []
    total_time = 0.0
    preview_frame = None

    try:
        with imageio.get_writer(assets / "jump-height-standing.mp4", fps=25) as writer:
            for idx, (label, target_h) in enumerate(jump_levels):
                # Phase timing for each jump in seconds:
                # 0.0 - 0.4s: stand/settle
                # 0.4 - 0.6s: crouch
                # 0.6 - 0.72s: takeoff
                # 0.72s - descent: airborne
                # touchdown to end: landing + settle
                t_jump = 0.0
                phase = "stand"
                airborne_seen = False
                landed = False
                peak_z = env.data.qpos[2]
                cycle_start_z = env.data.qpos[2]

                # We run each jump cycle for ~2.6 seconds (260 steps)
                n_cycle_steps = round(2.6 / dt)
                for step in range(n_cycle_steps):
                    t_in_cycle = step * dt

                    if t_in_cycle < 0.40:
                        phase = "stand"
                        cycle_start_z = env.data.qpos[2]
                    elif t_in_cycle < 0.60:
                        phase = "crouch"
                    elif t_in_cycle < 0.72:
                        phase = "takeoff"
                    elif not landed:
                        if airborne_seen and env.data.qvel[2] < 0 and env.data.qpos[2] < 0.30:
                            phase = "landing"
                        else:
                            phase = "airborne"
                    else:
                        phase = "stand"

                    targets = execute_skill(
                        "jump_up", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                        phase=phase, jump_height_cm=target_h
                    )
                    env.step(targets)

                    grounded = any(
                        env.floor_geom_id in (c.geom1, c.geom2)
                        and (c.geom1 in env.robot_geom_ids or c.geom2 in env.robot_geom_ids)
                        and c.dist <= 0 for c in env.data.contact
                    )
                    if not grounded and env.data.qpos[2] > cycle_start_z + 0.08:
                        airborne_seen = True
                    if phase == "landing" and grounded:
                        landed = True

                    z = float(env.data.qpos[2])
                    vz = float(env.data.qvel[2])
                    if t_in_cycle >= 0.60:
                        peak_z = max(peak_z, z)

                    total_time += dt
                    rise_now = (peak_z - cycle_start_z) * 100.0

                    rows.append([
                        total_time, idx, target_h, phase,
                        z * 100.0, vz * 100.0, rise_now, int(grounded)
                    ])

                    # Render frame
                    renderer.update_scene(env.data, camera=camera)
                    canvas = Image.new("RGB", (800, 608), "#142032")
                    canvas.paste(Image.fromarray(renderer.render()), (0, 160))
                    draw = ImageDraw.Draw(canvas)

                    # Top badges for 3 levels
                    for k, (name, _) in enumerate(jump_levels):
                        bx = 12 + k * 262
                        is_active = (k == idx)
                        draw.rounded_rectangle(
                            (bx, 12, bx + 252, 53), radius=7,
                            fill="#f2b544" if is_active else "#30425b"
                        )
                        draw.text(
                            (bx + 20, 19), name, font=badge_font,
                            fill="#142032" if is_active else "#ffffff"
                        )

                    # Telemetry text
                    draw.text((18, 66), f"Target: {target_h:.0f} cm  |  Rise: {rise_now:.1f} cm", font=large, fill="white")
                    draw.text((18, 108), f"Phase: {phase.upper()}   vz: {vz * 100:+.0f} cm/s", font=large, fill="#8ee5db")
                    draw.text((590, 66), f"Sim: {total_time:.2f} s", font=font, fill="white")
                    draw.text((590, 108), "2x slow motion", font=font, fill="white")

                    if step % video_stride == 0:
                        writer.append_data(np.asarray(canvas))

                    # Capture a high jump apex preview
                    if idx == 2 and phase == "airborne" and vz < 0.2 and preview_frame is None:
                        preview_frame = canvas.copy()

                measured_rise = (peak_z - cycle_start_z) * 100.0
                results_summary.append({
                    "level": label,
                    "requested_height_cm": target_h,
                    "measured_rise_cm": float(measured_rise),
                    "peak_core_height_cm": float(peak_z * 100.0),
                    "error_cm": float(measured_rise - target_h),
                })

        if preview_frame is not None:
            preview_frame.save(assets / "jump-height-standing-preview.png")
        else:
            canvas.save(assets / "jump-height-standing-preview.png")

        np.savetxt(
            assets / "jump-height-standing.csv", rows, delimiter=",", fmt="%s", comments="",
            header="time_s,jump_idx,requested_height_cm,phase,height_cm,vz_cm_s,measured_rise_cm,ground_contact"
        )
        report = {
            "skill": "jump_up",
            "seed": 42,
            "video_slowdown": 2,
            "jumps": results_summary,
        }
        (assets / "jump-height-standing-results.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Standing jump results:")
        print(json.dumps(report, indent=2))
    finally:
        renderer.close()
        env.close()


def render_moving_jump_comparison():
    assets = assets_dir()
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range, cfg.scenario.goal.y_range = [0., 0.], [-400., -400.]
    env = MujocoRadialSphereEnv(cfg, scenario=generate_scenario("goal", cfg, seed=42),
                               randomize=False, max_steps=10000)
    env.reset(seed=42)

    # Stand settle then initial runup
    for _ in range(150):
        env.step(execute_skill("jump_up", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, phase="stand"))

    origin = env.data.qpos[:3].copy()
    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90, -10, 2.3

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

    jump_levels = [
        ("LOW (20 cm)", 20.0),
        ("MEDIUM (35 cm)", 35.0),
        ("HIGH (50 cm)", 50.0),
    ]

    dt = float(env.model.opt.timestep * env.action_repeat)
    video_stride = max(1, round(1 / (25 * 2 * dt)))  # 2x slow motion

    rows = []
    results_summary = []
    total_time = 0.0
    preview_frame = None

    try:
        with imageio.get_writer(assets / "jump-height-moving.mp4", fps=25) as writer:
            for idx, (label, target_h) in enumerate(jump_levels):
                # Sequence for moving jump:
                # 0.0 - 1.4s: run forward with move skill (120 cm/s)
                # 1.4 - 1.47s: dip
                # 1.47 - 1.60s: launch (scaled by target_h)
                # 1.60s - descent: airborne
                # descent to touchdown: landing
                # touchdown to 2.8s: resume move forward (120 cm/s)
                phase = "move"
                airborne_seen = False
                landed = False
                peak_z = env.data.qpos[2]
                cycle_start_z = env.data.qpos[2]

                n_cycle_steps = round(2.8 / dt)
                for step in range(n_cycle_steps):
                    t_in_cycle = step * dt
                    quat = env.data.qpos[3:7].copy()

                    if t_in_cycle < 1.40:
                        phase = "move"
                        cycle_start_z = env.data.qpos[2]
                        targets = skill_targets(
                            env, "move", d_hat=[1., 0.], speed=1.2,
                            cross_track_error=float(env.data.qpos[1] - origin[1])
                        )
                    elif t_in_cycle < 1.47:
                        phase = "dip"
                        targets = execute_skill("jump_forward_while_moving", quat, env.dirs_body, env.max_extend,
                                                d_hat=np.array([1., 0.]), phase="dip")
                    elif t_in_cycle < 1.60:
                        phase = "launch"
                        targets = execute_skill("jump_forward_while_moving", quat, env.dirs_body, env.max_extend,
                                                d_hat=np.array([1., 0.]), phase="launch", jump_height_cm=target_h)
                    elif not landed:
                        if airborne_seen and env.data.qvel[2] < 0 and env.data.qpos[2] < 0.32:
                            phase = "landing"
                            targets = execute_skill("jump_forward_while_moving", quat, env.dirs_body, env.max_extend,
                                                    d_hat=np.array([1., 0.]), phase="landing")
                        else:
                            phase = "airborne"
                            targets = execute_skill("jump_forward_while_moving", quat, env.dirs_body, env.max_extend,
                                                    d_hat=np.array([1., 0.]), phase="airborne")
                    else:
                        phase = "move"
                        targets = skill_targets(
                            env, "move", d_hat=[1., 0.], speed=1.2,
                            cross_track_error=float(env.data.qpos[1] - origin[1])
                        )

                    env.step(targets)

                    grounded = any(
                        env.floor_geom_id in (c.geom1, c.geom2)
                        and (c.geom1 in env.robot_geom_ids or c.geom2 in env.robot_geom_ids)
                        and c.dist <= 0 for c in env.data.contact
                    )
                    if not grounded and env.data.qpos[2] > cycle_start_z + 0.08:
                        airborne_seen = True
                    if phase == "landing" and grounded:
                        landed = True

                    z = float(env.data.qpos[2])
                    vz = float(env.data.qvel[2])
                    vx = float(env.data.qvel[0]) * 100.0

                    if t_in_cycle >= 1.47:
                        peak_z = max(peak_z, z)

                    total_time += dt
                    rise_now = (peak_z - cycle_start_z) * 100.0

                    rows.append([
                        total_time, idx, target_h, phase,
                        z * 100.0, vz * 100.0, vx, rise_now, int(grounded)
                    ])

                    # Smooth camera follow
                    camera.lookat[0] = env.data.qpos[0]
                    camera.lookat[1] = env.data.qpos[1]
                    camera.lookat[2] = 0.65

                    renderer.update_scene(env.data, camera=camera)
                    canvas = Image.new("RGB", (800, 608), "#142032")
                    canvas.paste(Image.fromarray(renderer.render()), (0, 160))
                    draw = ImageDraw.Draw(canvas)

                    # Top badges for 3 levels
                    for k, (name, _) in enumerate(jump_levels):
                        bx = 12 + k * 262
                        is_active = (k == idx)
                        draw.rounded_rectangle(
                            (bx, 12, bx + 252, 53), radius=7,
                            fill="#f2b544" if is_active else "#30425b"
                        )
                        draw.text(
                            (bx + 20, 19), name, font=badge_font,
                            fill="#142032" if is_active else "#ffffff"
                        )

                    # Telemetry text
                    draw.text((18, 66), f"Target: {target_h:.0f} cm  |  Rise: {rise_now:.1f} cm", font=large, fill="white")
                    draw.text((18, 108), f"vx: {vx:+.0f} cm/s   vz: {vz * 100:+.0f} cm/s", font=large, fill="#8ee5db")
                    draw.text((590, 66), f"Sim: {total_time:.2f} s", font=font, fill="white")
                    draw.text((590, 108), "2x slow motion", font=font, fill="white")

                    if step % video_stride == 0:
                        writer.append_data(np.asarray(canvas))

                    # Capture high jump apex preview
                    if idx == 2 and phase == "airborne" and vz < 0.2 and preview_frame is None:
                        preview_frame = canvas.copy()

                measured_rise = (peak_z - cycle_start_z) * 100.0
                results_summary.append({
                    "level": label,
                    "requested_height_cm": target_h,
                    "measured_rise_cm": float(measured_rise),
                    "peak_core_height_cm": float(peak_z * 100.0),
                    "error_cm": float(measured_rise - target_h),
                })

        if preview_frame is not None:
            preview_frame.save(assets / "jump-height-moving-preview.png")
        else:
            canvas.save(assets / "jump-height-moving-preview.png")

        np.savetxt(
            assets / "jump-height-moving.csv", rows, delimiter=",", fmt="%s", comments="",
            header="time_s,jump_idx,requested_height_cm,phase,height_cm,vz_cm_s,vx_cm_s,measured_rise_cm,ground_contact"
        )
        report = {
            "skill": "jump_forward_while_moving",
            "seed": 42,
            "video_slowdown": 2,
            "jumps": results_summary,
        }
        (assets / "jump-height-moving-results.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Moving jump results:")
        print(json.dumps(report, indent=2))
    finally:
        renderer.close()
        env.close()


def main():
    print("Recording standing jump comparison...")
    render_standing_jump_comparison()
    print("\nRecording moving jump comparison...")
    render_moving_jump_comparison()
    print("\nAll jump height comparison videos successfully rendered!")


if __name__ == "__main__":
    main()
