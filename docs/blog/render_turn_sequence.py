"""Continuous straight -> right -> left demo, with a world-coordinate trail.

Run: PYTHONPATH=. MUJOCO_GL=egl python docs/blog/render_turn_sequence.py
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
from skills.runner import skill_targets


def main():
    assets = assets_dir()
    sequence = [("STRAIGHT", 0.), ("RIGHT", 90.), ("LEFT", -90.)]
    colors = ["#48d8bb", "#ffbd59", "#c49aff"]
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range, cfg.scenario.goal.y_range = [0., 0.], [-400., -400.]
    env = MujocoRadialSphereEnv(cfg, scenario=generate_scenario("goal", cfg, seed=42),
                               randomize=False, max_steps=10000)
    env.reset(seed=42)
    origin = env.data.qpos[:2].copy()
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = 800, 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90, -65, 1.8
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 19)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    dt = float(env.model.opt.timestep * env.action_repeat)
    phase_steps, stride = round(4 / dt), round(.04 / dt)
    reference = np.array([1., 0.])
    trails, rows, summaries = [], [], []

    def map_point(xy):
        return (float(840 + (xy[0] + 1) * 22), float(272 - xy[1] * 22))

    try:
        with imageio.get_writer(assets / "move-turn-sequence.mp4", fps=25) as writer:
            for phase, (label, angle_deg) in enumerate(sequence):
                turn = -np.deg2rad(angle_deg)
                c, s = np.cos(turn), np.sin(turn)
                heading = np.array([[c, -s], [s, c]]) @ reference
                normal = np.array([-heading[1], heading[0]])
                phase_origin = env.data.qpos[:2].copy()
                trail = [map_point(phase_origin - origin)]
                trails.append(trail)
                velocities = []
                for step in range(phase_steps):
                    error = float((env.data.qpos[:2] - phase_origin) @ normal)
                    targets = skill_targets(env, "turn", d_hat=reference, angle_deg=angle_deg,
                                            speed=1.2, cross_track_error=error)
                    _, _, terminated, truncated, _ = env.step(targets)
                    if terminated or truncated:
                        raise RuntimeError("Turn demonstration ended early")
                    xy = env.data.qpos[:2] - origin
                    vel = env.data.qvel[:2].copy()
                    velocities.append(vel)
                    elapsed = (phase * phase_steps + step + 1) * dt
                    rows.append([elapsed, phase + 1, *heading, *(xy * 100), *(vel * 100)])
                    if step % stride != 0:
                        continue
                    trail.append(map_point(xy))
                    camera.lookat[:] = env.data.qpos[:3]
                    renderer.update_scene(env.data, camera=camera)
                    canvas = Image.new("RGB", (1120, 608), "#142032")
                    canvas.paste(Image.fromarray(renderer.render()), (0, 160))
                    draw = ImageDraw.Draw(canvas)
                    for k, (name, _) in enumerate(sequence):
                        x = 12 + k * 260
                        draw.rounded_rectangle((x, 12, x + 248, 54), radius=7,
                                               fill=colors[k] if k == phase else "#30425b")
                        draw.text((x + 18, 17), name, font=large,
                                  fill="#142032" if k == phase else "white")
                    command = ["Along world +x", "90 degrees right: toward -y",
                               "90 degrees left: toward +x"][phase]
                    draw.text((18, 66), command, font=large, fill="white")
                    measured = np.linalg.norm(np.mean(velocities[-20:], axis=0)) * 100
                    draw.text((18, 108), f"Speed: {measured:.0f} cm/s  |  Requested: 120 cm/s",
                              font=font, fill="#8ee5db")
                    draw.text((842, 24), f"{elapsed:4.1f} / 12 s", font=large, fill="white")
                    draw.text((842, 70), "Real time", font=font, fill="white")
                    draw.text((824, 176), "Path from above", font=large, fill="white")
                    for k, points in enumerate(trails):
                        if len(points) > 1:
                            draw.line(points, fill=colors[k], width=4)
                    px, py = map_point(xy)
                    draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill="white")
                    tip = np.array([px, py]) + np.array([heading[0], -heading[1]]) * 26
                    draw.line((px, py, *tip), fill="#ff6262", width=3)
                    direction = np.array([heading[0], -heading[1]])
                    perp = np.array([-direction[1], direction[0]])
                    draw.polygon([tuple(tip), tuple(tip - direction * 8 + perp * 5),
                                  tuple(tip - direction * 8 - perp * 5)], fill="#ff6262")
                    draw.text((824, 454), "Arrow = requested direction", font=small, fill="#ff9292")
                    draw.text((824, 486), "+x right / +y up", font=font, fill="white")
                    draw.text((824, 528), "Continuous run", font=font, fill="white")
                    draw.text((824, 562), "No resets", font=font, fill="white")
                    writer.append_data(np.asarray(canvas))
                    if phase == 2 and step == 300:
                        canvas.save(assets / "move-turn-sequence-preview.png")
                avg = np.mean(velocities[-round(1 / dt):], axis=0)
                alignment = np.degrees(np.arctan2(float(avg @ normal), float(avg @ heading)))
                summaries.append({"phase": label.lower(), "angle_deg_right_positive": angle_deg,
                                  "requested_heading": heading.tolist(),
                                  "mean_final_second_velocity_cm_s": (avg * 100).tolist(),
                                  "final_second_direction_error_degrees": float(alignment)})
                reference = heading
        np.savetxt(assets / "move-turn-sequence.csv", rows, delimiter=",", comments="",
                   header="time_s,phase,heading_x,heading_y,x_cm,y_cm,vx_cm_s,vy_cm_s")
        report = {"duration_s": 12, "phase_duration_s": 4, "reset_between_phases": False,
                  "phases": summaries, "seed": 42, "mujoco_version": mujoco.__version__}
        (assets / "move-turn-sequence-results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
