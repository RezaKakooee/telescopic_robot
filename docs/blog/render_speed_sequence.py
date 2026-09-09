"""Record normal -> fast -> normal -> slow -> normal without resetting.

Run: PYTHONPATH=. MUJOCO_GL=egl python docs/blog/render_speed_sequence.py
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
    sequence = [("NORMAL", 1.2), ("FAST", 2.0), ("NORMAL", 1.2),
                ("SLOW", 0.6), ("NORMAL", 1.2)]
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range = [0., 0.]
    cfg.scenario.goal.y_range = [-400., -400.]
    env = MujocoRadialSphereEnv(cfg, scenario=generate_scenario("goal", cfg, seed=42),
                               randomize=False, max_steps=10000)
    env.reset(seed=42)
    origin = env.data.qpos[:2].copy()
    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90, -20, 1.5
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    dt = float(env.model.opt.timestep * env.action_repeat)
    stage_steps, stride = round(5 / dt), round(0.04 / dt)
    rows, summaries = [], []
    try:
        with imageio.get_writer(assets / "move-speed-sequence.mp4", fps=25) as writer:
            for stage, (label, speed) in enumerate(sequence):
                speeds = []
                for step in range(stage_steps):
                    targets = skill_targets(
                        env, "move", d_hat=[1., 0.], speed=speed,
                        cross_track_error=float(env.data.qpos[1] - origin[1]),
                    )
                    _, _, terminated, truncated, _ = env.step(targets)
                    if terminated or truncated:
                        raise RuntimeError("Speed sequence terminated early")
                    vx = float(env.data.qvel[0]) * 100
                    speeds.append(vx)
                    elapsed = (stage * stage_steps + step + 1) * dt
                    rows.append([elapsed, stage + 1, speed * 100, vx,
                                 (env.data.qpos[0] - origin[0]) * 100,
                                 (env.data.qpos[1] - origin[1]) * 100])
                    if step % stride == 0:
                        camera.lookat[:] = env.data.qpos[:3]
                        renderer.update_scene(env.data, camera=camera)
                        canvas = Image.new("RGB", (800, 608), "#142032")
                        canvas.paste(Image.fromarray(renderer.render()), (0, 160))
                        draw = ImageDraw.Draw(canvas)
                        for k, (name, _) in enumerate(sequence):
                            x = 12 + k * 157
                            draw.rounded_rectangle((x, 12, x + 147, 53), radius=7,
                                                   fill="#f2b544" if k == stage else "#30425b")
                            draw.text((x + 12, 18), name, font=font,
                                      fill="#142032" if k == stage else "#ffffff")
                        draw.text((18, 66), f"Requested: {speed * 100:.0f} cm/s", font=large, fill="white")
                        measured = float(np.mean(speeds[-round(0.2 / dt):]))
                        draw.text((18, 108), f"Measured: {measured:.0f} cm/s", font=large, fill="#8ee5db")
                        draw.text((605, 75), f"{elapsed:4.1f} / 25 s", font=font, fill="white")
                        draw.text((605, 113), "Real time", font=font, fill="white")
                        writer.append_data(np.asarray(canvas))
                        if stage == 1 and step == stage_steps // 2 - 2:
                            canvas.save(assets / "move-speed-sequence-preview.png")
                summaries.append({"phase": label.lower(), "requested_cm_s": speed * 100,
                                  "mean_last_2s_cm_s": float(np.mean(speeds[-round(2 / dt):]))})
        np.savetxt(assets / "move-speed-sequence.csv", rows, delimiter=",", comments="",
                   header="time_s,phase,requested_cm_s,measured_cm_s,forward_cm,sideways_cm")
        report = {"phase_duration_s": 5, "duration_s": 25, "reset_between_phases": False,
                  "phases": summaries, "seed": 42, "mujoco_version": mujoco.__version__}
        (assets / "move-speed-sequence-results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
