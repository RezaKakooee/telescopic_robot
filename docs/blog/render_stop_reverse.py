"""Record forward -> stop -> reverse -> stop without resetting the robot.

PYTHONPATH=. MUJOCO_GL=egl python docs/blog/render_stop_reverse.py
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
from skills.runner import skill_targets


def main():
    assets = Path(__file__).resolve().parent / "assets"
    sequence = [("FORWARD", 1.2), ("STOP", 0.),
                ("REVERSE", -1.2), ("STOP", 0.)]
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
    camera.azimuth, camera.elevation, camera.distance = 90, -20, 1.5
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 27)
    dt = float(env.model.opt.timestep * env.action_repeat)
    stage_steps, stride = round(4 / dt), round(.04 / dt)
    rows, summaries = [], []
    try:
        with imageio.get_writer(assets / "move-stop-reverse.mp4", fps=25) as writer:
            for phase, (label, speed) in enumerate(sequence):
                start = env.data.qpos[:2].copy()
                velocities = []
                for step in range(stage_steps):
                    error = float(env.data.qpos[1] - origin[1]) * (-1 if speed < 0 else 1)
                    targets = skill_targets(env, "move", d_hat=[1., 0.], speed=speed,
                                            cross_track_error=error)
                    _, _, terminated, truncated, _ = env.step(targets)
                    if terminated or truncated:
                        raise RuntimeError("Stop/reverse recording terminated early")
                    velocity = env.data.qvel[:2].copy() * 100
                    velocities.append(velocity)
                    elapsed = (phase * stage_steps + step + 1) * dt
                    xy = (env.data.qpos[:2] - origin) * 100
                    rows.append([elapsed, phase + 1, *xy, *velocity])
                    if step % stride:
                        continue
                    camera.lookat[:] = env.data.qpos[:3]
                    renderer.update_scene(env.data, camera=camera)
                    canvas = Image.new("RGB", (800, 608), "#142032")
                    canvas.paste(Image.fromarray(renderer.render()), (0, 160))
                    draw = ImageDraw.Draw(canvas)
                    for k, (name, _) in enumerate(sequence):
                        x = 12 + k * 197
                        draw.rounded_rectangle((x, 12, x + 185, 53), radius=7,
                                               fill="#f2b544" if k == phase else "#30425b")
                        draw.text((x + 13, 17), name, font=large,
                                  fill="#142032" if k == phase else "white")
                    vx = float(np.mean(velocities[-20:], axis=0)[0])
                    draw.text((18, 65), f"Velocity along x: {vx:+.0f} cm/s", font=large, fill="#8ee5db")
                    draw.text((18, 108), "+ forward (right) / - reverse (left)", font=font, fill="white")
                    draw.text((602, 70), f"{elapsed:4.1f} / 16 s", font=font, fill="white")
                    draw.text((602, 111), "Real time", font=font, fill="white")
                    writer.append_data(np.asarray(canvas))
                    if phase == 2 and step == 300:
                        canvas.save(assets / "move-stop-reverse-preview.png")
                summaries.append({"phase": label.lower(),
                                  "requested_speed_cm_s": speed * 100,
                                  "displacement_cm": ((env.data.qpos[:2] - start) * 100).tolist(),
                                  "mean_last_second_vx_cm_s": float(np.mean(velocities[-round(1 / dt):], axis=0)[0]),
                                  "end_horizontal_speed_cm_s": float(np.linalg.norm(velocities[-1]))})
        np.savetxt(assets / "move-stop-reverse.csv", rows, delimiter=",", comments="",
                   header="time_s,phase,x_cm,y_cm,vx_cm_s,vy_cm_s")
        report = {"duration_s": 16, "phase_duration_s": 4, "reset_between_phases": False,
                  "seed": 42, "phases": summaries, "mujoco_version": mujoco.__version__}
        (assets / "move-stop-reverse-results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
