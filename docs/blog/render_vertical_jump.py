"""Record a vertical jump at 2x slow motion, using explicit jump_up phases.

PYTHONPATH=. MUJOCO_GL=egl python docs/blog/render_vertical_jump.py
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


def main():
    assets = assets_dir()
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range, cfg.scenario.goal.y_range = [0., 0.], [-400., -400.]
    env = MujocoRadialSphereEnv(cfg, scenario=generate_scenario("goal", cfg, seed=42),
                               randomize=False, max_steps=10000)
    env.reset(seed=42)

    def advance(phase):
        targets = execute_skill("jump_up", env.data.qpos[3:7].copy(), env.dirs_body,
                                env.max_extend, phase=phase)
        _, _, ended, truncated, _ = env.step(targets)
        if ended or truncated:
            raise RuntimeError("Jump recording terminated early")

    for _ in range(100):
        advance("stand")
    origin = env.data.qpos[:3].copy()
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = 800, 448
    renderer = mujoco.Renderer(env.model, height=448, width=800)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90, -10, 2.3
    camera.lookat[:] = [origin[0], origin[1], .65]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    phases = ["crouch", "takeoff", "airborne", "landing"]
    dt = float(env.model.opt.timestep * env.action_repeat)
    video_stride = max(1, round(1 / (25 * 2 * dt)))
    rows, transitions = [], []
    phase, airborne_seen, landed = "crouch", False, False
    peak, last_phase = origin[2], None
    try:
        with imageio.get_writer(assets / "jump-up.mp4", fps=25) as writer:
            for step in range(round(3 / dt)):
                t = step * dt
                if t < .20:
                    phase = "crouch"
                elif t < .32:
                    phase = "takeoff"
                elif phase != "landing":
                    phase = "landing" if (airborne_seen and env.data.qvel[2] < 0
                                            and env.data.qpos[2] < .32) else "airborne"
                if phase != last_phase:
                    transitions.append({"phase": phase, "time_s": t})
                    last_phase = phase
                advance(phase)
                grounded = any(env.floor_geom_id in (c.geom1, c.geom2)
                               and (c.geom1 in env.robot_geom_ids or c.geom2 in env.robot_geom_ids)
                               and c.dist <= 0 for c in env.data.contact)
                if not grounded and env.data.qpos[2] > origin[2] + .10:
                    airborne_seen = True
                if phase == "landing" and grounded:
                    landed = True
                z, vz = float(env.data.qpos[2]), float(env.data.qvel[2])
                peak = max(peak, z)
                rows.append([t + dt, phases.index(phase), z * 100, vz * 100, int(grounded)])
                renderer.update_scene(env.data, camera=camera)
                canvas = Image.new("RGB", (800, 608), "#142032")
                canvas.paste(Image.fromarray(renderer.render()), (0, 160))
                draw = ImageDraw.Draw(canvas)
                for k, name in enumerate(phases):
                    x = 12 + k * 197
                    draw.rounded_rectangle((x, 12, x + 185, 53), radius=7,
                                           fill="#f2b544" if name == phase else "#30425b")
                    draw.text((x + 12, 18), name.upper(), font=font,
                              fill="#142032" if name == phase else "white")
                draw.text((18, 65), f"Core height: {z * 100:.0f} cm", font=large, fill="white")
                draw.text((18, 108), f"Vertical velocity: {vz * 100:+.0f} cm/s", font=large, fill="#8ee5db")
                draw.text((560, 65), f"Simulation: {t + dt:.2f} s", font=font, fill="white")
                draw.text((560, 108), "2x slow motion", font=font, fill="white")
                if (step + 1) % video_stride == 0:
                    writer.append_data(np.asarray(canvas))
                if step == 45:
                    canvas.save(assets / "jump-up-preview.png")
        report = {"seed": 42, "duration_s": 3, "video_slowdown": 2,
                  "starting_core_height_cm": float(origin[2] * 100),
                  "peak_core_height_cm": float(peak * 100),
                  "rise_cm": float((peak - origin[2]) * 100),
                  "airborne_verified": airborne_seen, "touchdown_verified": landed,
                  "final_vertical_speed_cm_s": float(env.data.qvel[2] * 100),
                  "phases": transitions}
        np.savetxt(assets / "jump-up.csv", rows, delimiter=",", comments="",
                   header="time_s,phase,core_height_cm,vertical_speed_cm_s,ground_contact")
        (assets / "jump-up-results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not airborne_seen or not landed:
            raise RuntimeError("The jump must include verified flight and touchdown")
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
