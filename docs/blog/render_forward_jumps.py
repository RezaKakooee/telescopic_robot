"""Record standing and running forward jumps at 2x slow motion.

PYTHONPATH=. MUJOCO_GL=egl python docs/blog/render_forward_jumps.py
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
from skills.runner import skill_targets


def record(moving):
    skill = "jump_forward_while_moving" if moving else "jump_forward_while_stopped"
    stem = "jump-forward-moving" if moving else "jump-forward-standing"
    assets = Path(__file__).resolve().parent / "assets"
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range, cfg.scenario.goal.y_range = [0., 0.], [-400., -400.]
    env = MujocoRadialSphereEnv(cfg, scenario=generate_scenario("goal", cfg, seed=42),
                               randomize=False, max_steps=10000)
    env.reset(seed=42)
    hold_targets = None
    resume_y = None

    def advance(phase, name=skill):
        if not moving and phase in ("stand", "stop"):
            targets = hold_targets if hold_targets is not None else execute_skill("stop", env.data.qpos[3:7].copy(), env.dirs_body,
                                    env.max_extend, lin_vel=env.data.qvel[:2].copy(),
                                    stop_distance=.15, stance_height=.025)
        elif moving and phase in ("move", "resume"):
            targets = skill_targets(env, "move", d_hat=[1., 0.], speed=1.2,
                                    cross_track_error=float(env.data.qpos[1] -
                                                            (resume_y if phase == "resume" else origin[1])))
        elif not moving and phase == "landing":
            # Prepare touchdown without the forward-jump skill's rollout push.
            targets = execute_skill("jump_up", env.data.qpos[3:7].copy(), env.dirs_body,
                                    env.max_extend, phase="landing")
        else:
            kwargs = {} if name == "jump_up" else {"d_hat": np.array([1., 0.])}
            targets = execute_skill(name, env.data.qpos[3:7].copy(), env.dirs_body,
                                    env.max_extend, phase=phase, **kwargs)
        _, _, ended, truncated, _ = env.step(targets)
        if ended or truncated:
            raise RuntimeError("Jump recording terminated early")
        return targets

    for _ in range(200):
        resting_targets = advance("stand", "jump_up")
    if not moving:
        hold_targets = np.zeros(env.n_bars, dtype=np.float32)
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
    phases = (["move", "dip", "launch", "airborne", "landing", "resume"] if moving else
              ["stand", "crouch", "takeoff", "airborne", "landing", "stop"])
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    dt = float(env.model.opt.timestep * env.action_repeat)
    video_stride = max(1, round(1 / (25 * 2 * dt)))
    rows, transitions = [], []
    phase, airborne_seen, landed = phases[0], False, False
    flight_start = touchdown = touchdown_time = None
    still_samples = []
    duration = 7 if moving else 5
    hold_time = None
    all_speeds = []
    peak, last_phase = origin[2], None
    try:
        with imageio.get_writer(assets / f"{stem}.mp4", fps=25) as writer:
            for step in range(round(duration / dt)):
                t = step * dt
                if moving and t < 2.0:
                    phase = "move"
                elif moving and t < 2.07:
                    phase = "dip"
                elif moving and t < 2.20:
                    phase = "launch"
                elif not moving and t < .5:
                    phase = "stand"
                elif not moving and t < .70:
                    phase = "crouch"
                    hold_targets = None
                elif not moving and t < .82:
                    phase = "takeoff"
                elif not moving and landed:
                    phase = "stop"
                    if hold_targets is None and np.linalg.norm(env.data.qvel[:2]) < .10:
                        # Hold actuator lengths, never freeze the physical state.
                        hold_targets = np.zeros(env.n_bars, dtype=np.float32)
                        hold_time = t
                elif moving and landed:
                    phase = "resume"
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
                if (phase in ("takeoff", "launch", "airborne") and not grounded
                        and env.data.qpos[2] > origin[2] + .10):
                    airborne_seen = True
                    if flight_start is None:
                        flight_start = float(env.data.qpos[0])
                if phase == "landing" and grounded:
                    landed = True
                    if touchdown is None:
                        touchdown = float(env.data.qpos[0])
                        touchdown_time = t + dt
                        resume_y = float(env.data.qpos[1])
                z, vz = float(env.data.qpos[2]), float(env.data.qvel[2])
                peak = max(peak, z)
                x = float(env.data.qpos[0] - origin[0]) * 100
                vx = float(env.data.qvel[0]) * 100
                all_speeds.append([t + dt, vx])
                rows.append([t + dt, phases.index(phase), z * 100, vz * 100, int(grounded), x, vx])
                if moving:
                    camera.lookat[0:2] = env.data.qpos[:2]
                else:
                    still_samples.append([t + dt, *env.data.qpos[:2],
                                          float(np.linalg.norm(env.data.qvel[:2]))])
                renderer.update_scene(env.data, camera=camera)
                canvas = Image.new("RGB", (800, 608), "#142032")
                canvas.paste(Image.fromarray(renderer.render()), (0, 160))
                draw = ImageDraw.Draw(canvas)
                for k, name in enumerate(phases):
                    box_width = 788 / len(phases)
                    left = 12 + k * box_width
                    draw.rounded_rectangle((left, 12, left + box_width - 12, 53), radius=7,
                                           fill="#f2b544" if name == phase else "#30425b")
                    still = (hold_targets is not None and np.linalg.norm(env.data.qvel[:3]) < .005)
                    display_name = "STILL" if name == "stop" and still else name.upper()
                    draw.text((left + 8, 18), display_name, font=font,
                              fill="#142032" if name == phase else "white")
                draw.text((18, 65), f"Height: {z * 100:.0f} cm  |  Forward: {x:.0f} cm", font=large, fill="white")
                draw.text((18, 108), f"vx: {vx:+.0f}   vz: {vz * 100:+.0f} cm/s", font=large, fill="#8ee5db")
                draw.text((560, 65), f"Simulation: {t + dt:.2f} s", font=font, fill="white")
                draw.text((560, 108), "2x slow motion", font=font, fill="white")
                if (step + 1) % video_stride == 0:
                    writer.append_data(np.asarray(canvas))
                if step == (225 if moving else 95):
                    canvas.save(assets / f"{stem}-preview.png")
        report = {"skill": skill, "seed": 42, "duration_s": duration, "video_slowdown": 2,
                  "starting_core_height_cm": float(origin[2] * 100),
                  "peak_core_height_cm": float(peak * 100),
                  "rise_cm": float((peak - origin[2]) * 100),
                  "airborne_verified": airborne_seen, "touchdown_verified": landed,
                  "forward_to_touchdown_cm": None if touchdown is None else (touchdown - float(origin[0])) * 100,
                  "final_vertical_speed_cm_s": float(env.data.qvel[2] * 100),
                  "final_horizontal_speed_cm_s": float(np.linalg.norm(env.data.qvel[:2]) * 100),
                  "phases": transitions}
        if not moving:
            samples = np.array(still_samples)
            final = samples[samples[:, 0] > duration - .5]
            report["post_touchdown_forward_travel_cm"] = float((env.data.qpos[0] - touchdown) * 100)
            report["final_half_second_max_speed_cm_s"] = float(final[:, 3].max() * 100)
            report["initial_stand_max_speed_cm_s"] = float(samples[samples[:, 0] <= .5, 3].max() * 100)
            report["final_half_second_displacement_cm"] = float(np.linalg.norm(final[-1, 1:3] - final[0, 1:3]) * 100)
            report["touchdown_time_s"] = touchdown_time
            report["hold_time_s"] = hold_time
            recorded = np.asarray(rows)
            report["final_half_second_max_vertical_speed_cm_s"] = float(abs(recorded[recorded[:, 0] > duration - .5, 3]).max())
        else:
            speeds = np.asarray(all_speeds)
            report["requested_before_and_after_cm_s"] = 120
            report["mean_before_jump_vx_cm_s"] = float(speeds[(speeds[:, 0] > 1.5) & (speeds[:, 0] <= 2), 1].mean())
            report["mean_final_second_vx_cm_s"] = float(speeds[speeds[:, 0] > duration - 1, 1].mean())
        np.savetxt(assets / f"{stem}.csv", rows, delimiter=",", comments="",
                   header="time_s,phase,core_height_cm,vertical_speed_cm_s,ground_contact,x_cm,vx_cm_s")
        (assets / f"{stem}-results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not airborne_seen or not landed:
            raise RuntimeError("The jump must include verified flight and touchdown")
        if not moving and (report["initial_stand_max_speed_cm_s"] > 3
                           or report["final_half_second_max_speed_cm_s"] > 3
                           or report["final_half_second_displacement_cm"] > 1
                           or report["final_half_second_max_vertical_speed_cm_s"] > .5):
            raise RuntimeError("The standing jump must begin and finish at rest")
        if moving and abs(report["mean_final_second_vx_cm_s"] - 120) > 10:
            raise RuntimeError("The running jump must resume the original forward speed")
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    record(False)
    record(True)
