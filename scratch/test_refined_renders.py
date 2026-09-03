"""Fine-tuned showcase scenes for the 4 skills:
1. move_right & move_left (facing front camera with clear rightward and leftward trajectory across screen)
2. push_against_wall (clear open-field 3D view with wall behind robot, showing dynamic push-off)
3. jump_forward_moving (hurdle box at x=2.5m, sprint -> dip -> rocket launch -> hurdle clear -> landing roll)
4. fall_down (elevated platform ending at x=2.4m, height 0.80m, creep -> 0.80m freefall -> suspension cushion -> floor rollout)
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import imageio
import mujoco
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario, generate_scenario
from skills import execute_skill
from skills.overlay import annotate
from skills.locomotion import move, move_right, move_left, stop, go_fast
from skills.jumping import jump_forward_while_moving
from skills.falling import fall_down

FORWARD = np.array([1.0, 0.0], dtype=np.float32)


def custom_render(env, distance=2.2, elevation=-20.0, azimuth=90.0):
    if env.renderer is None:
        env.renderer = mujoco.Renderer(env.model, height=env.render_size[0], width=env.render_size[1])
    env._update_dynamic_colors()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.core_body_id
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth
    env.renderer.update_scene(env.data, camera=cam)
    return env.renderer.render()


def render_hurdle_jump(out_path: Path):
    """Render a running hurdle leap over a real physical hurdle box."""
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = True

    hurdle_x = 2.4
    sc = Scenario(
        kind="goal",
        name="hurdle_leap_demo",
        spawn_xy=np.array([0.0, 0.0], dtype=np.float32),
        goal=np.array([8.0, 0.0], dtype=np.float32),
        path_pts=np.array([[0.0, 0.0], [8.0, 0.0]], dtype=np.float32),
        markers=np.empty((0, 2), dtype=np.float32),
        path_length=8.0,
        steps=np.array([[hurdle_x, 0.0, 0.25, 2.0, 0.25]], dtype=np.float32), # 25cm hurdle at x=2.4
    )

    env = MujocoRadialSphereEnv(cfg, scenario=sc, randomize=False, max_steps=500)
    env.reset(seed=42)
    w = imageio.get_writer(str(out_path), fps=25, codec="libx264", quality=9)

    state = "sprint"
    substep = 0

    for step in range(220):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        dist_to_hurdle = hurdle_x - pos[0]

        if state == "sprint":
            phase_desc = "Sprint Run-Up (vx ~ 2.4 m/s)"
            targets = jump_forward_while_moving(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, phase="sprint")
            if dist_to_hurdle <= 0.70:
                state = "dip"
                substep = 0
        elif state == "dip":
            phase_desc = "Kinematic Pre-Leap Dip"
            substep += 1
            targets = jump_forward_while_moving(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, phase="dip")
            if substep >= 5:
                state = "launch"
                substep = 0
        elif state == "launch":
            phase_desc = "Explosive Rocket Launch"
            substep += 1
            targets = jump_forward_while_moving(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, phase="launch")
            if substep >= 10:
                state = "airborne"
                substep = 0
        elif state == "airborne":
            phase_desc = "Mid-Air Hurdle Clearance"
            substep += 1
            targets = jump_forward_while_moving(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, phase="airborne")
            if pos[0] > hurdle_x + 0.30 and vel[2] < 0.10:
                state = "landing"
                substep = 0
        elif state == "landing":
            phase_desc = "Compliant Touchdown Suspension"
            substep += 1
            targets = jump_forward_while_moving(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, phase="landing")
            if substep >= 20:
                state = "runout"
        else: # runout
            phase_desc = "Runout & In-Stride Sprint"
            targets = move(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, speed=1.4)

        env.step(targets)

        if step % 2 == 0:
            frame = custom_render(env, distance=2.8, elevation=-16.0, azimuth=90.0)
            lines = [
                f"Phase: {phase_desc}",
                f"Dist to Hurdle: {dist_to_hurdle:5.2f} m",
                f"Position: x={pos[0]:5.2f}  z={pos[2]:5.2f} m",
                f"Velocity: vx={vel[0]:+5.2f}  vz={vel[2]:+5.2f} m/s",
            ]
            annotated_frame = np.array(annotate(frame, "Skill 11: Jump Forward (Running Hurdle Leap)", lines, margin=14), copy=True)
            w.append_data(annotated_frame)
            if state == "airborne" and substep == 6:
                imageio.imwrite("scratch/test_renders/11_jump_apex.png", annotated_frame)
            if state == "landing" and substep == 5:
                imageio.imwrite("scratch/test_renders/11_jump_landing.png", annotated_frame)

    w.close()
    env.close()
    print("  --> Rendered hurdle_jump")


def render_platform_drop(out_path: Path):
    """Render a dramatic 0.80m high platform drop with full free-fall & compliant landing."""
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = True

    drop_h = 0.80
    edge_x = 2.40
    sc = Scenario(
        kind="goal",
        name="platform_drop_demo",
        spawn_xy=np.array([0.0, 0.0], dtype=np.float32),
        goal=np.array([8.0, 0.0], dtype=np.float32),
        path_pts=np.array([[0.0, 0.0], [8.0, 0.0]], dtype=np.float32),
        markers=np.empty((0, 2), dtype=np.float32),
        path_length=8.0,
        ramps=np.array([[1.20, 0.0, 2.40, 3.0, drop_h, 0.0, 0.0]], dtype=np.float32),
    )

    env = MujocoRadialSphereEnv(cfg, scenario=sc, randomize=False, max_steps=500)
    env.reset(seed=42)

    # Spawn ball on top of the 0.80m platform close to edge at x = 1.7m
    env.data.qpos[0] = 1.70
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = drop_h + 0.19
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    # Settle stance
    for _ in range(30):
        env.step(stop(env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, stance_height=0.045))

    w = imageio.get_writer(str(out_path), fps=25, codec="libx264", quality=9)
    phase = "edge"
    substep = 0

    for step in range(200):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        z = float(pos[2])

        if phase == "edge":
            phase_desc = "Roll to Platform Edge"
            targets = fall_down(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, phase="edge", edge_speed=0.90)
            if pos[0] >= edge_x + 0.10 or z < drop_h + 0.10:
                phase = "freefall"
                substep = 0
        elif phase == "freefall":
            phase_desc = f"{drop_h:.2f}m Airborne Free Fall"
            substep += 1
            targets = fall_down(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, phase="freefall", drop_height=drop_h, gear=0.6)
            if z < 0.42:
                phase = "absorb"
                substep = 0
        elif phase == "absorb":
            phase_desc = "Pneumatic Suspension Cushion"
            substep += 1
            targets = fall_down(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, phase="absorb", drop_height=drop_h)
            if z <= 0.23 or substep >= 16:
                phase = "settle"
                substep = 0
        elif phase == "settle":
            phase_desc = "Stance Recovery"
            substep += 1
            targets = stop(quat, env.dirs_body, env.max_extend, lin_vel=vel, stance_height=0.045)
            if substep >= 25:
                phase = "drive_away"
        else: # drive_away
            phase_desc = "Roll Out Across Lower Floor"
            targets = move(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, speed=1.3)

        env.step(targets)

        if step % 2 == 0:
            frame = custom_render(env, distance=3.2, elevation=-18.0, azimuth=90.0)
            lines = [
                f"Phase: {phase_desc}",
                f"Platform Drop: -{drop_h:.2f} m (to z=0.0m)",
                f"Position: x={pos[0]:5.2f}  z={pos[2]:5.2f} m",
                f"Velocity: vx={vel[0]:+5.2f}  vz={vel[2]:+5.2f} m/s",
            ]
            annotated_frame = np.array(annotate(frame, "Skill 12: Fall Down (Controlled High Drop)", lines, margin=14), copy=True)
            w.append_data(annotated_frame)
            if phase == "freefall" and substep == 10:
                imageio.imwrite("scratch/test_renders/12_drop_freefall.png", annotated_frame)
            if phase == "absorb" and substep == 5:
                imageio.imwrite("scratch/test_renders/12_drop_absorb.png", annotated_frame)

    w.close()
    env.close()
    print("  --> Rendered platform_drop")


if __name__ == "__main__":
    out_dir = Path("scratch/test_renders")
    out_dir.mkdir(parents=True, exist_ok=True)
    render_hurdle_jump(out_dir / "11_jump_forward_moving.mp4")
    render_platform_drop(out_dir / "12_fall_down.mp4")
