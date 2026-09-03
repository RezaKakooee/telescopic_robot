"""Master Showcase Suite: Render and organize all 22 skill videos into a dedicated experiment folder.

Generates:
  01_move_forward.mp4
  02_move_right.mp4
  03_move_left.mp4
  04_stop_and_brake.mp4
  05_go_fast.mp4
  06_go_slow.mp4
  07_reverse.mp4
  08_push_wall.mp4
  09_jump_up.mp4
  10_jump_forward_stopped.mp4
  11_jump_forward_moving.mp4
  12_fall_down.mp4
  13_jump_to_pillars.mp4
  14_circle_orbit.mp4
  15_straddle_gap.mp4
  16_chimney_climb.mp4
  17_motordrome_wall_of_death.mp4
  18_wall_of_death_descent.mp4
  19_wall_run.mp4
  20_training_cones_slalom.mp4
  21_curved_training_cones_slalom.mp4
  22_stairs_climb_and_descent.mp4
  00_continuous_skill_course_parkour.mp4
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import shutil
import subprocess
import sys
import time
from pathlib import Path
import imageio
import mujoco
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario, generate_scenario, skill_course_platform
from skills import execute_skill
from skills.overlay import annotate
from skills.runner import skill_targets
from skills.locomotion import move, move_right, move_left, stop, go_fast
from skills.jumping import jump_forward_while_moving
from skills.falling import fall_down

FORWARD = np.array([1.0, 0.0], dtype=np.float32)


def custom_render(env, distance=2.2, elevation=-20.0, azimuth=90.0):
    """Render tracking camera at exact requested view angle without rotation jitter."""
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


def render_primitive(
    skill_name: str,
    title: str,
    out_path: Path,
    *,
    steps: int = 120,
    d_hat: np.ndarray = FORWARD,
    custom_fn = None,
    custom_render_fn = None,
    fps: int = 25,
) -> None:
    """Render a clean demonstration of a single primitive skill."""
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = True
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=steps + 100)
    env.reset(seed=42)

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=9)

    for step in range(steps):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        speed = float(np.linalg.norm(vel[:2]))

        if custom_fn is not None:
            targets = custom_fn(step, env)
        else:
            targets = skill_targets(env, skill_name, step, d_hat=d_hat)

        env.step(targets)

        if step % 2 == 0:
            if custom_render_fn is not None:
                frame = custom_render_fn(env)
            else:
                frame = env.render(camera_name="close")
            lines = [
                f"Skill: {skill_name}",
                f"Position: x={pos[0]:5.2f}  y={pos[1]:+5.2f}  z={pos[2]:5.2f}m",
                f"Speed: {speed:5.2f} m/s  (vx={vel[0]:+5.2f}, vy={vel[1]:+5.2f})",
                f"Step: {step:3d} / {steps:3d}",
            ]
            frame_annotated = np.array(annotate(frame, title, lines, margin=14), copy=True)
            writer.append_data(frame_annotated)

    writer.close()
    env.close()
    print(f"  --> Saved: {out_path.name}")


def main():
    timestamp = time.strftime("%Y%m%d_%H%M")
    exp_dir = Path(f"storage_local/{timestamp}__local__all_skills_showcase/renders")
    exp_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/all_skills_showcase")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"  RENDERING ALL 22 SKILLS SHOWCASE")
    print(f"  Experiment Directory: {exp_dir}")
    print(f"  Artifact Directory:   {artifact_dir}")
    print(f"=======================================================\n")

    # 1. move_forward
    render_primitive("move_forward", "Skill 01: Move Forward (Nominal Rolling Gait)", exp_dir / "01_move_forward.mp4", steps=100)

    # 2. move_right (along -y, rolls right across screen in front-facing camera)
    def render_front_cam(env):
        return custom_render(env, distance=1.8, elevation=-22.0, azimuth=0.0)
    render_primitive(
        "move_right",
        "Skill 02: Move Right (Lateral Strafe)",
        exp_dir / "02_move_right.mp4",
        steps=120,
        custom_fn=lambda step, env: move_right(env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, d_hat=FORWARD, speed=1.5),
        custom_render_fn=render_front_cam,
    )

    # 3. move_left (along +y, rolls left across screen in front-facing camera)
    render_primitive(
        "move_left",
        "Skill 03: Move Left (Lateral Strafe)",
        exp_dir / "03_move_left.mp4",
        steps=120,
        custom_fn=lambda step, env: move_left(env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, d_hat=FORWARD, speed=1.5),
        custom_render_fn=render_front_cam,
    )

    # 4. stop & brake
    def stop_brake_fn(step, env):
        quat = env.data.qpos[3:7].copy()
        vel = env.data.qvel[:3].copy()
        if step < 40:
            return execute_skill("go_fast", quat, env.dirs_body, env.max_extend, d_hat=FORWARD)
        else:
            return execute_skill("stop", quat, env.dirs_body, env.max_extend, lin_vel=vel, stance_height=0.045)
    render_primitive("stop", "Skill 04: Stop & Active Braking", exp_dir / "04_stop_and_brake.mp4", steps=110, custom_fn=stop_brake_fn)

    # 5. go_fast
    render_primitive("go_fast", "Skill 05: Go Fast (High-Speed Sprint)", exp_dir / "05_go_fast.mp4", steps=110)

    # 6. go_slow
    render_primitive("go_slow", "Skill 06: Go Slow (Precision Crawl)", exp_dir / "06_go_slow.mp4", steps=110)

    # 7. reverse
    render_primitive("reverse", "Skill 07: Reverse (Backward Locomotion)", exp_dir / "07_reverse.mp4", steps=110, d_hat=FORWARD)

    # 8. push_against_wall (full 5-second dynamic push showcase)
    out08 = exp_dir / "08_push_wall.mp4"
    cfg_w = load_config("configs/rl/config.yaml")
    cfg_w.camera.enabled = True
    sc_w = Scenario(
        kind="goal",
        name="wall_push_demo",
        spawn_xy=np.array([0.0, -0.2], dtype=np.float32),
        goal=np.array([10.0, 0.0], dtype=np.float32),
        path_pts=np.array([[0.0, -0.2], [10.0, -0.2]], dtype=np.float32),
        markers=np.empty((0, 2), dtype=np.float32),
        path_length=10.0,
        walls=np.array([[-1.0, -0.6, 12.0, -0.6]], dtype=np.float32),
    )
    env_w = MujocoRadialSphereEnv(cfg_w, scenario=sc_w, randomize=False, max_steps=500)
    env_w.reset(seed=42)
    w_w = imageio.get_writer(str(out08), fps=25, codec="libx264", quality=9)
    wall_normal = np.array([0.0, 1.0], dtype=np.float32)
    for step in range(250):
        quat = env_w.data.qpos[3:7].copy()
        pos = env_w.data.qpos[:3].copy()
        vel = env_w.data.qvel[:3].copy()
        dist_to_wall = float(pos[1] - (-0.60) - 0.173)
        if step < 40:
            phase_desc = "Approach Wall Contact"
            targets = move(quat, env_w.dirs_body, env_w.max_extend, d_hat=np.array([1.0, -0.35]), speed=1.1)
        elif step < 130:
            phase_desc = "Active Wall Thrust & Lateral Shove"
            targets = execute_skill("push_against_wall", quat, env_w.dirs_body, env_w.max_extend, wall_normal=wall_normal)
        elif step < 170:
            phase_desc = "Steer Back Toward Wall"
            targets = move(quat, env_w.dirs_body, env_w.max_extend, d_hat=np.array([1.0, -0.30]), speed=1.1)
        else:
            phase_desc = "Second Lateral Push Off"
            targets = execute_skill("push_against_wall", quat, env_w.dirs_body, env_w.max_extend, wall_normal=wall_normal)
        env_w.step(targets)
        if step % 2 == 0:
            frame = custom_render(env_w, distance=1.8, elevation=-20.0, azimuth=-50.0)
            lines = [
                f"Action: {phase_desc}",
                f"Wall Distance: {dist_to_wall*100:5.1f} cm",
                f"Position: x={pos[0]:5.2f}  y={pos[1]:+5.2f}  z={pos[2]:5.2f}m",
                f"Lateral Thrust: vy={vel[1]:+5.2f} m/s",
            ]
            w_w.append_data(np.array(annotate(frame, "Skill 08: Push Against Wall (Active Shove)", lines, margin=14), copy=True))
    w_w.close()
    env_w.close()
    print(f"  --> Saved: {out08.name}")

    # 9. jump_up
    def jump_up_fn(step, env):
        quat = env.data.qpos[3:7].copy()
        if step < 20: phase = "crouch"
        elif step < 32: phase = "takeoff"
        elif env.data.qpos[2] > 0.28: phase = "airborne"
        else: phase = "landing"
        return execute_skill("jump_up", quat, env.dirs_body, env.max_extend, phase=phase)
    render_primitive("jump_up", "Skill 09: Jump Up (Stationary Vertical Leap)", exp_dir / "09_jump_up.mp4", steps=80, custom_fn=jump_up_fn)

    # 10. jump_forward_while_stopped
    def jump_fwd_stop_fn(step, env):
        quat = env.data.qpos[3:7].copy()
        if step < 20: phase = "crouch"
        elif step < 32: phase = "takeoff"
        elif env.data.qpos[2] > 0.28: phase = "airborne"
        else: phase = "landing"
        return execute_skill("jump_forward_while_stopped", quat, env.dirs_body, env.max_extend, d_hat=FORWARD, phase=phase)
    render_primitive("jump_forward_while_stopped", "Skill 10: Jump Forward (Standing Directional)", exp_dir / "10_jump_forward_stopped.mp4", steps=90, custom_fn=jump_fwd_stop_fn)

    # 11. jump_forward_while_moving (running hurdle leap over actual 0.25m obstacle box)
    out11 = exp_dir / "11_jump_forward_moving.mp4"
    cfg_j = load_config("configs/rl/config.yaml")
    cfg_j.camera.enabled = True
    hurdle_x = 2.40
    sc_j = Scenario(
        kind="goal",
        name="hurdle_leap_demo",
        spawn_xy=np.array([0.0, 0.0], dtype=np.float32),
        goal=np.array([8.0, 0.0], dtype=np.float32),
        path_pts=np.array([[0.0, 0.0], [8.0, 0.0]], dtype=np.float32),
        markers=np.empty((0, 2), dtype=np.float32),
        path_length=8.0,
        steps=np.array([[hurdle_x, 0.0, 0.25, 2.0, 0.25]], dtype=np.float32),
    )
    env_j = MujocoRadialSphereEnv(cfg_j, scenario=sc_j, randomize=False, max_steps=500)
    env_j.reset(seed=42)
    w_j = imageio.get_writer(str(out11), fps=25, codec="libx264", quality=9)
    state = "sprint"
    substep = 0
    for step in range(220):
        quat = env_j.data.qpos[3:7].copy()
        pos = env_j.data.qpos[:3].copy()
        vel = env_j.data.qvel[:3].copy()
        dist_to_hurdle = hurdle_x - pos[0]
        if state == "sprint":
            phase_desc = "Sprint Run-Up (vx ~ 2.4 m/s)"
            targets = jump_forward_while_moving(quat, env_j.dirs_body, env_j.max_extend, d_hat=FORWARD, phase="sprint")
            if dist_to_hurdle <= 0.70:
                state, substep = "dip", 0
        elif state == "dip":
            phase_desc = "Kinematic Pre-Leap Dip"
            substep += 1
            targets = jump_forward_while_moving(quat, env_j.dirs_body, env_j.max_extend, d_hat=FORWARD, phase="dip")
            if substep >= 5:
                state, substep = "launch", 0
        elif state == "launch":
            phase_desc = "Explosive Rocket Launch"
            substep += 1
            targets = jump_forward_while_moving(quat, env_j.dirs_body, env_j.max_extend, d_hat=FORWARD, phase="launch")
            if substep >= 10:
                state, substep = "airborne", 0
        elif state == "airborne":
            phase_desc = "Mid-Air Hurdle Clearance"
            substep += 1
            targets = jump_forward_while_moving(quat, env_j.dirs_body, env_j.max_extend, d_hat=FORWARD, phase="airborne")
            if pos[0] > hurdle_x + 0.30 and vel[2] < 0.10:
                state, substep = "landing", 0
        elif state == "landing":
            phase_desc = "Compliant Touchdown Suspension"
            substep += 1
            targets = jump_forward_while_moving(quat, env_j.dirs_body, env_j.max_extend, d_hat=FORWARD, phase="landing")
            if substep >= 20:
                state = "runout"
        else:
            phase_desc = "Runout & In-Stride Sprint"
            targets = move(quat, env_j.dirs_body, env_j.max_extend, d_hat=FORWARD, speed=1.4)
        env_j.step(targets)
        if step % 2 == 0:
            frame = custom_render(env_j, distance=2.8, elevation=-16.0, azimuth=90.0)
            lines = [
                f"Phase: {phase_desc}",
                f"Dist to Hurdle: {dist_to_hurdle:5.2f} m",
                f"Position: x={pos[0]:5.2f}  z={pos[2]:5.2f} m",
                f"Velocity: vx={vel[0]:+5.2f}  vz={vel[2]:+5.2f} m/s",
            ]
            w_j.append_data(np.array(annotate(frame, "Skill 11: Jump Forward (Running Hurdle Leap)", lines, margin=14), copy=True))
    w_j.close()
    env_j.close()
    print(f"  --> Saved: {out11.name}")

    # 12. fall_down (dramatic 0.80m platform drop with free-fall and suspension landing)
    out12 = exp_dir / "12_fall_down.mp4"
    cfg_f = load_config("configs/rl/config.yaml")
    cfg_f.camera.enabled = True
    drop_h = 0.80
    edge_x = 2.40
    sc_f = Scenario(
        kind="goal",
        name="platform_drop_demo",
        spawn_xy=np.array([0.0, 0.0], dtype=np.float32),
        goal=np.array([8.0, 0.0], dtype=np.float32),
        path_pts=np.array([[0.0, 0.0], [8.0, 0.0]], dtype=np.float32),
        markers=np.empty((0, 2), dtype=np.float32),
        path_length=8.0,
        ramps=np.array([[1.20, 0.0, 2.40, 3.0, drop_h, 0.0, 0.0]], dtype=np.float32),
    )
    env_f = MujocoRadialSphereEnv(cfg_f, scenario=sc_f, randomize=False, max_steps=500)
    env_f.reset(seed=42)
    env_f.data.qpos[0] = 1.70
    env_f.data.qpos[1] = 0.0
    env_f.data.qpos[2] = drop_h + 0.19
    env_f.data.qvel[:] = 0
    mujoco.mj_forward(env_f.model, env_f.data)
    for _ in range(30):
        env_f.step(stop(env_f.data.qpos[3:7].copy(), env_f.dirs_body, env_f.max_extend, stance_height=0.045))
    w_f = imageio.get_writer(str(out12), fps=25, codec="libx264", quality=9)
    phase = "edge"
    substep = 0
    for step in range(200):
        quat = env_f.data.qpos[3:7].copy()
        pos = env_f.data.qpos[:3].copy()
        vel = env_f.data.qvel[:3].copy()
        z = float(pos[2])
        if phase == "edge":
            phase_desc = "Roll to Platform Edge"
            targets = fall_down(quat, env_f.dirs_body, env_f.max_extend, d_hat=FORWARD, phase="edge", edge_speed=0.90)
            if pos[0] >= edge_x + 0.10 or z < drop_h + 0.10:
                phase, substep = "freefall", 0
        elif phase == "freefall":
            phase_desc = f"{drop_h:.2f}m Airborne Free Fall"
            substep += 1
            targets = fall_down(quat, env_f.dirs_body, env_f.max_extend, d_hat=FORWARD, phase="freefall", drop_height=drop_h, gear=0.6)
            if z < 0.42:
                phase, substep = "absorb", 0
        elif phase == "absorb":
            phase_desc = "Pneumatic Suspension Cushion"
            substep += 1
            targets = fall_down(quat, env_f.dirs_body, env_f.max_extend, d_hat=FORWARD, phase="absorb", drop_height=drop_h)
            if z <= 0.23 or substep >= 16:
                phase, substep = "settle", 0
        elif phase == "settle":
            phase_desc = "Stance Recovery"
            substep += 1
            targets = stop(quat, env_f.dirs_body, env_f.max_extend, lin_vel=vel, stance_height=0.045)
            if substep >= 25:
                phase = "drive_away"
        else:
            phase_desc = "Roll Out Across Lower Floor"
            targets = move(quat, env_f.dirs_body, env_f.max_extend, d_hat=FORWARD, speed=1.3)
        env_f.step(targets)
        if step % 2 == 0:
            frame = custom_render(env_f, distance=3.2, elevation=-18.0, azimuth=90.0)
            lines = [
                f"Phase: {phase_desc}",
                f"Platform Drop: -{drop_h:.2f} m (to z=0.0m)",
                f"Position: x={pos[0]:5.2f}  z={pos[2]:5.2f} m",
                f"Velocity: vx={vel[0]:+5.2f}  vz={vel[2]:+5.2f} m/s",
            ]
            w_f.append_data(np.array(annotate(frame, "Skill 12: Fall Down (Controlled High Drop)", lines, margin=14), copy=True))
    w_f.close()
    env_f.close()
    print(f"  --> Saved: {out12.name}")

    # 13. Pillars (jump_to)
    print("  Rendering Skill 13 (jump_to on pillars)...")
    out13 = exp_dir / "13_jump_to_pillars.mp4"
    subprocess.run([sys.executable, "scripts/skills/run_pillars.py", "--video", "--seed", "42"], check=True)
    p_vids = sorted(Path("storage_local").glob("**/renders/pillar_course.mp4"), key=os.path.getmtime)
    if p_vids:
        shutil.copy2(str(p_vids[-1]), str(out13))
    print(f"  --> Saved: {out13.name}")

    # 14. Circle Orbit
    print("  Rendering Skill 14 (circle orbit)...")
    out14 = exp_dir / "14_circle_orbit.mp4"
    subprocess.run([sys.executable, "scripts/skills/run_circle.py", "--video", "--laps", "1"], check=True)
    c_vids = sorted(Path("storage_local").glob("**/renders/circle_skill_composite.mp4"), key=os.path.getmtime)
    if c_vids:
        shutil.copy2(str(c_vids[-1]), str(out14))
    print(f"  --> Saved: {out14.name}")

    # 15. Straddle Gap
    print("  Rendering Skill 15 (straddle_gap)...")
    out15 = exp_dir / "15_straddle_gap.mp4"
    subprocess.run([sys.executable, "scripts/skills/run_gap.py", "--video"], check=True)
    g_vids = sorted(Path("storage_local").glob("**/renders/gap_straddle_composite.mp4"), key=os.path.getmtime)
    if g_vids:
        shutil.copy2(str(g_vids[-1]), str(out15))
    print(f"  --> Saved: {out15.name}")

    # 16. Chimney Climb
    print("  Rendering Skill 16 (chimney climb)...")
    out16 = exp_dir / "16_chimney_climb.mp4"
    subprocess.run([sys.executable, "scripts/skills/run_chimney.py", "--video"], check=True)
    ch_vids = sorted(Path("storage_local").glob("**/renders/chimney_climb.mp4"), key=os.path.getmtime)
    if ch_vids:
        shutil.copy2(str(ch_vids[-1]), str(out16))
    print(f"  --> Saved: {out16.name}")

    # 17 & 18. Motordrome Wall of Death (Ascent & Descent)
    print("  Rendering Skills 17 & 18 (wall_of_death ascent & descent)...")
    out17 = exp_dir / "17_motordrome_wall_of_death.mp4"
    out18 = exp_dir / "18_wall_of_death_descent.mp4"
    from scripts.skills import run_motordrome_wall_of_death
    res_wod = run_motordrome_wall_of_death.run(seconds=55.0, descend_after=35.0, seed=42, record_video=True)
    if "video" in res_wod and res_wod["video"] and Path(res_wod["video"]).exists():
        shutil.copy2(str(res_wod["video"]), str(out17))
        shutil.copy2(str(res_wod["video"]), str(out18))
    else:
        w_vids = sorted(Path("storage_local").glob("**/renders/motordrome_showcase.mp4"), key=os.path.getmtime)
        if w_vids:
            shutil.copy2(str(w_vids[-1]), str(out17))
            shutil.copy2(str(w_vids[-1]), str(out18))
    print(f"  --> Saved: {out17.name} & {out18.name}")

    # 19. Wall Run
    print("  Rendering Skill 19 (wall_run)...")
    out19 = exp_dir / "19_wall_run.mp4"
    from scripts.skills import run_wall_run
    res_wr = run_wall_run.run(mode="flat_multistep", seed=3, record_video=True)
    if "video" in res_wr and res_wr["video"] and Path(res_wr["video"]).exists():
        shutil.copy2(str(res_wr["video"]), str(out19))
    print(f"  --> Saved: {out19.name}")

    # 20. Training Cones Slalom
    print("  Rendering Skill 20 (training_cones slalom)...")
    out20 = exp_dir / "20_training_cones_slalom.mp4"
    from scripts.skills import run_training_cones
    res_tc = run_training_cones.run(seed=42, record_video=True)
    if "video" in res_tc and res_tc["video"] and Path(res_tc["video"]).exists():
        shutil.copy2(str(res_tc["video"]), str(out20))
    print(f"  --> Saved: {out20.name}")

    # 21. Curved Training Cones Slalom
    print("  Rendering Skill 21 (curved_training_cones slalom)...")
    out21 = exp_dir / "21_curved_training_cones_slalom.mp4"
    from scripts.skills import run_curved_training_cones
    res_ctc = run_curved_training_cones.run(seed=42, record_video=True)
    if "video" in res_ctc and res_ctc["video"] and Path(res_ctc["video"]).exists():
        shutil.copy2(str(res_ctc["video"]), str(out21))
    print(f"  --> Saved: {out21.name}")

    # 22. Stairs Climb and Descent
    print("  Rendering Skill 22 (stairs climb and descent)...")
    out22 = exp_dir / "22_stairs_climb_and_descent.mp4"
    from scripts.skills import run_stairs
    res_st = run_stairs.run(seed=42, record_video=True)
    if "video" in res_st and res_st["video"] and Path(res_st["video"]).exists():
        shutil.copy2(str(res_st["video"]), str(out22))
    print(f"  --> Saved: {out22.name}")

    # 00. Complete Continuous Skill Course Parkour
    print("  Rendering Complete Continuous Skill Course Parkour...")
    out00 = exp_dir / "00_continuous_skill_course_parkour.mp4"
    subprocess.run([sys.executable, "scripts/skills/run_course.py", "--video"], check=True)
    crs_vids = sorted(Path("storage_local").glob("**/renders/skill_course.mp4"), key=os.path.getmtime)
    if crs_vids:
        shutil.copy2(str(crs_vids[-1]), str(out00))
    print(f"  --> Saved: {out00.name}")

    # Copy all rendered clips into brain artifacts directory
    for f in sorted(exp_dir.glob("*.mp4")):
        shutil.copy2(str(f), str(artifact_dir / f.name))

    print(f"\n=======================================================")
    print(f"  ALL 22 SKILL VIDEOS SUCCESSFULLY RENDERED & SAVED!")
    print(f"  Experiment Directory: {exp_dir}")
    print(f"  Artifact Directory:   {artifact_dir}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
