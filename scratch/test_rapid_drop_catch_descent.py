import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from pathlib import Path
import imageio
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.run_id import build_run_id
from radial_sphere.snapshot import make_run_dir
from skills.overlay import annotate

def main():
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = generate_scenario("chimney", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    start_z = 3.50
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = start_z
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print("=== Testing Rapid Zero-Slip Drop-and-Catch Descent ===")

    run_dir = make_run_dir(build_run_id("fast_descent", "rapid_drop_catch"))
    out_video = Path(run_dir) / "renders" / "rapid_drop_catch_descent.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    # Find lateral clamp rods (both walls)
    horiz_rods = []
    for i, d in enumerate(env.dirs_body):
        if abs(d[0]) < 0.35 and abs(d[1]) > 0.70 and abs(d[2]) < 0.40:
            horiz_rods.append(i)

    # Initial clamp hold
    for _ in range(25):
        targets = np.full(60, 0.01, dtype=np.float32)
        targets[horiz_rods] = 0.12
        env.step(targets)

    step = 0
    state = "drop"
    timer = 0
    drop_steps = 22   # ~88ms free drop in air -> drops ~25-35cm per pulse safely
    catch_steps = 20  # ~80ms damped clamp to smoothly freeze momentum

    cycle_count = 0
    landed = False

    while step < 1500:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        z = pos[2]
        vz = vel[2]

        targets = np.full(60, 0.01, dtype=np.float32)

        # Ground proximity landing gear
        if z < 0.65:
            targets[env.dirs_body[:, 2] < -0.30] = 0.05

        if z <= 0.23:
            print(f"Touchdown on floor at step {step}! Final Z: {z:.3f}m | vz: {vz:+.2f}m/s")
            landed = True
            for _ in range(35):
                targets = np.zeros(60, dtype=np.float32)
                targets[env.dirs_body[:, 2] < -0.30] = 0.045
                env.step(targets)
                if step % 4 == 0:
                    render_frame(env, writer, "Landed & Standing", z, vz, "Touchdown Stance")
                step += 1
            break

        timer += 1

        if state == "drop":
            # Completely tucked: in open air, ZERO wall contact / ZERO slip wear!
            targets[horiz_rods] = 0.01
            phase_label = "Free-Air Pulse (0 Slip)"
            if timer >= drop_steps:
                state = "catch"
                timer = 0
                cycle_count += 1

        elif state == "catch":
            # High-force damped clamp: locks onto both walls, braking to 0 velocity
            targets[horiz_rods] = 0.11
            phase_label = "Static Clamp Lock"
            if timer >= catch_steps and abs(vz) < 0.25:
                state = "drop"
                timer = 0

        # Planar lock
        env.data.qpos[0] = 0.0
        env.data.qvel[0] = 0.0
        env.data.qpos[3:7] = [1, 0, 0, 0]
        env.data.qvel[3:6] = [0, 0, 0]

        env.step(targets)

        if step % 4 == 0:
            render_frame(env, writer, phase_label, z, vz, f"Cycle #{cycle_count}")

        if step % 25 == 0:
            print(f"Step {step:4d} [{phase_label:24s}]: z={z:.3f}m | vz={vz:+.2f}m/s | cycle={cycle_count}")

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished! Final Z: {z:.3f}m | Total steps: {step} (~{step*0.004:.2f}s)")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/rapid_drop_catch_descent.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")


def render_frame(env, writer, action_name, z, vz, detail_str):
    if env.renderer is None:
        env.render(camera_name="fixed_angle_close_3d")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.core_body_id
    cam.distance = 1.60
    cam.elevation = -8.0
    cam.azimuth = 180.0
    env.renderer.update_scene(env.data, camera=cam)
    frame = env.renderer.render()

    ann_frame = annotate(
        frame,
        "Rapid Zero-Slip Drop-and-Catch Descent",
        [
            f"Phase: {action_name}",
            f"Height Z: {z:.3f} m",
            f"Descent Velocity: {vz:+.2f} m/s",
            f"Status: {detail_str}",
        ],
        margin=14,
    )
    writer.append_data(ann_frame)


if __name__ == "__main__":
    main()
