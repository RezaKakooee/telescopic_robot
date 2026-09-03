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
from omegaconf import OmegaConf
from radial_sphere.geometry import quat_to_rotmat

def main():
    cyl_rad = 0.35
    cyl_height = 4.0

    cfg = load_config("configs/rl/chimney.yaml")
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "scenario") or cfg.scenario is None:
        cfg.scenario = {}
    cfg.scenario.cylinder_radius = cyl_rad
    cfg.scenario.cylinder_height = cyl_height

    scenario = generate_scenario("vertical_cylinder", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    # Spawn resting at base
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Perfect 3D Helical Spiral Climbing in Vertical Cylinder ===")

    run_dir = make_run_dir(build_run_id("cylinder_spiral", "perfect_helix"))
    out_video = Path(run_dir) / "renders" / "perfect_vertical_cylinder_spiral.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    # Settle
    for _ in range(15):
        t = np.zeros(60, dtype=np.float32)
        t[env.dirs_body[:, 2] < -0.30] = 0.045
        env.step(t)

    step = 0
    state = "launch"
    timer = 0
    max_z = 0.22
    jump_count = 0

    while step < 1200:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        z = float(pos[2])
        max_z = max(max_z, z)

        r = float(np.hypot(pos[0], pos[1]))
        theta = float(np.arctan2(pos[1], pos[0]))

        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        timer += 1
        targets = np.full(60, 0.010, dtype=np.float32)

        if state == "launch":
            # Explosive upward launch off ground
            targets[dirs_world[:, 2] < -0.20] = env.max_extend
            phase_label = "Ground Takeoff"
            if timer >= 12:
                state = "fly"
                timer = 0
                jump_count += 1

        elif state == "fly":
            # Tucked in free air along 3D spiral flight path
            targets[:] = 0.010
            phase_label = f"Spiral Flight #{jump_count}"
            # At apex (vz <= 0.0) or after flight duration
            if timer >= 16 and vel[2] < 0.35:
                state = "apex_lock"
                timer = 0

        elif state == "apex_lock":
            # Clamp against curved cylinder walls to arrest fall
            lat_proj = np.hypot(dirs_world[:, 0], dirs_world[:, 1])
            targets[(lat_proj > 0.60) & (abs(dirs_world[:, 2]) < 0.45)] = 0.12
            phase_label = f"Wall Lock #{jump_count}"
            if timer >= 10:
                state = "spiral_boost"
                timer = 0

        elif state == "spiral_boost":
            # Explosive Helical Boost: Push into wall (+r) + tangential (-theta) + downward (-z)
            # Produces upward-forward spiral leap!
            r_hat = np.array([np.cos(theta), np.sin(theta), 0.0]) if r > 1e-4 else np.array([1, 0, 0])
            theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])

            # Unit vector pointing into the wall contact zone
            push_unit = 0.70 * r_hat - 0.40 * theta_hat - 0.58 * np.array([0, 0, 1])
            push_unit = push_unit / np.linalg.norm(push_unit)

            p = dirs_world @ push_unit
            # Only push rods extend, all other rods stay tucked to prevent wall drag
            targets[p > 0.35] = env.max_extend
            phase_label = f"Helical Boost (گشتاور) #{jump_count}"
            if timer >= 10:
                state = "fly"
                timer = 0
                jump_count += 1

        env.step(targets)

        if step % 4 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")

            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.core_body_id
            cam.distance = 2.40
            cam.elevation = -22.0
            cam.azimuth = float(np.degrees(theta)) + 120.0
            env.renderer.update_scene(env.data, camera=cam)
            frame = env.renderer.render()

            ann_frame = annotate(
                frame,
                "Vertical Cylinder Spiral Vortex (گشتاور)",
                [
                    f"Phase: {phase_label}",
                    f"Altitude Z: {z:.3f} m (Peak: {max_z:.2f}m)",
                    f"Ascent Speed vz: {vel[2]:+.2f} m/s",
                    f"Helical Strides: {jump_count}",
                ],
                margin=14,
            )
            writer.append_data(ann_frame)

        if step % 25 == 0:
            print(f"Step {step:4d} [{phase_label:26s}]: z={z:.3f}m | vz={vel[2]:+.2f}m/s | peak={max_z:.3f}m")

        if z >= 3.50:
            print(f"\nTarget height reached! Z = {z:.3f}m at step {step}")
            for _ in range(40):
                lat_proj = np.hypot(dirs_world[:, 0], dirs_world[:, 1])
                targets_top = np.zeros(60, dtype=np.float32)
                targets_top[lat_proj > 0.60] = 0.12
                env.step(targets_top)
                if step % 4 == 0:
                    render_frame(env, writer, "Apex Lock at Top Rim", z, vel[2], jump_count)
                step += 1
            break

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished Helical Climb! Start Z: 0.22m -> Peak Z: {max_z:.3f}m in {jump_count} spiral boosts!")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/perfect_vertical_cylinder_spiral.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

def render_frame(env, writer, label, z, vz, count):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.core_body_id
    cam.distance = 2.40
    cam.elevation = -22.0
    cam.azimuth = 135.0
    env.renderer.update_scene(env.data, camera=cam)
    frame = env.renderer.render()
    ann_frame = annotate(
        frame,
        "Vertical Cylinder Spiral Vortex (گشتاور)",
        [
            f"Phase: {label}",
            f"Altitude Z: {z:.3f} m",
            f"Ascent Speed: {vz:+.2f} m/s",
            f"Helical Strides: {count}",
        ],
        margin=14,
    )
    writer.append_data(ann_frame)

if __name__ == "__main__":
    main()
