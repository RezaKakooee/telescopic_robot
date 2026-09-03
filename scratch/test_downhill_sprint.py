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
    cyl_rad = 0.85
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

    # Spawn at top of ramp
    env.data.qpos[0] = -3.80
    env.data.qpos[1] = -cyl_rad
    env.data.qpos[2] = 1.35
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Downhill Power Sprint & High-Speed Cylinder Spiral ===")

    run_dir = make_run_dir(build_run_id("ramp_launch", "high_speed_spiral"))
    out_video = Path(run_dir) / "renders" / "high_speed_ramp_cylinder_spiral.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    # Downhill pitch angle on ramp (-16.7 deg)
    slope_pitch = np.radians(16.7)
    # Downhill trajectory vector
    v_downhill = np.array([np.cos(slope_pitch), 0.0, -np.sin(slope_pitch)])
    # Push vector is backward-upward relative to slope
    v_push_ramp = np.array([-np.cos(slope_pitch), 0.0, np.sin(slope_pitch)])

    step = 0
    in_cylinder = False
    max_z = 0.0

    while step < 1200:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        z = float(pos[2])
        x = float(pos[0])
        y = float(pos[1])
        spd = float(np.linalg.norm(vel[:2]))
        max_z = max(max_z, z)

        r = float(np.hypot(x, y))
        theta = float(np.arctan2(y, x))

        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        if x < -0.10 and not in_cylinder:
            # Stage 1: Downhill Power Acceleration Sprint
            phase_label = "Ramp Acceleration Sprint"
            # Project rods onto push direction
            p_push = dirs_world @ v_push_ramp
            # Project rods onto ground contact
            p_norm = dirs_world @ np.array([np.sin(slope_pitch), 0.0, np.cos(slope_pitch)])

            # Trailing contact rods fire with maximum extension
            is_push = (p_push > 0.0) & (p_norm < 0.20)
            targets = np.full(60, 0.010, dtype=np.float32)
            targets[is_push] = env.max_extend

        else:
            if not in_cylinder:
                in_cylinder = True
                print(f"🚀 IN CYLINDER at step {step}! Speed = {spd:.2f} m/s | Z = {z:.3f}m")

            # Stage 2: Wall of Death Centrifugal Spiral
            phase_label = "Wall of Death Spiral Ascent"

            # Tangential direction around circle (+theta)
            r_hat = np.array([np.cos(theta), np.sin(theta), 0.0]) if r > 1e-4 else np.array([1, 0, 0])
            theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])

            # Spiral push: pushes into wall (+r) + rearward along tangent (-theta) + downward (-z)
            push_3d = 0.65 * r_hat - 0.45 * theta_hat - 0.60 * np.array([0, 0, 1])
            push_3d = push_3d / np.linalg.norm(push_3d)

            p = dirs_world @ push_3d
            targets = np.full(60, 0.010, dtype=np.float32)
            targets[p > 0.25] = env.max_extend

        env.step(targets)

        a_c = (spd ** 2) / max(r, 0.1) if in_cylinder else 0.0

        if step % 4 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")

            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.core_body_id
            cam.distance = 3.20
            cam.elevation = -24.0
            cam.azimuth = 140.0 if not in_cylinder else float(np.degrees(theta)) + 120.0
            env.renderer.update_scene(env.data, camera=cam)
            frame = env.renderer.render()

            ann_frame = annotate(
                frame,
                "Acceleration Launch Ramp -> Cylinder Spiral",
                [
                    f"Phase: {phase_label}",
                    f"Position: ({x:+.2f}, {y:+.2f}, {z:.2f}m)",
                    f"Speed: {spd:.2f} m/s",
                    f"Centrifugal a_c: {a_c:.1f} m/s² ({a_c/9.81:.2f} G)",
                ],
                margin=14,
            )
            writer.append_data(ann_frame)

        if step % 25 == 0:
            print(f"Step {step:4d} [{phase_label:26s}]: pos=({x:+.2f}, {y:+.2f}, {z:.2f}m) | spd={spd:.2f}m/s | a_c={a_c:.1f}m/s²")

        if z >= 3.50 and in_cylinder:
            print(f"\nTop reached! Z = {z:.3f}m at step {step}")
            break

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished! Peak Z: {max_z:.3f}m")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/high_speed_ramp_cylinder_spiral.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

if __name__ == "__main__":
    main()
