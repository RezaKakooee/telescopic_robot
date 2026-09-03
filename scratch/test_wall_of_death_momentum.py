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
from skills.locomotion import move

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

    # Spawn at top of launch ramp
    y_target = -cyl_rad
    env.data.qpos[0] = -3.80
    env.data.qpos[1] = y_target
    env.data.qpos[2] = 1.35
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Wall of Death High-Speed Momentum Spiral ===")

    run_dir = make_run_dir(build_run_id("ramp_launch", "wall_of_death_spiral"))
    out_video = Path(run_dir) / "renders" / "wall_of_death_ramp_spiral.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    step = 0
    in_cylinder = False
    max_z = 0.0
    entry_spd = 0.0
    laps = 0.0
    prev_th = -np.pi / 2.0
    accum_th = 0.0

    while step < 1500:
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

        if x < -0.15 and not in_cylinder:
            # Stage 1: Downhill Power Sprint down the ramp
            phase_label = "Ramp Acceleration Sprint"
            y_err = y - y_target
            d_cmd = np.array([1.0, -2.5 * y_err])
            d_cmd /= np.linalg.norm(d_cmd)

            targets = move(
                quat, env.dirs_body, env.max_extend,
                d_hat=d_cmd,
                speed=2.8,
                back_gain=3.2,
                min_offset=0.025
            )

        else:
            if not in_cylinder:
                in_cylinder = True
                entry_spd = spd
                prev_th = theta
                print(f"\n⚡ ENTERED CYLINDER at step {step}! Speed = {spd:.2f} m/s | Z = {z:.3f}m | Centrifugal a_c = {(spd**2)/r:.1f} m/s² ({(spd**2)/(r*9.81):.2f} G)")

            # Track lap progress around the cylinder
            d_th = theta - prev_th
            if d_th > np.pi: d_th -= 2*np.pi
            elif d_th < -np.pi: d_th += 2*np.pi
            accum_th += max(d_th, 0.0)
            prev_th = theta
            laps = accum_th / (2 * np.pi)

            # Stage 2: Wall of Death High-Speed Spiral
            phase_label = "Wall of Death Spiral Ascent"

            # Tangential direction around circle (+theta)
            r_hat = np.array([np.cos(theta), np.sin(theta), 0.0]) if r > 1e-4 else np.array([1, 0, 0])
            theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])

            # Push vector against the vertical wall:
            # Pushes into wall (+r_hat) + backward along tangent (-theta_hat) + downward (-z)
            push_3d = 0.55 * r_hat - 0.60 * theta_hat - 0.58 * np.array([0, 0, 1])
            push_3d = push_3d / np.linalg.norm(push_3d)

            # World projection of rods
            p_push = dirs_world @ push_3d
            p_radial = dirs_world @ r_hat  # pointing outward into vertical wall

            # Wave drive on trailing wall-contact hemisphere
            drive_wave = np.clip((p_push - 0.05) / 0.70, 0.0, 1.0)
            targets = np.full(60, 0.015, dtype=np.float32)
            # Contact rods push with full power
            targets[p_push > 0.05] = 0.02 + 0.14 * drive_wave[p_push > 0.05]

            # Inward-facing rods stay tucked so they don't drag in air
            targets[p_radial < -0.10] = 0.010

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
            cam.azimuth = 135.0 if not in_cylinder else float(np.degrees(theta)) + 120.0
            env.renderer.update_scene(env.data, camera=cam)
            frame = env.renderer.render()

            ann_frame = annotate(
                frame,
                "Acceleration Launch Ramp -> Wall of Death",
                [
                    f"Phase: {phase_label}",
                    f"Position: ({x:+.2f}, {y:+.2f}, {z:.2f}m)",
                    f"Speed: {spd:.2f} m/s",
                    f"Laps: {laps:.2f} | a_c: {a_c:.1f} m/s² ({a_c/9.81:.2f} G)",
                ],
                margin=14,
            )
            writer.append_data(ann_frame)

        if step % 25 == 0:
            print(f"Step {step:4d} [{phase_label:26s}]: pos=({x:+.2f}, {y:+.2f}, {z:.2f}m) | spd={spd:.2f}m/s | laps={laps:.2f} | a_c={a_c:.1f}m/s²")

        if z >= 3.50 and in_cylinder:
            print(f"\nTop reached! Z = {z:.3f}m at step {step}")
            break

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished! Start Z: 1.35m -> Entry Speed: {entry_spd:.2f} m/s -> Peak Z: {max_z:.3f}m | Laps: {laps:.2f}")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/wall_of_death_ramp_spiral.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

if __name__ == "__main__":
    main()
