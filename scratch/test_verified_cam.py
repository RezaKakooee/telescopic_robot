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
    floor_r = 1.6
    wall_r = 2.4
    apron_h = 0.8
    total_h = 4.5

    cfg = load_config("configs/rl/circle_track.yaml")
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "scenario") or cfg.scenario is None:
        cfg.scenario = {}
    cfg.scenario.floor_radius = floor_r
    cfg.scenario.wall_radius = wall_r
    cfg.scenario.apron_height = apron_h
    cfg.scenario.cylinder_height = total_h

    scenario = generate_scenario("motordrome", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn at center of floor (r=1.0m)
    env.data.qpos[0] = 1.00
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print("=== Testing Verified Camera Motordrome ===")

    run_dir = make_run_dir(build_run_id("motordrome", "verified_cam"))
    out_video = Path(run_dir) / "renders" / "motordrome_verified_cam.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    step = 0
    max_z = 0.22
    accum_th = 0.0
    prev_th = np.arctan2(env.data.qpos[1], env.data.qpos[0])
    max_spd = 0.0
    max_ac = 0.0

    cam = mujoco.MjvCamera()

    while step < 1500:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        z = float(pos[2])
        x = float(pos[0])
        y = float(pos[1])
        spd = float(np.linalg.norm(vel[:3]))
        max_z = max(max_z, z)
        max_spd = max(max_spd, spd)

        r = float(np.hypot(x, y))
        theta = float(np.arctan2(y, x))

        d_th = theta - prev_th
        if d_th > np.pi: d_th -= 2*np.pi
        elif d_th < -np.pi: d_th += 2*np.pi
        accum_th += abs(d_th)
        prev_th = theta
        laps = accum_th / (2 * np.pi)

        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        r_hat = np.array([np.cos(theta), np.sin(theta), 0.0]) if r > 1e-4 else np.array([1, 0, 0])
        theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])

        if spd > 0.30:
            v_dir = vel / spd
            d_traj = 0.80 * v_dir + 0.20 * theta_hat
            d_traj /= np.linalg.norm(d_traj)
        else:
            d_traj = theta_hat

        # Steer inward on the floor to build up speed before climbing
        if spd < 4.2 and r < floor_r - 0.05:
            d_traj = d_traj - 0.2 * r_hat
            d_traj /= np.linalg.norm(d_traj)

        a_c = (spd ** 2) / max(r, 0.1)
        max_ac = max(max_ac, a_c)

        if r < floor_r - 0.05 and z < 0.28:
            phase_label = "Floor: Building Speed"
            u_push = -0.707 * d_traj - 0.707 * z_hat
            p_push = dirs_world @ u_push
            wave = np.clip((p_push - 0.05) / 0.70, 0.0, 1.0) * 1.5
            wave = np.clip(wave, 0.0, 1.0)
            wave[dirs_world @ d_traj > -0.05] = 0.0

            targets = np.full(60, 0.015, dtype=np.float32)
            targets[p_push > 0.05] = 0.015 + (env.max_extend - 0.015) * wave[p_push > 0.05]

        elif r < wall_r - 0.05 and z < apron_h + 0.05:
            phase_label = "Ramp: Climbing Apron"
            n_bank = (-r_hat + z_hat) / np.sqrt(2.0)

            u_push = -0.7 * d_traj - 0.7 * n_bank
            u_push /= np.linalg.norm(u_push)
            p_push = dirs_world @ u_push

            wave = np.clip((p_push - 0.05) / 0.65, 0.0, 1.0) * 1.5
            wave = np.clip(wave, 0.0, 1.0)
            wave[dirs_world @ d_traj > 0.2] = 0.0

            targets = np.full(60, 0.015, dtype=np.float32)
            targets[p_push > 0.05] = 0.015 + (env.max_extend - 0.015) * wave[p_push > 0.05]

        else:
            phase_label = "Wall: Riding Wall of Death"
            u_push = 0.7 * r_hat - 0.7 * d_traj
            u_push /= np.linalg.norm(u_push)
            p_push = dirs_world @ u_push

            wave = np.clip((p_push - 0.05) / 0.70, 0.0, 1.0) * 1.5
            wave = np.clip(wave, 0.0, 1.0)
            wave[dirs_world @ d_traj > 0.2] = 0.0

            targets = np.full(60, 0.015, dtype=np.float32)
            targets[p_push > 0.05] = 0.015 + (env.max_extend - 0.015) * wave[p_push > 0.05]

        env.step(targets)

        if step % 4 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")

            # Close-up interior tracking camera (distance=1.6m guarantees camera is inside the silo)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.core_body_id
            cam.distance = 1.8
            cam.elevation = -25.0
            cam.azimuth = float(np.degrees(theta)) + 140.0
            env.renderer.update_scene(env.data, camera=cam)
            frame = env.renderer.render()

            ann_frame = annotate(
                frame,
                "Authentic Motordrome (Wall of Death)",
                [
                    f"Phase: {phase_label}",
                    f"Altitude Z: {z:.3f} m (Radius: {r:.2f}m)",
                    f"Speed: {spd:.2f} m/s ({spd * 3.6:.1f} km/h)",
                    f"Centrifugal Acc: {a_c:.1f} m/s² ({a_c/9.81:.2f} G)",
                    f"Laps: {laps:.2f}",
                ],
                margin=14,
            )
            writer.append_data(ann_frame)

        if step % 50 == 0:
            print(f"Step {step:4d} [{phase_label:28s}]: r={r:.2f}m | z={z:.3f}m | spd={spd:.2f}m/s | laps={laps:.2f} | a_c={a_c:.1f}m/s² ({a_c/9.81:.2f}G)")

        step += 1

    writer.close()
    env.close()

    print(f"\nFinal: Peak Z = {max_z:.3f}m | Max Speed = {max_spd:.2f}m/s | Laps = {laps:.2f}")

    # Copy video to artifact path
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/motordrome_wall_of_death.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Copied to artifact path: {dst}")

if __name__ == "__main__":
    main()
