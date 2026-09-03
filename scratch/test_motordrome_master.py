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
    cfg = load_config("configs/rl/circle_track.yaml")
    scenario = generate_scenario("motordrome", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    # Spawn on the apron ramp at r = 1.80m, z = 0.20m (pre-accelerating down the banked bowl)
    th_0 = 0.0
    env.data.qpos[0] = 1.80 * np.cos(th_0)
    env.data.qpos[1] = 1.80 * np.sin(th_0)
    env.data.qpos[2] = 0.35
    # Initial circular velocity along tangent
    env.data.qvel[0] = -2.0 * np.sin(th_0)
    env.data.qvel[1] = 2.0 * np.cos(th_0)
    env.data.qvel[2] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Motordrome Master Wall of Death Ascent & High Orbit ===")

    run_dir = make_run_dir(build_run_id("motordrome", "master"))
    out_video = Path(run_dir) / "renders" / "motordrome_master.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    step = 0
    max_z = 0.35
    accum_th = 0.0
    prev_th = th_0

    while step < 1600:
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

        d_th = theta - prev_th
        if d_th > np.pi: d_th -= 2*np.pi
        elif d_th < -np.pi: d_th += 2*np.pi
        accum_th += abs(d_th)
        prev_th = theta
        laps = accum_th / (2 * np.pi)

        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        # Cylindrical coordinates
        r_hat = np.array([np.cos(theta), np.sin(theta), 0.0]) if r > 1e-4 else np.array([1, 0, 0])
        theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])

        if z < 0.70:
            phase_label = "45° Apron Ramp Ascent"
            # Bank normal (inward & upward)
            n_bank = (-r_hat + z_hat) / np.sqrt(2.0)
            u_upslope = (r_hat + z_hat) / np.sqrt(2.0)
            # Helical climb heading on the bank
            u_traj = np.cos(np.radians(15.0)) * theta_hat + np.sin(np.radians(15.0)) * u_upslope
            # Push vector: opposite to trajectory + into the slope
            u_push = -u_traj - 0.25 * n_bank
            u_push /= np.linalg.norm(u_push)

            p_push = dirs_world @ u_push
            wave = np.clip((p_push - 0.05) / 0.65, 0.0, 1.0)
            wave[dirs_world @ u_traj > -0.05] = 0.0

            targets = np.full(60, 0.025, dtype=np.float32)
            targets[p_push > 0.05] = 0.025 + (env.max_extend - 0.025) * wave[p_push > 0.05]

        else:
            phase_label = "Vertical Wall of Death High Orbit"
            # On vertical wall: push against wall (+r_hat) + rearward (-theta_hat) + downward (-z_hat)
            u_push = 0.55 * r_hat - 0.55 * theta_hat - 0.62 * z_hat
            u_push /= np.linalg.norm(u_push)

            p_push = dirs_world @ u_push
            wave = np.clip((p_push - 0.05) / 0.65, 0.0, 1.0)
            wave[dirs_world @ theta_hat > -0.05] = 0.0

            targets = np.full(60, 0.025, dtype=np.float32)
            targets[p_push > 0.05] = 0.025 + (env.max_extend - 0.025) * wave[p_push > 0.05]

        env.step(targets)

        a_c = (spd ** 2) / max(r, 0.1)

        if step % 4 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")

            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.core_body_id
            cam.distance = 4.20
            cam.elevation = -26.0
            cam.azimuth = float(np.degrees(theta)) + 115.0
            env.renderer.update_scene(env.data, camera=cam)
            frame = env.renderer.render()

            ann_frame = annotate(
                frame,
                "Motordrome Wall of Death Simulation",
                [
                    f"Phase: {phase_label}",
                    f"Altitude Z: {z:.3f} m (Radius: {r:.2f}m)",
                    f"Speed: {spd:.2f} m/s | Laps: {laps:.2f}",
                    f"Centrifugal Acc: {a_c:.1f} m/s² ({a_c/9.81:.2f} G)",
                ],
                margin=14,
            )
            writer.append_data(ann_frame)

        if step % 25 == 0:
            print(f"Step {step:4d} [{phase_label:32s}]: r={r:.2f}m | z={z:.3f}m | spd={spd:.2f}m/s | laps={laps:.2f} | a_c={a_c:.1f}m/s² ({a_c/9.81:.2f}G)")

        if z >= 3.80:
            print(f"\nTop rim reached! Z = {z:.3f}m at step {step}")
            break

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished! Final: r={r:.2f}m, z={z:.3f}m | Peak Z: {max_z:.3f}m | Laps: {laps:.2f}")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/motordrome_master.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

if __name__ == "__main__":
    main()
