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
    floor_r = 1.0
    wall_r = 1.6
    apron_h = 0.6
    total_h = 4.0

    cfg = load_config("configs/rl/circle_track.yaml")
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "scenario") or cfg.scenario is None:
        cfg.scenario = {}
    cfg.scenario.floor_radius = floor_r
    cfg.scenario.wall_radius = wall_r
    cfg.scenario.apron_height = apron_h
    cfg.scenario.cylinder_height = total_h
    if hasattr(cfg, "sim2real") and cfg.sim2real is not None:
        cfg.sim2real.rubber_friction_sliding = 1.4
        cfg.sim2real.actuator_force_limit = 100.0

    scenario = generate_scenario("motordrome", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    # Spawn on flat bottom floor
    env.data.qpos[0] = 0.70
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Motordrome Smooth Apron Ascent & Wall Riding ===")

    run_dir = make_run_dir(build_run_id("motordrome", "smooth_ascent"))
    out_video = Path(run_dir) / "renders" / "motordrome_smooth_ascent.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    step = 0
    max_z = 0.22
    accum_th = 0.0
    prev_th = 0.0

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
        accum_th += max(d_th, 0.0)
        prev_th = theta
        laps = accum_th / (2 * np.pi)

        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        # Radial, Tangential, and Vertical unit vectors
        r_hat = np.array([np.cos(theta), np.sin(theta), 0.0]) if r > 1e-4 else np.array([1, 0, 0])
        theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])

        # Local surface slope angle beta
        if r < 0.95:
            # Flat bottom floor
            beta = 0.0
            phase_label = "Floor Spin-Up (Torque)"
            climb_pitch = 0.0
            target_r = 0.80
        elif r < 1.50 and z < 0.65:
            # 45-degree Banked Apron Ramp
            beta = np.radians(45.0)
            phase_label = "45° Banked Apron Climb"
            climb_pitch = np.radians(16.0)
            target_r = 1.45
        else:
            # Vertical 90-degree Wooden Wall
            beta = np.radians(90.0)
            phase_label = "Vertical Wall of Death"
            climb_pitch = np.radians(20.0)
            target_r = 1.48

        # Local surface normal and up-slope vectors
        n_surf = -np.sin(beta) * r_hat + np.cos(beta) * z_hat
        u_upslope = np.cos(beta) * r_hat + np.sin(beta) * z_hat

        # Trajectory heading along surface
        fwd_traj = np.cos(climb_pitch) * theta_hat + np.sin(climb_pitch) * u_upslope
        fwd_traj /= np.linalg.norm(fwd_traj)
        # Push vector is opposite to trajectory (trailing contact zone)
        push_3d = -fwd_traj - 0.25 * n_surf
        push_3d /= np.linalg.norm(push_3d)

        # Decompose body rods along push vector and surface normal
        p_push = dirs_world @ push_3d
        p_surf = dirs_world @ (-n_surf) # rods pointing into surface

        # Active push wave
        gain = 3.2 if r < 0.95 else 3.8
        drive_wave = np.clip((p_push - 0.05) / 0.65, 0.0, 1.0)
        targets = np.full(60, 0.025, dtype=np.float32)
        # Push rods extend dynamically
        is_push = (p_push > 0.05) & (p_surf > -0.20)
        targets[is_push] = 0.025 + (env.max_extend - 0.025) * drive_wave[is_push]

        env.step(targets)

        a_c = (spd ** 2) / max(r, 0.1)

        if step % 4 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")

            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.core_body_id
            cam.distance = 3.80
            cam.elevation = -28.0
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
            print(f"Step {step:4d} [{phase_label:26s}]: r={r:.2f}m | z={z:.3f}m | spd={spd:.2f}m/s | laps={laps:.2f} | a_c={a_c:.1f}m/s²")

        if z >= 3.50:
            print(f"\nTop rim reached! Z = {z:.3f}m at step {step}")
            break

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished! Final: r={r:.2f}m, z={z:.3f}m | Peak Z: {max_z:.3f}m | Laps: {laps:.2f}")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/motordrome_smooth_ascent.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

if __name__ == "__main__":
    main()
