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

    # Spawn on bottom floor
    env.data.qpos[0] = 0.70
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Motordrome High-Torque Tight Orbit & Apron Climb ===")

    run_dir = make_run_dir(build_run_id("motordrome", "tight_orbit"))
    out_video = Path(run_dir) / "renders" / "motordrome_tight_orbit.mp4"
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

        # Cylindrical frame basis
        r_hat = np.array([np.cos(theta), np.sin(theta), 0.0]) if r > 1e-4 else np.array([1, 0, 0])
        theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])

        # Progressive target radius:
        # Step 0-400: spin-up on floor (r = 0.75m)
        # Step 400-1000: expand onto 45-deg apron ramp (r: 0.75 -> 1.45m)
        # Step 1000+: vertical wooden wall orbit (r = 1.48m)
        if step < 400:
            target_r = 0.75
            phase_label = "Floor Spin-Up (Torque)"
            gain = 3.2
            pitch_deg = 0.0
        elif step < 1000:
            prog = (step - 400) / 600.0
            target_r = 0.75 + prog * (1.45 - 0.75)
            phase_label = "45° Apron Ramp Climb"
            gain = 3.8
            pitch_deg = 15.0
        else:
            target_r = 1.48
            phase_label = "Vertical Wall of Death"
            gain = 4.2
            pitch_deg = 22.0

        # Continuous Tangential Travel vector with radial centering & vertical pitch
        r_err = r - target_r
        # Heading: forward (+theta) - radial correction (-r_hat) + pitch (+z_hat)
        heading_3d = theta_hat - 1.5 * r_err * r_hat + np.tan(np.radians(pitch_deg)) * z_hat
        heading_3d /= np.linalg.norm(heading_3d)

        # Decompose body rods along instantaneous heading
        u_long = dirs_world @ heading_3d
        u_lat = dirs_world @ r_hat
        u_z = dirs_world[:, 2]

        rear = np.clip((-u_long - 0.08) / 0.92, 0.0, 1.0)
        # Downward/surface contact factor
        down = np.clip(1.0 - abs(u_z + 0.35) / 0.85, 0.0, 1.0) if z < 0.35 else np.clip(1.0 - abs(u_lat - 0.40) / 0.80, 0.0, 1.0)
        wave = np.clip((rear ** 1.1) * down * gain, 0.0, 1.0)

        # Leading rods stay strictly shut to prevent braking
        wave[u_long > -0.05] = 0.0
        if z < 0.35:
            wave[u_z > 0.10] = 0.0

        targets = 0.020 + (env.max_extend - 0.020) * wave
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

        if step % 50 == 0:
            print(f"Step {step:4d} [{phase_label:28s}]: r={r:.2f}m | z={z:.3f}m | spd={spd:.2f}m/s | laps={laps:.2f} | a_c={a_c:.1f}m/s² ({a_c/9.81:.2f}G)")

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished! Final: r={r:.2f}m, z={z:.3f}m | Peak Z: {max_z:.3f}m | Laps: {laps:.2f}")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/motordrome_tight_orbit.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

if __name__ == "__main__":
    main()
