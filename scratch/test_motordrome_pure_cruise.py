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
from skills.locomotion import move

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

    scenario = generate_scenario("motordrome", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    # Spawn on flat bottom floor
    env.data.qpos[0] = 0.70
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Motordrome Smooth Pure Cruising & Apron Climb ===")

    run_dir = make_run_dir(build_run_id("motordrome", "pure_cruise"))
    out_video = Path(run_dir) / "renders" / "motordrome_pure_cruise.mp4"
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

        # Dynamic target radius:
        # Step 0-300: r = 0.75m on flat floor
        # Step 300-800: expand r: 0.75 -> 1.40m up 45-deg apron ramp
        # Step 800+: r = 1.45m on vertical wall
        if step < 300:
            target_r = 0.75
            phase_label = "Floor Spin-Up (Torque)"
            cmd_spd = 2.4
        elif step < 900:
            prog = (step - 300) / 600.0
            target_r = 0.75 + prog * (1.42 - 0.75)
            phase_label = "45° Banked Apron Climb"
            cmd_spd = 2.8
        else:
            target_r = 1.45
            phase_label = "Vertical Wall of Death"
            cmd_spd = 3.0

        # Tangential direction (+theta) with gentle radial centering
        theta_hat = np.array([-np.sin(theta), np.cos(theta)])
        r_hat = np.array([np.cos(theta), np.sin(theta)])
        r_err = r - target_r
        d_cmd = theta_hat - 0.35 * r_err * r_hat
        d_cmd /= np.linalg.norm(d_cmd)

        targets = move(
            quat, env.dirs_body, env.max_extend,
            d_hat=d_cmd,
            speed=cmd_spd,
            back_gain=3.6,
            min_offset=0.025,
        )

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
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/motordrome_pure_cruise.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

if __name__ == "__main__":
    main()
