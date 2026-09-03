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

    # Spawn on floor at r=1.0m
    env.data.qpos[0] = 1.00
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print("=== Pure Motorcycle Wall of Death Controller ===")

    run_dir = make_run_dir(build_run_id("motordrome", "pure_motorcycle"))
    out_video = Path(run_dir) / "renders" / "motordrome_pure_motorcycle.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    step = 0
    max_z = 0.22
    accum_th = 0.0
    prev_th = 0.0
    max_spd = 0.0
    max_ac = 0.0

    cam = mujoco.MjvCamera()

    while step < 2500:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        z = float(pos[2])
        x = float(pos[0])
        y = float(pos[1])
        spd = float(np.linalg.norm(vel))
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

        r_hat = np.array([np.cos(theta), np.sin(theta), 0.0]) if r > 1e-4 else np.array([1.0, 0.0, 0.0])
        theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])

        # State determination purely by geometry radius:
        if r < floor_r - 0.05:
            # Floor stage (r < 1.55m)
            phase = "1. Floor Spin-Up"
            if spd < 4.8:
                # Active centripetal steering to hold r = 1.1m circle while accelerating
                r_err = r - 1.10
                inward_steer = np.clip(0.40 + 0.80 * r_err, 0.10, 0.90)
                v_target = theta_hat - inward_steer * r_hat
            else:
                # Once v >= 4.8 m/s, release centripetal hold and transition outward to ramp
                v_target = theta_hat + 0.20 * r_hat
            v_target /= np.linalg.norm(v_target)
            
            # Floor normal is +Z
            n_surf = z_hat
            # Maximum torque: push strongly backward along travel direction
            u_push = -0.90 * v_target - 0.40 * n_surf
        elif r < wall_r - 0.08:
            # 45 deg Apron Ramp stage (1.55m <= r < 2.32m)
            phase = "2. Apron Climb"
            n_surf = (-r_hat + z_hat) / np.sqrt(2.0)
            u_upslope = (r_hat + z_hat) / np.sqrt(2.0)
            
            # Tangential drive + slight climb angle
            climb_angle = np.radians(12.0)
            v_target = np.cos(climb_angle) * theta_hat + np.sin(climb_angle) * u_upslope
            v_target /= np.linalg.norm(v_target)
            
            # Push into ramp + drive forward
            u_push = -0.85 * v_target - 0.55 * n_surf
        else:
            # Vertical Wall stage (r >= 2.32m)
            phase = "3. Wall of Death Ride"
            n_surf = -r_hat
            
            # Pure tangential drive + slight up
            climb_angle = np.radians(8.0)
            v_target = np.cos(climb_angle) * theta_hat + np.sin(climb_angle) * z_hat
            v_target /= np.linalg.norm(v_target)
            
            # Centrifugal force holds robot against wall; push into wall & backward
            u_push = -0.85 * v_target - 0.55 * n_surf

        u_push /= np.linalg.norm(u_push)

        a_c = (spd ** 2) / max(r, 0.1)
        max_ac = max(max_ac, a_c)

        # CPG Wave: Rods pointing towards u_push extend (push off surface)
        p_push = dirs_world @ u_push
        p_lead = dirs_world @ v_target

        wave = np.clip((p_push - 0.05) / 0.65, 0.0, 1.0) * 1.5
        wave = np.clip(wave, 0.0, 1.0)
        # Only tuck rods pointing directly forward in the direction of motion
        wave[p_lead > 0.45] = 0.0

        min_ext = 0.015
        max_ext = env.max_extend
        targets = min_ext + (max_ext - min_ext) * wave

        env.step(targets.astype(np.float32))

        # Render tracking camera showing full arena & ball clearly
        if step % 4 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")

            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            # Look at robot location with elevated offset
            cam.lookat[0] = 0.0
            cam.lookat[1] = 0.0
            cam.lookat[2] = max(z, 0.8)
            cam.distance = 6.2
            cam.elevation = -28.0
            cam.azimuth = float(np.degrees(theta)) + 140.0
            env.renderer.update_scene(env.data, camera=cam)
            frame = env.renderer.render()

            ann_frame = annotate(
                frame,
                "Authentic Wall of Death (Motorcycle CPG)",
                [
                    f"Phase: {phase}",
                    f"Altitude Z: {z:.2f} m | Radius: {r:.2f} m",
                    f"Speed: {spd:.2f} m/s ({spd * 3.6:.1f} km/h)",
                    f"Centrifugal Acc: {a_c/9.81:.2f} G ({a_c:.1f} m/s²)",
                    f"Laps: {laps:.2f}",
                ],
                margin=14,
            )
            writer.append_data(ann_frame)

        if step % 50 == 0:
            print(f"Step {step:4d} [{phase:20s}]: r={r:.2f}m | z={z:.3f}m | spd={spd:.2f}m/s | laps={laps:.2f} | a_c={a_c:.1f}m/s² ({a_c/9.81:.2f}G)")

        if z >= 3.8:
            print(f"Rim reached at step {step}!")
            break

        step += 1

    writer.close()
    env.close()

    print(f"\nFinal Summary: Peak Z = {max_z:.3f}m | Max Speed = {max_spd:.2f}m/s | Laps = {laps:.2f}")

if __name__ == "__main__":
    main()
