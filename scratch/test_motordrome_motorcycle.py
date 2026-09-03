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

def get_surface_normal_and_target(pos, vel, floor_r=1.6, wall_r=2.4, apron_h=0.8):
    """
    Computes exact continuous surface normal and rolling propulsion target.
    Like a motorcycle:
    1. Forward drive vector (tangential, CCW along cylinder) + slight climb angle.
    2. Contact normal vector pointing out of the surface towards robot.
    """
    x, y, z = pos[:3]
    r = float(np.hypot(x, y))
    theta = float(np.arctan2(y, x))
    
    r_hat = np.array([np.cos(theta), np.sin(theta), 0.0]) if r > 1e-4 else np.array([1.0, 0.0, 0.0])
    theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])
    z_hat = np.array([0.0, 0.0, 1.0])
    
    # Surface normal pointing OUT of the surface towards the robot center
    if r <= floor_r:
        # Flat floor: normal is +Z
        n_surf = z_hat
        phase = "Floor Spin-Up"
    elif r < wall_r and z < apron_h + 0.1:
        # 45 deg apron ramp: normal points inward & upward
        n_surf = (-r_hat + z_hat) / np.sqrt(2.0)
        phase = "Apron Bank Climb"
    else:
        # Vertical wall: normal points horizontally inward (-r_hat)
        n_surf = -r_hat
        phase = "Vertical Wall Ride"
        
    return n_surf, r_hat, theta_hat, z_hat, r, theta, phase

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

    # Spawn on floor at r=1.1m
    env.data.qpos[0] = 1.10
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print("=== Testing Continuous Smooth Motordrome Motorcycle Controller ===")

    run_dir = make_run_dir(build_run_id("motordrome", "motorcycle_ride"))
    out_video = Path(run_dir) / "renders" / "motordrome_motorcycle_ride.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    step = 0
    max_z = 0.22
    accum_th = 0.0
    prev_th = 0.0
    max_spd = 0.0
    max_ac = 0.0

    # Camera placement: Elevated interior camera mounted high inside the cylinder
    # looking across and down at the robot so both the floor, ramp, and vertical wall are clearly visible!
    cam = mujoco.MjvCamera()

    while step < 2000:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        z = float(pos[2])
        x = float(pos[0])
        y = float(pos[1])
        spd = float(np.linalg.norm(vel))
        max_z = max(max_z, z)
        max_spd = max(max_spd, spd)

        n_surf, r_hat, theta_hat, z_hat, r, theta, phase = get_surface_normal_and_target(
            pos, vel, floor_r, wall_r, apron_h
        )

        d_th = theta - prev_th
        if d_th > np.pi: d_th -= 2*np.pi
        elif d_th < -np.pi: d_th += 2*np.pi
        accum_th += abs(d_th)
        prev_th = theta
        laps = accum_th / (2 * np.pi)

        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        # Motorcycle throttle direction:
        # We want to travel CCW along theta_hat.
        # As speed builds, we allow climbing upward along the surface.
        if phase == "Floor Spin-Up":
            # On flat floor: steer purely in CCW circle (with slight inward centering until speed builds)
            if spd < 3.5:
                # Keep inside circle
                v_target = theta_hat - 0.15 * r_hat
            else:
                # Let centrifugal force drift out to ramp
                v_target = theta_hat + 0.10 * r_hat
        elif phase == "Apron Bank Climb":
            # On ramp: drive tangentially + slight up-slope to ascend
            u_upslope = (r_hat + z_hat) / np.sqrt(2.0)
            climb_angle = np.radians(15.0) if spd > 3.0 else np.radians(5.0)
            v_target = np.cos(climb_angle) * theta_hat + np.sin(climb_angle) * u_upslope
        else:
            # On vertical wall: drive tangentially + slight upward pitch to ascend
            climb_angle = np.radians(12.0) if spd > 3.5 else np.radians(2.0)
            v_target = np.cos(climb_angle) * theta_hat + np.sin(climb_angle) * z_hat

        v_target /= np.linalg.norm(v_target)

        # Centrifugal acceleration
        a_c = (spd ** 2) / max(r, 0.1)
        max_ac = max(max_ac, a_c)

        # Contact push vector:
        # The robot pushes AGAINST the surface (in direction of -n_surf)
        # AND pushes BACKWARD along the travel direction (-v_target) to generate forward torque.
        # This mimics the contact patch of a tire!
        # Contact vector in world coordinates:
        u_contact = -0.65 * n_surf - 0.75 * v_target
        u_contact /= np.linalg.norm(u_contact)

        # Continuous smooth CPG wave
        # Rods pointing towards u_contact extend, rods pointing forward retract
        p_contact = dirs_world @ u_contact
        p_forward = dirs_world @ v_target

        # Smooth bell-curve extension around u_contact
        wave = np.clip((p_contact - 0.10) / 0.65, 0.0, 1.0)
        # Smoothly tuck rods that are leading in front to prevent snagging / tripping
        tuck_factor = np.clip(1.0 - np.clip(p_forward / 0.35, 0.0, 1.0), 0.0, 1.0)
        final_ext = wave * tuck_factor

        min_ext = 0.015
        max_ext = env.max_extend
        targets = min_ext + (max_ext - min_ext) * final_ext

        env.step(targets.astype(np.float32))

        # Render
        if step % 4 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")

            # Camera: Looking from above opposite side across the cylinder
            # Placed at height Z=3.5, looking down at the robot
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[0] = 0.0
            cam.lookat[1] = 0.0
            cam.lookat[2] = 1.5
            cam.distance = 5.8
            cam.elevation = -32.0
            cam.azimuth = float(np.degrees(theta)) + 120.0 # smoothly tracks angle
            env.renderer.update_scene(env.data, camera=cam)
            frame = env.renderer.render()

            ann_frame = annotate(
                frame,
                "Motordrome Motorcycle Wall of Death",
                [
                    f"Phase: {phase}",
                    f"Altitude Z: {z:.2f} m | Radius: {r:.2f} m",
                    f"Speed: {spd:.2f} m/s ({spd * 3.6:.1f} km/h)",
                    f"Centrifugal: {a_c/9.81:.2f} G ({a_c:.1f} m/s²)",
                    f"Laps: {laps:.2f}",
                ],
                margin=14,
            )
            writer.append_data(ann_frame)

        if step % 50 == 0:
            print(f"Step {step:4d} [{phase:18s}]: r={r:.2f}m | z={z:.3f}m | spd={spd:.2f}m/s | laps={laps:.2f} | a_c={a_c:.1f}m/s² ({a_c/9.81:.2f}G)")

        if z >= 3.8:
            print(f"Reached rim at step {step}!")
            break

        step += 1

    writer.close()
    env.close()

    print(f"\nSummary: Peak Z = {max_z:.3f}m | Max Speed = {max_spd:.2f}m/s | Laps = {laps:.2f}")

if __name__ == "__main__":
    main()
