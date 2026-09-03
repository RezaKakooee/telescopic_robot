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
    cyl_rad = 0.65
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

    # Spawn resting on floor near the curved wall
    r_spawn = cyl_rad - 0.16
    env.data.qpos[0] = r_spawn
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Wall of Death Vortex Spiral (گشتاور & Centrifugal Climb) ===")
    print(f"Cylinder Radius: {cyl_rad:.2f}m | Critical Centrifugal Speed: {np.sqrt(9.81 * cyl_rad / 1.2):.2f} m/s")

    run_dir = make_run_dir(build_run_id("vortex_spiral", "wall_of_death"))
    out_video = Path(run_dir) / "renders" / "wall_of_death_vortex_spiral.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    step = 0
    max_z = 0.22

    while step < 2000:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        angvel = env.data.qvel[3:6].copy()
        z = float(pos[2])
        max_z = max(max_z, z)

        # Cylindrical coordinates
        dx, dy = pos[0], pos[1]
        r = float(np.hypot(dx, dy))
        theta = float(np.arctan2(dy, dx))
        v_tan = float(-vel[0] * np.sin(theta) + vel[1] * np.cos(theta))
        v_rad = float(vel[0] * np.cos(theta) + vel[1] * np.sin(theta))

        # Radial unit vector (pointing from cylinder center to ball)
        r_hat = np.array([np.cos(theta), np.sin(theta), 0.0])
        # Tangential direction (forward along the circle)
        theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])

        # Wall-of-Death Helical Vector:
        # At start (spin-up): 4 deg pitch to build pure speed & centrifugal torque
        # Once v_tan > 1.8 m/s: increase pitch to 24 deg to climb the spiral!
        if step < 180:
            pitch_deg = 4.0 + (step / 180.0) * 14.0
            phase_label = "Spin-Up (گشتاور)"
        else:
            pitch_deg = 22.0
            phase_label = "Vortex Helical Climb"

        pitch_rad = np.radians(pitch_deg)
        # Trajectory direction
        u_traj = np.cos(pitch_rad) * theta_hat + np.sin(pitch_rad) * z_hat
        # Push direction is opposite to trajectory (rear-downward push against contact surface)
        u_push = -u_traj

        # World directions of rods
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        # Projections
        p_push = dirs_world @ u_push
        p_radial = dirs_world @ r_hat   # pointing outward toward the curved wall
        p_z = dirs_world[:, 2]

        # Active CPG push wave on the contact quadrant (facing wall + pointing backward along spiral)
        # Rods pushing against the wall:
        is_contact_quadrant = (p_push > 0.05) & (p_radial > -0.20)
        drive_wave = np.clip((p_push - 0.05) / 0.70, 0.0, 1.0)

        # Ground support when near bottom
        if z < 0.35:
            is_contact_quadrant |= (p_z < -0.30) & (p_push > 0.0)

        targets = np.full(60, 0.015, dtype=np.float32)
        # Contact rods extend dynamically with high power
        targets[is_contact_quadrant] = 0.02 + 0.14 * drive_wave[is_contact_quadrant]

        # Outward radial bias to maintain contact on curved wall
        wall_bias = (p_radial > 0.40) & (abs(p_z) < 0.50)
        targets[wall_bias] = np.maximum(targets[wall_bias], 0.09)

        # Centrifugal acceleration term
        f_centrifugal = (vel[0]**2 + vel[1]**2) / max(r, 0.1)

        env.step(targets)

        if step % 4 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")

            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.core_body_id
            cam.distance = 2.60
            cam.elevation = -24.0
            cam.azimuth = float(np.degrees(theta)) + 120.0  # Orbital chase cam
            env.renderer.update_scene(env.data, camera=cam)
            frame = env.renderer.render()

            ann_frame = annotate(
                frame,
                "Wall of Death Helical Vortex (گشتاور)",
                [
                    f"Phase: {phase_label}",
                    f"Altitude Z: {z:.3f} m",
                    f"Tangential Velocity: {v_tan:+.2f} m/s",
                    f"Centrifugal Acc: {f_centrifugal:.1f} m/s²",
                ],
                margin=14,
            )
            writer.append_data(ann_frame)

        if step % 50 == 0:
            print(f"Step {step:4d} [{phase_label:22s}]: z={z:.3f}m | v_tan={v_tan:+.2f}m/s | vz={vel[2]:+.2f}m/s | r={r:.3f}m | a_c={f_centrifugal:.1f}m/s²")

        if z >= 3.50:
            print(f"Top reached! Z = {z:.3f}m")
            break

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished! Start Z: 0.22m -> Peak Z: {max_z:.3f}m")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/wall_of_death_vortex_spiral.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

if __name__ == "__main__":
    main()
