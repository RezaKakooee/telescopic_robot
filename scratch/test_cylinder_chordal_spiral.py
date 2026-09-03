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
    cyl_rad = 0.36
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

    # Spawn resting on floor at base
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Chordal Helical Spiral Climbing in Vertical Cylinder (R = {cyl_rad:.2f}m) ===")

    run_dir = make_run_dir(build_run_id("cylinder_spiral", "chordal_helix"))
    out_video = Path(run_dir) / "renders" / "vertical_cylinder_chordal_spiral_climb.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    # Settle at base
    for _ in range(15):
        t = np.zeros(60, dtype=np.float32)
        t[env.dirs_body[:, 2] < -0.30] = 0.045
        env.step(t)

    # 1. Floor Launch (Explosive upward launch)
    print("🚀 1. Floor Launch...")
    for step in range(14):
        R = quat_to_rotmat(env.data.qpos[3:7].copy())
        dirs_world = env.dirs_body @ R.T
        t = np.zeros(60, dtype=np.float32)
        t[(dirs_world[:, 2] < -0.15) & (np.hypot(dirs_world[:, 0], dirs_world[:, 1]) < 0.40)] = env.max_extend
        env.step(t)
        if step % 4 == 0:
            render_frame(env, writer, "Floor Takeoff", env.data.qpos[2], env.data.qvel[2], 0)

    # Flight to Apex 1
    for step in range(18):
        env.step(np.full(60, 0.010, dtype=np.float32))
        if step % 4 == 0:
            render_frame(env, writer, "Ascent to Apex 1", env.data.qpos[2], env.data.qvel[2], 1)

    print(f"Apex 1 reached: z = {env.data.qpos[2]:.3f}m")

    # Spiral climb loop: 6 triangular / chordal wall launches spiraling around the 360-degree cylinder
    current_wall_angle = 0.0
    stride_count = 0

    for spiral_idx in range(7):
        z_curr = float(env.data.qpos[2])
        if z_curr >= 3.50:
            print(f"Target reached at stride {spiral_idx}! Z = {z_curr:.3f}m")
            break

        # A. Apex Wall Clamp: lock onto cylinder wall
        print(f"🔒 Stride {spiral_idx+1}: Wall Lock at z = {z_curr:.3f}m...")
        for step in range(35):
            R = quat_to_rotmat(env.data.qpos[3:7].copy())
            dirs_world = env.dirs_body @ R.T
            lat_proj = np.hypot(dirs_world[:, 0], dirs_world[:, 1])
            t = np.full(60, 0.010, dtype=np.float32)
            t[(lat_proj > 0.60) & (abs(dirs_world[:, 2]) < 0.45)] = 0.12
            env.step(t)
            if step % 4 == 0:
                render_frame(env, writer, f"Wall Lock #{spiral_idx+1}", env.data.qpos[2], env.data.qvel[2], spiral_idx+1)

        # B. Helical Shove: rotate launch angle by +120 degrees around cylinder axis
        # Push vector points into current wall (-angle) + downward (-z)
        launch_angle = current_wall_angle
        push_dir_xy = np.array([np.cos(launch_angle), np.sin(launch_angle), 0.0])
        # Upward-forward flight direction
        flight_angle = launch_angle + np.radians(120.0)
        flight_dir_xy = np.array([np.cos(flight_angle), np.sin(flight_angle), 0.0])

        push_3d = 0.70 * push_dir_xy - 0.70 * np.array([0, 0, 1])
        push_3d = push_3d / np.linalg.norm(push_3d)

        print(f"💥 Stride {spiral_idx+1}: Helical Shove (گشتاور) at angle {np.degrees(launch_angle):.0f}°...")
        for step in range(12):
            R = quat_to_rotmat(env.data.qpos[3:7].copy())
            dirs_world = env.dirs_body @ R.T
            t = np.full(60, 0.010, dtype=np.float32)
            p = dirs_world @ push_3d
            t[p > 0.35] = env.max_extend
            env.step(t)
            if step % 4 == 0:
                render_frame(env, writer, f"Helical Shove (گشتاور) #{spiral_idx+1}", env.data.qpos[2], env.data.qvel[2], spiral_idx+1)

        # C. Flight across cylinder chord: all rods tucked in air
        for step in range(20):
            env.step(np.full(60, 0.010, dtype=np.float32))
            if step % 4 == 0:
                render_frame(env, writer, f"Spiral Flight #{spiral_idx+1}", env.data.qpos[2], env.data.qvel[2], spiral_idx+1)

        current_wall_angle = flight_angle
        stride_count += 1
        print(f"Stride {stride_count} completed: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # Hold at top
    print(f"\nFinal Z: {env.data.qpos[2]:.3f}m! Holding apex lock...")
    for _ in range(40):
        R = quat_to_rotmat(env.data.qpos[3:7].copy())
        dirs_world = env.dirs_body @ R.T
        lat_proj = np.hypot(dirs_world[:, 0], dirs_world[:, 1])
        t = np.zeros(60, dtype=np.float32)
        t[lat_proj > 0.60] = 0.12
        env.step(t)
        render_frame(env, writer, "Apex Lock at Top Rim", env.data.qpos[2], env.data.qvel[2], stride_count)

    writer.close()
    env.close()

    print(f"\nVideo saved to: {out_video}")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/vertical_cylinder_chordal_spiral_climb.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Copied to artifact path: {dst}")

def render_frame(env, writer, label, z, vz, count):
    if env.renderer is None:
        env.render(camera_name="fixed_angle_close_3d")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.core_body_id
    cam.distance = 2.40
    cam.elevation = -22.0
    cam.azimuth = 135.0
    env.renderer.update_scene(env.data, camera=cam)
    frame = env.renderer.render()

    ann_frame = annotate(
        frame,
        "Vertical Cylinder Helical Spiral (گشتاور)",
        [
            f"Phase: {label}",
            f"Altitude Z: {z:.3f} m",
            f"Ascent Speed: {vz:+.2f} m/s",
            f"Spiral Turns: {count}",
        ],
        margin=14,
    )
    writer.append_data(ann_frame)

if __name__ == "__main__":
    main()
