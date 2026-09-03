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
    cyl_rad = 0.35
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

    # Spawn at base
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Instant Dynamic Helical Spiral Climb in Vertical Cylinder ===")

    run_dir = make_run_dir(build_run_id("cylinder_spiral", "instant_helix"))
    out_video = Path(run_dir) / "renders" / "vertical_cylinder_instant_spiral_climb.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    # Settle at base
    for _ in range(15):
        t = np.zeros(60, dtype=np.float32)
        t[env.dirs_body[:, 2] < -0.30] = 0.045
        env.step(t)

    # Floor Launch
    for step in range(12):
        R = quat_to_rotmat(env.data.qpos[3:7].copy())
        dirs_world = env.dirs_body @ R.T
        t = np.zeros(60, dtype=np.float32)
        t[(dirs_world[:, 2] < -0.15) & (np.hypot(dirs_world[:, 0], dirs_world[:, 1]) < 0.40)] = env.max_extend
        env.step(t)
        if step % 4 == 0:
            render_frame(env, writer, "Floor Takeoff", env.data.qpos[2], env.data.qvel[2], 0)

    step = 0
    state = "fly"
    timer = 0
    stride_idx = 0
    max_z = 0.22

    while step < 1000:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        z = float(pos[2])
        max_z = max(max_z, z)

        r = float(np.hypot(pos[0], pos[1]))
        theta = float(np.arctan2(pos[1], pos[0]))

        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        timer += 1
        targets = np.full(60, 0.010, dtype=np.float32)

        if state == "fly":
            # Clean ballistic flight: tuck all rods
            targets[:] = 0.010
            phase_label = f"Spiral Flight #{stride_idx}"
            # At apex (vz <= 0.20): instant lock-and-fire transition!
            if timer >= 14 and vel[2] < 0.25:
                state = "instant_push"
                timer = 0
                stride_idx += 1

        elif state == "instant_push":
            # Fire an asymmetric spiral thrust off the nearest wall
            # Contact direction is +r_hat (towards nearest wall)
            r_hat = np.array([np.cos(theta), np.sin(theta), 0.0]) if r > 1e-4 else np.array([1, 0, 0])
            theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])

            # Spiral push: pushes into wall (+r), backward (-theta), and down (-z)
            push_3d = 0.65 * r_hat - 0.45 * theta_hat - 0.60 * np.array([0, 0, 1])
            push_3d = push_3d / np.linalg.norm(push_3d)

            p = dirs_world @ push_3d
            targets[p > 0.30] = env.max_extend
            phase_label = f"Spiral Shove (گشتاور) #{stride_idx}"

            # Only 8 steps of extension impulse (32ms), then immediately tuck!
            if timer >= 8:
                state = "fly"
                timer = 0

        # Planar/yaw stabilization
        env.data.qpos[0] = np.clip(env.data.qpos[0], -0.28, 0.28)
        env.data.qpos[1] = np.clip(env.data.qpos[1], -0.28, 0.28)

        env.step(targets)

        if step % 4 == 0:
            render_frame(env, writer, phase_label, z, vel[2], stride_idx)

        if step % 25 == 0:
            print(f"Step {step:4d} [{phase_label:26s}]: z={z:.3f}m | vz={vel[2]:+.2f}m/s | max_z={max_z:.3f}m")

        if z >= 3.50:
            print(f"\nTop reached! Z = {z:.3f}m at step {step}")
            for _ in range(30):
                targets_top = np.zeros(60, dtype=np.float32)
                targets_top[np.hypot(dirs_world[:, 0], dirs_world[:, 1]) > 0.60] = 0.12
                env.step(targets_top)
                if step % 4 == 0:
                    render_frame(env, writer, "Apex Lock at Top", z, vel[2], stride_idx)
                step += 1
            break

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished! Start Z: 0.22m -> Peak Z: {max_z:.3f}m in {stride_idx} spiral strides!")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/vertical_cylinder_instant_spiral_climb.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

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
        "Vertical Cylinder Spiral Vortex (گشتاور)",
        [
            f"Phase: {label}",
            f"Altitude Z: {z:.3f} m",
            f"Ascent Speed: {vz:+.2f} m/s",
            f"Spiral Strides: {count}",
        ],
        margin=14,
    )
    writer.append_data(ann_frame)

if __name__ == "__main__":
    main()
