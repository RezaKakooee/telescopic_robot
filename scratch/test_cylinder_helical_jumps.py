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

    # Spawn resting at base
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Testing Helical Spiral Ascent in Vertical Cylinder (R = {cyl_rad:.2f}m) ===")

    run_dir = make_run_dir(build_run_id("cylinder_spiral", "helical_jumps"))
    out_video = Path(run_dir) / "renders" / "vertical_cylinder_helical_spiral_climb.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    step = 0
    state = "launch"
    timer = 0
    max_z = 0.22
    jump_count = 0
    target_angle = 0.0

    while step < 1800:
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

        if state == "launch":
            # Explosive upward-forward launch off floor
            # Ground rods fire
            targets[dirs_world[:, 2] < -0.20] = env.max_extend
            phase_label = "Initial Ground Launch"
            if timer >= 14:
                state = "fly"
                timer = 0
                jump_count += 1

        elif state == "fly":
            # Flight through air in spiral chord: tuck all rods
            targets[:] = 0.010
            phase_label = f"Spiral Flight #{jump_count}"
            # At apex (vz <= 0.15) or when near wall, catch
            if timer >= 18 and vel[2] < 0.20:
                state = "catch"
                timer = 0

        elif state == "catch":
            # Lock onto curved wall with 360 lateral clamp
            lat_mask = np.hypot(dirs_world[:, 0], dirs_world[:, 1]) > 0.65
            targets[lat_mask] = 0.12
            phase_label = f"Wall Catch #{jump_count}"
            if timer >= 16:
                state = "spiral_push"
                timer = 0

        elif state == "spiral_push":
            # Fire diagonal downward-tangential push wave off the wall
            # Contact wall normal points toward center (-r_hat)
            r_hat = np.array([np.cos(theta), np.sin(theta), 0.0]) if r > 1e-4 else np.array([1, 0, 0])
            theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])

            # Helical push vector: pushes into the wall (+r_hat) + backward (-theta_hat) + downward (-z)
            push_vec = 0.60 * r_hat - 0.50 * theta_hat - 0.60 * np.array([0, 0, 1])
            push_vec = push_vec / np.linalg.norm(push_vec)

            p = dirs_world @ push_vec
            targets[p > 0.40] = env.max_extend
            phase_label = f"Helical Shove (گشتاور) #{jump_count}"
            if timer >= 12:
                state = "fly"
                timer = 0
                jump_count += 1

        env.step(targets)

        if step % 4 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")

            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.core_body_id
            cam.distance = 2.40
            cam.elevation = -20.0
            cam.azimuth = float(np.degrees(theta)) + 125.0
            env.renderer.update_scene(env.data, camera=cam)
            frame = env.renderer.render()

            ann_frame = annotate(
                frame,
                "Vertical Cylinder Helical Spiral (گشتاور)",
                [
                    f"Phase: {phase_label}",
                    f"Altitude Z: {z:.3f} m (Peak: {max_z:.2f}m)",
                    f"Velocity vz: {vel[2]:+.2f} m/s",
                    f"Helical Jumps: {jump_count}",
                ],
                margin=14,
            )
            writer.append_data(ann_frame)

        if step % 30 == 0:
            print(f"Step {step:4d} [{phase_label:26s}]: z={z:.3f}m | vz={vel[2]:+.2f}m/s | peak={max_z:.3f}m")

        if z >= 3.50:
            print(f"Reached top rim! Z = {z:.3f}m")
            break

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished Helical Climb! Start Z: 0.22m -> Peak Z: {max_z:.3f}m in {jump_count} spiral pushes!")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/vertical_cylinder_helical_spiral_climb.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

if __name__ == "__main__":
    main()
