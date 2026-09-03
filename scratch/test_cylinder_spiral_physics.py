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

def run_spiral_test(cylinder_rad=0.32, pitch_deg=30.0):
    cfg = load_config("configs/rl/chimney.yaml")
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "scenario") or cfg.scenario is None:
        cfg.scenario = {}
    cfg.scenario.cylinder_radius = cylinder_rad
    cfg.scenario.cylinder_height = 4.0

    scenario = generate_scenario("vertical_cylinder", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    # Spawn in center of cylinder at z = 0.22m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Testing 360° Helical Screw Climbing (R = {cylinder_rad:.2f}m, Pitch = {pitch_deg:.1f}°) ===")

    run_dir = make_run_dir(build_run_id("cylinder_screw", f"r{int(cylinder_rad*100)}_p{int(pitch_deg)}"))
    out_video = Path(run_dir) / "renders" / "vertical_cylinder_screw_climb.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    # Time-dependent helical wave angle phi(t) = omega * t
    omega_spin = 12.0  # rad/s spin frequency
    pitch_rad = np.radians(pitch_deg)

    step = 0
    max_z = 0.22

    while step < 1500:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        z = float(pos[2])
        max_z = max(max_z, z)

        t = step * 0.004
        phi_wave = omega_spin * t

        from radial_sphere.geometry import quat_to_rotmat
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        # Radial angle in horizontal plane for each rod
        rod_theta = np.arctan2(dirs_world[:, 1], dirs_world[:, 0])
        rod_uz = dirs_world[:, 2]
        rod_ulat = np.hypot(dirs_world[:, 0], dirs_world[:, 1])

        # Helical phase for each rod: phase = rod_theta + (rod_uz / tan(pitch)) - phi_wave
        # A helical wave moving upward:
        phase = rod_theta - (rod_uz * 2.5) - phi_wave
        # Continuous sinusoidal push wave with cosine shaping
        wave = 0.5 * (1.0 + np.cos(phase))

        # Radial pre-extension: all lateral rods push into cylinder walls (r = 0.32m -> extension ~0.17m)
        targets = np.full(60, 0.02, dtype=np.float32)
        # Lateral rods touching cylinder wall:
        lat_mask = rod_ulat > 0.40
        # Blend stance extension + active traveling wave
        targets[lat_mask] = 0.08 + 0.08 * wave[lat_mask]

        env.step(targets)

        if step % 4 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")

            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.core_body_id
            cam.distance = 2.20
            cam.elevation = -18.0
            cam.azimuth = 145.0
            env.renderer.update_scene(env.data, camera=cam)
            frame = env.renderer.render()

            ann_frame = annotate(
                frame,
                "Vertical Cylinder Helical Vortex (گشتاور)",
                [
                    f"Height Z: {z:.3f} m",
                    f"Vertical Velocity: {vel[2]:+.2f} m/s",
                    f"Spin Freq: {omega_spin:.1f} rad/s",
                    f"Cylinder Radius: {cylinder_rad:.2f} m",
                ],
                margin=14,
            )
            writer.append_data(ann_frame)

        if step % 50 == 0:
            print(f"Step {step:4d}: z={z:.3f}m | vz={vel[2]:+.2f}m/s | max_z={max_z:.3f}m")

        if z >= 3.50:
            print(f"Reached top! Z = {z:.3f}m")
            break

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished! Start Z: 0.22m -> Final Z: {z:.3f}m | Peak: {max_z:.3f}m")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/vertical_cylinder_screw_climb.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

if __name__ == "__main__":
    run_spiral_test()
