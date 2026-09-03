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
from skills.locomotion import circle, move
from skills.overlay import annotate
from omegaconf import OmegaConf

def main():
    cyl_rad = 1.0  # 2.0m diameter cylinder arena
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

    # Spawn near the wall on the floor
    r_spawn = cyl_rad - 0.20
    env.data.qpos[0] = r_spawn
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Spinning Inside Vertical Cylinder (R = {cyl_rad:.2f}m) ===")

    run_dir = make_run_dir(build_run_id("vortex_drive", f"r{int(cyl_rad*100)}"))
    out_video = Path(run_dir) / "renders" / "cylinder_vortex_drive.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    step = 0
    accum_th = 0.0
    prev_th = 0.0

    while step < 1200:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        angvel = env.data.qvel[3:6].copy()
        z = float(pos[2])

        r = float(np.hypot(pos[0], pos[1]))
        theta = float(np.arctan2(pos[1], pos[0]))
        d_th = theta - prev_th
        if d_th > np.pi: d_th -= 2*np.pi
        elif d_th < -np.pi: d_th += 2*np.pi
        accum_th += abs(d_th)
        prev_th = theta

        laps = accum_th / (2 * np.pi)
        spd = float(np.hypot(vel[0], vel[1]))

        # Dynamic Circle Drive inside cylinder
        # Lookahead along circumference
        targets = circle(
            quat, env.dirs_body, env.max_extend,
            ball_xy=pos[0:2],
            center_xy=(0.0, 0.0),
            radius=cyl_rad - 0.22,
            speed=2.2,
            clockwise=False,
            radial_gain=2.5,
        )

        env.step(targets)

        if step % 4 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")

            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.core_body_id
            cam.distance = 2.80
            cam.elevation = -28.0
            cam.azimuth = float(np.degrees(theta)) + 110.0
            env.renderer.update_scene(env.data, camera=cam)
            frame = env.renderer.render()

            ann_frame = annotate(
                frame,
                "Cylinder Vortex Torque (گشتاور)",
                [
                    f"Laps Completed: {laps:.2f}",
                    f"Speed: {spd:.2f} m/s",
                    f"Orbit Radius: {r:.2f} m",
                    f"Centrifugal a_c: {(spd**2)/max(r,0.1):.1f} m/s²",
                ],
                margin=14,
            )
            writer.append_data(ann_frame)

        if step % 50 == 0:
            print(f"Step {step:4d}: Laps={laps:.2f} | Speed={spd:.2f}m/s | r={r:.3f}m | z={z:.3f}m")

        step += 1

    writer.close()
    env.close()

    print(f"\nFinished! Laps: {laps:.2f} | Final Speed: {spd:.2f} m/s")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/cylinder_vortex_drive.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

if __name__ == "__main__":
    main()
