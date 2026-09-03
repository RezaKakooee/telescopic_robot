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
from skills.locomotion import circle

def main():
    floor_r = 1.0
    wall_r = 1.6
    apron_h = 0.6
    total_h = 4.0

    cfg = load_config("configs/rl/chimney.yaml")
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
    env.data.qpos[0] = 0.65
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Motordrome Circle Progression to Vertical Wall of Death ===")

    run_dir = make_run_dir(build_run_id("motordrome", "circle_progression"))
    out_video = Path(run_dir) / "renders" / "motordrome_circle_progression.mp4"
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

        # Dynamic Radius Progression:
        # Step 0-250: Spin up on flat bottom floor (r = 0.65m)
        # Step 250-700: Climb up 45-deg apron ramp (r: 0.65 -> 1.42m)
        # Step 700+: Ride vertical wooden wall of death (r = 1.45m)
        if step < 250:
            target_r = 0.65
            phase_label = "Floor Spin-Up (Torque)"
            cmd_spd = 2.0
            cmd_gain = 2.4
        elif step < 750:
            prog = (step - 250) / 500.0
            target_r = 0.65 + prog * (wall_r - 0.18 - 0.65)
            phase_label = "45° Apron Ramp Climb"
            cmd_spd = 2.6
            cmd_gain = 3.2
        else:
            target_r = wall_r - 0.16
            phase_label = "Vertical Wall of Death Orbit"
            cmd_spd = 2.8
            cmd_gain = 3.6

        targets = circle(
            quat, env.dirs_body, env.max_extend,
            ball_xy=pos[:2],
            center_xy=(0.0, 0.0),
            radius=target_r,
            speed=cmd_spd,
            clockwise=False,
            radial_gain=2.4,
            back_gain=cmd_gain,
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

    print(f"\nFinished! Start: r=0.65m -> Final: r={r:.2f}m, z={z:.3f}m | Peak Z: {max_z:.3f}m | Laps: {laps:.2f}")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/motordrome_circle_progression.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")

if __name__ == "__main__":
    main()
