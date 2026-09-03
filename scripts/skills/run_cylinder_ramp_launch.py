"""High-Speed Acceleration Launch Ramp & Vertical Cylinder Spiral Climbing.

The ball starts at the top of an acceleration incline ramp (z = 1.30m),
accelerates down the slope to build high speed (v > 3.0 m/s) and rotational torque,
shoots tangentially into the vertical cylinder, and rides the vertical wall
in a centrifugal helical spiral (Wall of Death).

Usage:
    python scripts/skills/run_cylinder_ramp_launch.py --video
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir
from skills.locomotion import move
from skills.interaction import cylinder_spiral_climb
from skills.overlay import annotate


def run_ramp_launch(
    target_z: float = 3.50,
    seed: int = 42,
    record_video: bool = True,
    max_steps: int = 1500,
):
    cyl_rad = 0.85
    cyl_height = 4.0

    cfg = load_config("configs/rl/chimney.yaml")
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "scenario") or cfg.scenario is None:
        cfg.scenario = {}
    cfg.scenario.cylinder_radius = cyl_rad
    cfg.scenario.cylinder_height = cyl_height

    scenario = generate_scenario("vertical_cylinder", cfg, seed=seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=seed)

    # Spawn at top of launch ramp
    y_target = -cyl_rad + 0.08
    env.data.qpos[0] = -3.80
    env.data.qpos[1] = y_target
    env.data.qpos[2] = 1.35
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Acceleration Launch Ramp & Vertical Cylinder (R = {cyl_rad:.2f}m) ===")

    run_dir = make_run_dir(build_run_id("ramp_launch", "wall_of_death"))
    out_video = Path(run_dir) / "renders" / "cylinder_ramp_launch_spiral.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    step = 0
    max_z = 0.0
    entered_cylinder = False
    reached = False
    accum_th = 0.0
    prev_th = -np.pi / 2.0

    while step < max_steps:
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

        if not entered_cylinder and x < -0.10:
            # Stage 1: Downhill Power Sprint with Centerline Steering
            phase_label = "Ramp Acceleration Sprint"
            y_err = y - y_target
            d_cmd = np.array([1.0, -2.5 * y_err])
            d_cmd /= np.linalg.norm(d_cmd)

            targets = move(
                quat, env.dirs_body, env.max_extend,
                d_hat=d_cmd,
                speed=3.0,
                back_gain=3.6,
                min_offset=0.025
            )

        else:
            if not entered_cylinder:
                entered_cylinder = True
                prev_th = theta
                print(f"⚡ Cylinder Entrance at step {step}: Speed = {spd:.2f} m/s | Z = {z:.3f}m | Centrifugal a_c = {(spd**2)/r:.1f} m/s² ({(spd**2)/(r*9.81):.2f} G)")

            # Stage 2: Wall of Death Centrifugal Spiral Climbing
            phase_label = "Wall of Death Spiral Ascent"

            # Pure-Pursuit Target ahead on the circle
            lookahead_angle = 0.35
            th_target = theta + lookahead_angle
            r_target = cyl_rad - 0.16
            target_pt = np.array([r_target * np.cos(th_target), r_target * np.sin(th_target)])

            # Heading vector towards circular lookahead target
            heading_xy = target_pt - pos[:2]
            heading_xy /= max(np.linalg.norm(heading_xy), 1e-6)

            targets = move(
                quat, env.dirs_body, env.max_extend,
                d_hat=heading_xy,
                speed=3.0,
                back_gain=3.8,
                min_offset=0.020
            )

        env.step(targets)

        # Centrifugal acceleration
        a_centrifugal = (spd ** 2) / max(r, 0.1) if entered_cylinder else 0.0

        if z >= target_z and entered_cylinder:
            print(f"Goal height reached! Z = {z:.3f}m at step {step}")
            reached = True
            break

        if writer is not None and step % 4 == 0:
            render_frame(env, writer, phase_label, x, y, z, spd, a_centrifugal, entered_cylinder)

        if step % 50 == 0:
            print(f"Step {step:4d} [{phase_label:26s}]: pos=({x:+.2f}, {y:+.2f}, {z:.2f}m) | spd={spd:.2f}m/s | a_c={a_centrifugal:.1f}m/s²")

        step += 1

    if writer is not None:
        writer.close()
        print(f"Video saved to: {out_video}")
        dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/cylinder_ramp_launch_spiral.mp4")
        import shutil
        shutil.copyfile(str(out_video), str(dst))
        print(f"Copied to artifact path: {dst}")

    env.close()

    print(f"\n--- Run Summary ---")
    print(f"Start: (-3.80m, z=1.32m) -> Entry Speed: {spd:.2f}m/s -> Peak Z: {max_z:.3f}m")
    return {
        "reached": reached,
        "entry_speed": spd,
        "peak_z": max_z,
        "video_path": str(out_video) if out_video else None
    }


def render_frame(env, writer, phase_label, x, y, z, spd, a_c, in_cyl):
    if env.renderer is None:
        env.render(camera_name="fixed_angle_close_3d")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.core_body_id
    cam.distance = 3.20
    cam.elevation = -24.0
    # Wide viewing angle capturing both the ramp and the cylinder
    cam.azimuth = 140.0 if not in_cyl else float(np.degrees(np.arctan2(y, x))) + 120.0
    env.renderer.update_scene(env.data, camera=cam)
    frame = env.renderer.render()

    ann_frame = annotate(
        frame,
        "Acceleration Launch Ramp -> Cylinder Spiral",
        [
            f"Phase: {phase_label}",
            f"Position: ({x:+.2f}, {y:+.2f}, {z:.2f}m)",
            f"Speed: {spd:.2f} m/s",
            f"Centrifugal Acc: {a_c:.1f} m/s² ({a_c/9.81:.2f} G)",
        ],
        margin=14,
    )
    writer.append_data(ann_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-z", type=float, default=3.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", default=True)
    args = parser.parse_args()

    run_ramp_launch(
        target_z=args.target_z,
        seed=args.seed,
        record_video=args.video
    )
