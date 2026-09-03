"""Vertical Cylinder Helical Vortex Climbing (گشتاور / Spiral Climb).

The ball spins up inside an empty vertical transparent glass cylinder (R = 0.55m, H = 4.0m),
develops rotational momentum (گشتاور) and centrifugal force, and ascends via an upward helical spiral.

Usage:
    python scripts/skills/run_vertical_cylinder.py --video
    python scripts/skills/run_vertical_cylinder.py --pitch 28 --target-z 3.5
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

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir
from skills.interaction import cylinder_spiral_climb
from skills.overlay import annotate


from omegaconf import OmegaConf

def run_vertical_cylinder(
    target_z: float = 3.50,
    pitch_angle: float = 28.0,
    seed: int = 42,
    record_video: bool = True,
    max_steps: int = 2500,
):
    in_rad = 0.45
    cfg = load_config("configs/rl/chimney.yaml")
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "scenario") or cfg.scenario is None:
        cfg.scenario = {}
    cfg.scenario.cylinder_radius = in_rad
    cfg.scenario.cylinder_height = 4.0

    scenario = generate_scenario("vertical_cylinder", cfg, seed=seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=seed)

    # Spawn at base near the cylinder wall
    env.data.qpos[0] = in_rad - 0.16
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Starting Helical Cylinder Climb (Target Z: {target_z:.2f}m, Pitch: {pitch_angle:.1f}°) ===")

    writer = None
    out_video = None
    if record_video:
        run_dir = make_run_dir(build_run_id("cylinder_climb", f"helical_pitch{int(pitch_angle)}"))
        out_video = Path(run_dir) / "renders" / "vertical_cylinder_spiral_climb.mp4"
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    # Initial settling steps
    for _ in range(20):
        env.step(np.full(env.n_bars, 0.03, dtype=np.float32))

    step = 0
    peak_z = 0.22
    reached = False

    while step < max_steps:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        angvel = env.data.qvel[3:6].copy()

        z = float(pos[2])
        peak_z = max(peak_z, z)

        # Angular position and velocity around cylinder axis
        r = float(np.hypot(pos[0], pos[1]))
        theta = float(np.arctan2(pos[1], pos[0]))
        omega_z = float(angvel[2])
        v_tan = float(-vel[0] * np.sin(theta) + vel[1] * np.cos(theta))

        # Two-stage control:
        # 1. Spin-Up (first 100 steps): 10 deg pitch to build high tangential speed / torque
        # 2. Helical Climb: ramp pitch up to full pitch_angle to convert speed into altitude!
        if step < 100:
            active_pitch = 8.0 + (step / 100.0) * (pitch_angle - 8.0)
            phase_label = "Spin-Up Torque (گشتاور)"
            spd = 2.0
            radial_gain = 0.50
        else:
            active_pitch = pitch_angle
            phase_label = "Helical Vortex Climb"
            spd = 2.5
            radial_gain = 0.45

        targets = cylinder_spiral_climb(
            quat, env.dirs_body, env.max_extend, pos, vel,
            center_xy=(0.0, 0.0), cylinder_radius=in_rad,
            direction=+1, pitch_angle_deg=active_pitch,
            speed=spd, radial_brace_gain=radial_gain
        )

        env.step(targets)

        if z >= target_z:
            print(f"Goal height reached at step {step}! Z = {z:.3f}m")
            reached = True
            # Hold at top for 30 steps
            for _ in range(40):
                targets_top = cylinder_spiral_climb(
                    env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                    env.data.qpos[:3].copy(), pitch_angle_deg=0.0,
                    radial_brace_gain=0.80
                )
                env.step(targets_top)
                if writer is not None and step % 4 == 0:
                    render_frame(env, writer, "Apex Orbit / Hold", z, v_tan, active_pitch, r)
                step += 1
            break

        if writer is not None and step % 4 == 0:
            render_frame(env, writer, phase_label, z, v_tan, active_pitch, r)

        if step % 50 == 0:
            print(f"Step {step:4d} [{phase_label:22s}]: z={z:.3f}m | v_tan={v_tan:+.2f}m/s | vz={vel[2]:+.2f}m/s | r={r:.3f}m")

        step += 1

    if writer is not None:
        writer.close()
        print(f"Video saved to: {out_video}")
        artifact_path = Path(f"/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/vertical_cylinder_spiral_climb.mp4")
        import shutil
        shutil.copyfile(str(out_video), str(artifact_path))
        print(f"Copied to artifact path: {artifact_path}")

    env.close()

    print(f"\n--- Climb Summary ---")
    print(f"Initial Z: 0.22m -> Peak Z: {peak_z:.3f}m (Net Climb: +{peak_z - 0.22:.2f}m)")
    print(f"Target Reached: {reached}")

    return {
        "reached": reached,
        "peak_z": peak_z,
        "video_path": str(out_video) if out_video else None
    }


def render_frame(env, writer, phase_label, z, v_tan, pitch_deg, r):
    if env.renderer is None:
        env.render(camera_name="fixed_angle_close_3d")

    # Camera 1: Elevated isometric 3D view looking down through the transparent silo
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
        "Vertical Cylinder Helical Vortex (گشتاور)",
        [
            f"Phase: {phase_label}",
            f"Altitude Z: {z:.3f} m",
            f"Tangential Speed: {v_tan:+.2f} m/s",
            f"Pitch Angle: {pitch_deg:.1f}° | Radius: {r:.2f}m",
        ],
        margin=14,
    )
    writer.append_data(ann_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-z", type=float, default=3.50)
    parser.add_argument("--pitch", type=float, default=26.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", default=True)
    args = parser.parse_args()

    run_vertical_cylinder(
        target_z=args.target_z,
        pitch_angle=args.pitch,
        seed=args.seed,
        record_video=args.video
    )
