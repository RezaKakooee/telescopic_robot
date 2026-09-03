"""Smooth controlled chimney descent under free physics.

Tests both active friction-servo rappel and smooth alternating stepping descent
from high elevation (z = 3.50m) all the way down to a soft landing on the floor (z = 0.22m).

Usage:
    python scripts/skills/run_chimney_descend.py --mode servo --v-target -0.50 --video
    python scripts/skills/run_chimney_descend.py --mode step --video
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
from skills.interaction import chimney_friction_servo, chimney_step_down
from skills.overlay import annotate

AXIS = np.array([0.0, 1.0])


def run_descent(
    mode: str = "servo",
    v_target: float = -0.50,
    start_z: float = 3.50,
    seed: int = 42,
    record_video: bool = True,
):
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = generate_scenario("chimney", cfg, seed=seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=seed)

    # Spawn high in the chimney clamped between walls
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = start_z
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Starting Smooth Chimney Descent (Mode: {mode.upper()}, Start Z: {start_z:.2f}m) ===")

    writer = None
    out_video = None
    if record_video:
        run_dir = make_run_dir(build_run_id("chimney_descend", f"{mode}_v{abs(v_target):.2f}"))
        out_video = Path(run_dir) / "renders" / f"smooth_chimney_descend_{mode}.mp4"
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    # Initial clamp hold to settle
    print("Holding initial stance...")
    integral_err = 0.0
    for _ in range(30):
        quat = env.data.qpos[3:7].copy()
        targets, _, _ = chimney_friction_servo(
            quat, env.dirs_body, env.max_extend, v_z=0.0,
            v_target=0.0, clamp_ext_nominal=0.15, wall_axis=AXIS
        )
        env.step(targets)

    step = 0
    landed = False
    land_step = None
    step_side = +1
    step_timer = 0
    step_period = 30  # steps per stepping phase

    # Data collection for analysis
    z_history = []
    vz_history = []
    clamp_history = []

    while step < 2000:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        z = float(pos[2])
        vz = float(vel[2])

        z_history.append(z)
        vz_history.append(vz)

        near_floor = (z < 0.65)

        if z <= 0.23 and abs(vz) < 0.15:
            landed = True
            land_step = step
            print(f"Touchdown confirmed at step {step}! Final Z: {z:.3f}m | vz: {vz:+.3f}m/s")
            # Stand on floor for 30 more steps
            for _ in range(40):
                quat = env.data.qpos[3:7].copy()
                targets = np.zeros(env.n_bars, dtype=np.float32)
                targets[env.dirs_body[:, 2] < -0.30] = 0.045
                env.step(targets)
                if writer is not None and step % 4 == 0:
                    render_frame(env, writer, mode, "Landed & Standing", z, vz, 0.045)
                step += 1
            break

        if mode == "servo":
            targets, clamp_cmd, integral_err = chimney_friction_servo(
                quat, env.dirs_body, env.max_extend,
                v_z=vz, v_target=v_target,
                wall_axis=AXIS, near_floor=near_floor,
                kp=0.060, ki=0.003, integral_err=integral_err,
                clamp_ext_nominal=0.038
            )
            clamp_history.append(clamp_cmd)
            phase_label = "Landing Flare" if near_floor else "Controlled Rappel"

        else:  # "step" mode
            step_timer += 1
            if step_timer >= step_period:
                step_timer = 0
                step_side *= -1

            ratio = step_timer / float(step_period)
            targets = chimney_step_down(
                quat, env.dirs_body, env.max_extend,
                phase_ratio=ratio, side=step_side,
                wall_axis=AXIS, near_floor=near_floor,
                clamp_ext=0.080
            )
            clamp_cmd = 0.080
            clamp_history.append(clamp_cmd)
            phase_label = f"Step-Down ({'Left' if step_side > 0 else 'Right'})"

        env.step(targets)

        if writer is not None and step % 4 == 0:
            render_frame(env, writer, mode, phase_label, z, vz, clamp_cmd)

        if step % 50 == 0:
            print(f"Step {step:4d} [{phase_label:18s}]: z={z:.3f}m | vz={vz:+.2f}m/s | clamp={clamp_cmd*100:.1f}cm")

        step += 1

    if writer is not None:
        writer.close()
        print(f"Video saved to: {out_video}")
        # Copy to artifact
        artifact_path = Path(f"/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/smooth_chimney_descend_{mode}.mp4")
        import shutil
        shutil.copyfile(str(out_video), str(artifact_path))
        print(f"Copied to artifact path: {artifact_path}")

    env.close()

    mean_vz = np.mean(vz_history[30:land_step]) if land_step and land_step > 30 else np.mean(vz_history)
    print(f"\n--- Performance Summary ({mode.upper()}) ---")
    print(f"Descent Range: {start_z:.2f}m -> {z:.2f}m (Net: {start_z - z:.2f}m)")
    print(f"Mean Descent Velocity: {mean_vz:+.3f} m/s (Target: {v_target:+.2f} m/s)")
    print(f"Touchdown Velocity: {vz_history[-1]:+.3f} m/s | Soft Landing: {abs(vz_history[-1]) < 0.3}")

    return {
        "mode": mode,
        "landed": landed,
        "mean_vz": mean_vz,
        "final_z": z,
        "video_path": str(out_video) if out_video else None
    }


def render_frame(env, writer, mode, phase_label, z, vz, clamp_cmd):
    if env.renderer is None:
        env.render(camera_name="fixed_angle_close_3d")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.core_body_id
    cam.distance = 1.60
    cam.elevation = -8.0
    cam.azimuth = 180.0
    env.renderer.update_scene(env.data, camera=cam)
    frame = env.renderer.render()

    ann_frame = annotate(
        frame,
        f"Smooth Chimney Descent ({mode.upper()})",
        [
            f"Phase: {phase_label}",
            f"Z Height: {z:.3f} m",
            f"Descent Rate: {vz:+.2f} m/s",
            f"Clamp Extension: {clamp_cmd*100:.1f} cm",
        ],
        margin=14,
    )
    writer.append_data(ann_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["servo", "step"], default="servo")
    parser.add_argument("--v-target", type=float, default=-0.50)
    parser.add_argument("--start-z", type=float, default=3.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", default=True)
    args = parser.parse_args()

    run_descent(
        mode=args.mode,
        v_target=args.v_target,
        start_z=args.start_z,
        seed=args.seed,
        record_video=args.video
    )
