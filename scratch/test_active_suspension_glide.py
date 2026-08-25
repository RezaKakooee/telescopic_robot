"""Evaluation of Active Terrain-Canceling Suspension Mechanism.

Compares:
1. Without Active Suspension (Rigid Stance): Core bumps, tilts, and jumps across rocks.
2. With Active Terrain Suspension (Skyhook Heave Canceling + Rock-Bump Compliance):
   Underbelly rods dynamically absorb jagged rock peaks and dips so the ball core
   glides horizontally at constant altitude like rolling on a flat floor.
"""
import datetime
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.controller import bar_targets, desired_direction


def evaluate_suspension(enable_suspension: bool, out_dir: Path, label: str):
    print(f"\n=== Evaluating: {label} (enable_active_suspension={enable_suspension}) ===")
    cfg = load_config("configs/rl/rocky_mountain_terrain.yaml")
    cfg.controller.enable_active_suspension = enable_suspension
    cfg.controller.enable_underbelly_contact = True
    cfg.controller.underbelly_stance_gain = 0.40
    cfg.controller.underbelly_threshold_z = -0.20
    cfg.controller.back_gain = 2.0

    env = MujocoRadialSphereEnv(cfg, max_steps=1500)
    obs, info = env.reset(seed=42)

    v_side = out_dir / f"suspension_{label}_ground_side.mp4"
    v_dual = out_dir / f"suspension_{label}_dual_low_angle.mp4"

    w_side = imageio.get_writer(str(v_side), fps=24, codec="libx264")
    w_dual = imageio.get_writer(str(v_dual), fps=24, codec="libx264")

    ctrl = env.cfg.controller
    z_history = []
    vz_history = []
    frames_dual = []

    for step in range(500):
        ball_pos = env.data.qpos[0:3]
        ball_vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]

        z_history.append(ball_pos[2])
        vz_history.append(ball_vel[2])

        forces = None
        if hasattr(env, "get_rod_contact_forces"):
            forces = env.get_rod_contact_forces()

        d_hat, drive = desired_direction(ball_pos[:2], env.path_pts, lookahead=float(ctrl.lookahead))
        targets = bar_targets(
            quat,
            env.dirs_body,
            env.max_extend,
            d_hat,
            drive=drive,
            min_offset=float(ctrl.base),
            back_gain=float(ctrl.back_gain),
            enable_curb_vaulting=bool(getattr(ctrl, "enable_curb_vaulting", True)),
            curb_boost_gain=float(getattr(ctrl, "curb_boost_gain", 2.6)),
            enable_underbelly_contact=True,
            underbelly_stance_gain=float(getattr(ctrl, "underbelly_stance_gain", 0.40)),
            underbelly_threshold_z=float(getattr(ctrl, "underbelly_threshold_z", -0.20)),
            enable_active_suspension=enable_suspension,
            core_z=float(ball_pos[2]),
            core_vz=float(ball_vel[2]),
            target_ride_height=float(getattr(ctrl, "target_ride_height", 0.28)),
            suspension_kp=float(getattr(ctrl, "suspension_kp", 0.75)),
            suspension_kd=float(getattr(ctrl, "suspension_kd", 0.15)),
            suspension_force_compliance=float(getattr(ctrl, "suspension_force_compliance", 0.0018)),
            nominal_support_force=float(getattr(ctrl, "nominal_support_force", 10.0)),
            contact_forces=forces,
        )

        obs, rew, term, trunc, info = env.step(targets)

        f_side = env.render(camera_name="underbelly_side_low")
        f_rear = env.render(camera_name="underbelly_rear_low")
        f_dual = np.concatenate([f_side, f_rear], axis=1)

        w_side.append_data(f_side)
        w_dual.append_data(f_dual)
        frames_dual.append(f_dual)

        if term or trunc or info["distance"] < 0.45:
            print(f"Goal reached at step {step + 1}!")
            break

    w_side.close()
    w_dual.close()
    env.close()

    z_arr = np.array(z_history)
    vz_arr = np.array(vz_history)
    z_std = float(np.std(z_arr))
    z_range = float(np.max(z_arr) - np.min(z_arr))
    vz_max = float(np.max(np.abs(vz_arr)))

    print(f"Results for {label}:")
    print(f"  - Core Altitude Standard Deviation (Bounce): {z_std * 100:.2f} cm")
    print(f"  - Total Peak-to-Peak Altitude Variance: {z_range * 100:.2f} cm")
    print(f"  - Peak Vertical Heave Velocity: {vz_max:.3f} m/s")
    print(f"  - Video (Ground Side): {v_side}")
    print(f"  - Video (Dual Low Angle): {v_dual}")

    return {
        "frames_dual": frames_dual,
        "z_std": z_std,
        "z_range": z_range,
        "vz_max": vz_max,
        "video_side": v_side,
        "video_dual": v_dual,
    }


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__active_suspension_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Testing Active Terrain-Canceling Suspension -> {out_dir} ===")

    # 1. Passive / Without Active Suspension
    res_passive = evaluate_suspension(False, out_dir, "without_suspension")

    # 2. With Active Terrain-Filtering Suspension
    res_active = evaluate_suspension(True, out_dir, "with_active_suspension")

    # Extract comparison stills
    if len(res_active["frames_dual"]) > 50:
        f_pass = res_passive["frames_dual"][50]
        f_act = res_active["frames_dual"][50]
        comp = np.concatenate([f_pass, f_act], axis=0) # Stack top/bottom
        Image.fromarray(comp).save("docs/project_journey/assets/active_suspension_flat_glide_comparison.png")
        print("\nSaved suspension comparison image in docs/project_journey/assets/!")


if __name__ == "__main__":
    main()
