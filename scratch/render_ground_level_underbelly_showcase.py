"""Render Ground-Level Low-Angle Video Showcasing Underbelly Rods Touching the Ground.

Features:
1. Ground-Level Lateral Profile ('underbelly_side_low', elevation=-4°):
   Direct horizontal view of the bottom-most rods actively touching the floor & rocks.
2. Low-Angle Rear Chase ('underbelly_rear_low', elevation=-4°):
   Looking up underneath the sphere to observe how bottom support rods and rear pushers interact with the rock bed.
3. 25-Meter Full Expedition with 900 Boulders (40+ seconds duration).
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


def render_ground_level_underbelly_showcase():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__ground_level_underbelly_showcase")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering Ground-Level Underbelly Contact Showcase -> {out_dir} ===")

    cfg = load_config("configs/rl/rocky_mountain_terrain.yaml")
    cfg.controller.enable_underbelly_contact = True
    cfg.controller.enable_adaptive_grouping = False
    cfg.controller.enable_gaussian_stance = False
    cfg.controller.underbelly_stance_gain = 0.45
    cfg.controller.underbelly_threshold_z = -0.20
    cfg.controller.back_gain = 2.0

    env = MujocoRadialSphereEnv(cfg, max_steps=3500)
    obs, info = env.reset(seed=42)

    v_side_low = out_dir / "underbelly_ground_level_side.mp4"
    v_rear_low = out_dir / "underbelly_ground_level_rear.mp4"
    v_dual_low = out_dir / "underbelly_ground_level_dual_composite.mp4"

    w_side = imageio.get_writer(str(v_side_low), fps=24, codec="libx264")
    w_rear = imageio.get_writer(str(v_rear_low), fps=24, codec="libx264")
    w_dual = imageio.get_writer(str(v_dual_low), fps=24, codec="libx264")

    print(f"Course: {env.scenario.path_length:.2f} meters with {env.scenario.stones[0][4]} rocks")
    print(f"Camera Angles: Ground-Level Low-Angle Lateral & Rear Chase (elevation=-4.0°)")

    ctrl = env.cfg.controller
    step = 0
    frames_dual = []

    print("\nSimulating with ground-level underbelly cameras...")
    while True:
        ball_pos = env.data.qpos[0:3]
        quat = env.data.qpos[3:7]

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
            underbelly_stance_gain=float(getattr(ctrl, "underbelly_stance_gain", 0.45)),
            underbelly_threshold_z=float(getattr(ctrl, "underbelly_threshold_z", -0.20)),
        )

        obs, rew, term, trunc, info = env.step(targets)
        step += 1

        f_side = env.render(camera_name="underbelly_side_low")
        f_rear = env.render(camera_name="underbelly_rear_low")
        f_dual = np.concatenate([f_side, f_rear], axis=1)

        w_side.append_data(f_side)
        w_rear.append_data(f_rear)
        w_dual.append_data(f_dual)
        frames_dual.append(f_dual)

        if step % 100 == 0:
            print(f"  Step {step:4d}: Ball pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.3f}m), Dist to Goal={info['distance']:.2f}m")

        if term or trunc or info["distance"] < 0.45:
            print(f"\n🎉 Goal Touchdown Achieved at Step {step}! Final distance={info['distance']:.3f}m")
            break

        if step >= 1100:
            break

    w_side.close()
    w_rear.close()
    w_dual.close()
    env.close()

    duration_sec = step / 24.0
    print(f"\nGround-Level Underbelly Showcase Render Complete!")
    print(f"  - Total Frames: {step} frames")
    print(f"  - Video Duration: {duration_sec:.1f} seconds")
    print(f"  - Ground-Level Side Video: {v_side_low}")
    print(f"  - Ground-Level Rear Chase Video: {v_rear_low}")
    print(f"  - Dual Ground-Level Composite Video: {v_dual_low}")

    if len(frames_dual) > 100:
        preview_path = Path("docs/project_journey/assets/ground_level_underbelly_dual_preview.png")
        Image.fromarray(frames_dual[len(frames_dual) // 2]).save(preview_path)
        print(f"Saved Ground-Level Preview still -> {preview_path}")


if __name__ == "__main__":
    render_ground_level_underbelly_showcase()
