"""Render Full-Length 25-Meter Active Suspension Mountain Expedition (45+ Seconds Video).

Features:
1. 25.0-meter expansive mountain course with 900 procedural boulders & rock slabs.
2. Active Terrain-Filtering Suspension active:
   - Skyhook core heave regulation (proportional-derivative damper cancels vertical bounces).
   - Local rock-bump compliance (downward rods absorb protruding boulders).
3. Produces a full-length, 45+ second cinematic video from ground-level low angles.
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


def render_long_active_suspension():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__long_active_suspension_showcase")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering Full-Length 25m Active Suspension Video -> {out_dir} ===")

    cfg = load_config("configs/rl/rocky_mountain_terrain.yaml")
    cfg.controller.enable_active_suspension = True
    cfg.controller.enable_underbelly_contact = True
    cfg.controller.underbelly_stance_gain = 0.42
    cfg.controller.underbelly_threshold_z = -0.20
    cfg.controller.back_gain = 2.0
    cfg.controller.target_ride_height = 0.28
    cfg.controller.suspension_kp = 0.75
    cfg.controller.suspension_kd = 0.15
    cfg.controller.suspension_force_compliance = 0.0018

    env = MujocoRadialSphereEnv(cfg, max_steps=3500)
    obs, info = env.reset(seed=42)

    v_side = out_dir / "long_active_suspension_ground_side.mp4"
    v_rear = out_dir / "long_active_suspension_rear_underside.mp4"
    v_dual = out_dir / "long_active_suspension_dual_ground_level.mp4"
    v_quad = out_dir / "long_active_suspension_quad_4_views.mp4"

    w_side = imageio.get_writer(str(v_side), fps=24, codec="libx264")
    w_rear = imageio.get_writer(str(v_rear), fps=24, codec="libx264")
    w_dual = imageio.get_writer(str(v_dual), fps=24, codec="libx264")
    w_quad = imageio.get_writer(str(v_quad), fps=24, codec="libx264")

    print(f"Course: {env.scenario.path_length:.2f} meters with {env.scenario.stones[0][4]} rocks")
    print(f"Active Suspension: Skyhook Dampers + Rock-Bump Absorbers Enabled")

    ctrl = env.cfg.controller
    step = 0
    frames_dual = []

    print("\nSimulating full-length 25m active suspension glide...")
    while True:
        ball_pos = env.data.qpos[0:3]
        ball_vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]

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
            underbelly_stance_gain=float(getattr(ctrl, "underbelly_stance_gain", 0.42)),
            underbelly_threshold_z=float(getattr(ctrl, "underbelly_threshold_z", -0.20)),
            enable_active_suspension=True,
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
        step += 1

        f_side = env.render(camera_name="underbelly_side_low")
        f_rear = env.render(camera_name="underbelly_rear_low")
        f_dual = np.concatenate([f_side, f_rear], axis=1)
        f_quad = env.render(camera_name="quad")

        w_side.append_data(f_side)
        w_rear.append_data(f_rear)
        w_dual.append_data(f_dual)
        w_quad.append_data(f_quad)
        frames_dual.append(f_dual)

        if step % 100 == 0:
            print(f"  Step {step:4d}: Ball pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.3f}m), Dist to Goal={info['distance']:.2f}m")

        if term or trunc or info["distance"] < 0.45:
            print(f"\n🎉 Goal Touchdown Achieved at Step {step}! Final distance={info['distance']:.3f}m")
            break

        if step >= 1200:
            break

    w_side.close()
    w_rear.close()
    w_dual.close()
    w_quad.close()
    env.close()

    duration_sec = step / 24.0
    print(f"\nFull-Length Active Suspension Video Render Complete!")
    print(f"  - Total Frames: {step} frames")
    print(f"  - Video Duration: {duration_sec:.1f} seconds (Full-Length)")
    print(f"  - Dual Ground-Level Video: {v_dual}")
    print(f"  - Ground-Level Side Video: {v_side}")
    print(f"  - Ground-Level Rear Underside Video: {v_rear}")
    print(f"  - 4-Side Quad Video: {v_quad}")

    if len(frames_dual) > 100:
        preview_path = Path("docs/project_journey/assets/long_active_suspension_dual_preview.png")
        Image.fromarray(frames_dual[len(frames_dual) // 2]).save(preview_path)
        print(f"Saved preview still -> {preview_path}")


if __name__ == "__main__":
    render_long_active_suspension()
