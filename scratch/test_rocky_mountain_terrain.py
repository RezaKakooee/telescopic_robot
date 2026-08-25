"""Evaluation and Multi-Camera Recording Suite for Rugged Mountainous Rocky Floor Terrain.

Tests & Validates:
1. 260+ Procedural granite, slate, basalt, and sandstone boulders & jagged slabs on the floor.
2. The 60-bar radial sphere conforming its viscoelastic rubber feet and active suspension across uneven rocks.
3. Multi-camera video suite (Dual map/chase, 4 Side tracking cameras 2x2 Quad, Fixed edge cameras).
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


def test_rocky_terrain_navigation():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__rocky_mountain_terrain_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Testing Rugged Mountainous Rocky Floor Terrain -> {out_dir} ===")

    cfg = load_config("configs/rl/rocky_mountain_terrain.yaml")
    env = MujocoRadialSphereEnv(cfg, max_steps=1200)
    obs, info = env.reset(seed=42)

    video_dual_path = out_dir / "rocky_terrain_dual_view.mp4"
    video_quad_path = out_dir / "rocky_terrain_quad_4_side_views.mp4"

    w_dual = imageio.get_writer(str(video_dual_path), fps=24, codec="libx264")
    w_quad = imageio.get_writer(str(video_quad_path), fps=24, codec="libx264")

    print(f"Spawn: {info['ball_xy']}, Goal: {env.scenario.goal}")
    print(f"Rocky Terrain: {env.scenario.stones[0][4]} procedural boulders & rock slabs across {env.scenario.path_length:.2f}m course")

    ctrl = env.cfg.controller
    z_history = []
    x_history = []
    frames_dual = []
    frames_quad = []

    print("\nSimulating navigation across crazy rocky terrain...")
    for step in range(360):
        ball_pos = env.data.qpos[0:3]
        quat = env.data.qpos[3:7]
        x_history.append(ball_pos[0])
        z_history.append(ball_pos[2])

        d_hat, drive = desired_direction(ball_pos[:2], env.path_pts, lookahead=float(ctrl.lookahead))
        targets = bar_targets(
            quat,
            env.dirs_body,
            env.max_extend,
            d_hat,
            drive=drive,
            min_offset=float(ctrl.base),
            back_gain=float(ctrl.back_gain),
            enable_gaussian_stance=bool(getattr(ctrl, "enable_gaussian_stance", False)),
            enable_curb_vaulting=bool(getattr(ctrl, "enable_curb_vaulting", True)),
            curb_boost_gain=float(getattr(ctrl, "curb_boost_gain", 2.6)),
        )

        obs, rew, term, trunc, info = env.step(targets)

        if step % 2 == 0:
            frame_bird = env.render(camera_name="bird_fixed")
            frame_chase = env.render(camera_name="bird_chase")
            composite = np.concatenate([frame_bird, frame_chase], axis=1)

            frame_quad = env.render(camera_name="quad")

            w_dual.append_data(composite)
            w_quad.append_data(frame_quad)

            frames_dual.append(composite)
            frames_quad.append(frame_quad)

        if step % 50 == 0:
            print(f"  Step {step:4d}: Ball pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.3f}m), Dist to Goal={info['distance']:.2f}m")

        if term or trunc:
            print(f"\nReached goal at step {step + 1}! Final distance={info['distance']:.3f}m, Success={info.get('success', False)}")
            break

    w_dual.close()
    w_quad.close()
    env.close()

    z_arr = np.array(z_history)
    print(f"\nRocky Mountain Terrain Results:")
    print(f"  - Dual Video: {video_dual_path}")
    print(f"  - 4-Side Quad Video: {video_quad_path}")
    print(f"  - Baseline z: {z_arr[0]:.4f} m")
    print(f"  - Max z across boulders: {np.max(z_arr):.4f} m (+{(np.max(z_arr) - z_arr[0])*100:.1f} cm)")
    print(f"  - Final Distance to Goal: {info['distance']:.3f} m")
    print(f"  - Goal Success: {info.get('success', False)}")

    # Extract preview stills
    if len(frames_quad) > 40:
        Image.fromarray(frames_quad[30]).save("docs/project_journey/assets/rocky_mountain_quad_preview.png")
        Image.fromarray(frames_dual[30]).save("docs/project_journey/assets/rocky_mountain_dual_preview.png")
        print("Saved preview stills in docs/project_journey/assets/!")


if __name__ == "__main__":
    test_rocky_terrain_navigation()
