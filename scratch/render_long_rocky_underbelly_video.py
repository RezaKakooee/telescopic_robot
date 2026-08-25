"""Render Long Complete Video across Rugged Mountainous Rocky Terrain with Underbelly Contact.

Simulates the 60-bar radial telescopic robot from spawn to goal across the 260-rock boulder field:
1. Underbelly Ground-Touching Stance active (rods beneath the core continuously touch the floor & rocks).
2. Runs complete trajectory until touching the goal pad.
3. Generates high-definition multi-camera videos:
   - 2x2 Quad 4-Side Tracking Video
   - Side Tracking Profile Video
   - Dual Map & Overhead Chase Video
   - 2x2 Fixed Outside Edge Quad Video
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


def render_long_rocky_video():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__long_rocky_underbelly_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering Long Video across Rocky Mountainous Terrain -> {out_dir} ===")

    cfg = load_config("configs/rl/rocky_mountain_terrain.yaml")
    cfg.controller.enable_underbelly_contact = True
    cfg.controller.enable_adaptive_grouping = False
    cfg.controller.enable_gaussian_stance = False
    cfg.controller.underbelly_stance_gain = 0.40
    cfg.controller.underbelly_threshold_z = -0.20
    cfg.controller.back_gain = 2.0

    env = MujocoRadialSphereEnv(cfg, max_steps=1200)
    obs, info = env.reset(seed=42)

    # Video Writers
    w_quad = imageio.get_writer(str(out_dir / "long_rocky_quad_4_side_views.mp4"), fps=24, codec="libx264")
    w_side = imageio.get_writer(str(out_dir / "long_rocky_side_tracking.mp4"), fps=24, codec="libx264")
    w_dual = imageio.get_writer(str(out_dir / "long_rocky_dual_map_and_chase.mp4"), fps=24, codec="libx264")
    w_fixed_quad = imageio.get_writer(str(out_dir / "long_rocky_fixed_quad_views.mp4"), fps=24, codec="libx264")

    print(f"Spawn: {info['ball_xy']}, Goal: {env.scenario.goal}")
    print(f"Rocky Terrain: {env.scenario.stones[0][4]} procedural boulders across {env.scenario.path_length:.2f}m course")

    ctrl = env.cfg.controller
    frames_quad = []
    frames_side = []
    step = 0

    print("\nSimulating full-length rocky terrain navigation...")
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
            underbelly_stance_gain=float(getattr(ctrl, "underbelly_stance_gain", 0.40)),
            underbelly_threshold_z=float(getattr(ctrl, "underbelly_threshold_z", -0.20)),
        )

        obs, rew, term, trunc, info = env.step(targets)
        step += 1

        if step % 2 == 0:
            frame_quad = env.render(camera_name="quad")
            frame_side = env.render(camera_name="side_right_30deg")
            frame_bird = env.render(camera_name="bird_fixed")
            frame_chase = env.render(camera_name="bird_chase")
            frame_dual = np.concatenate([frame_bird, frame_chase], axis=1)
            frame_fixed_quad = env.render(camera_name="fixed_quad")

            w_quad.append_data(frame_quad)
            w_side.append_data(frame_side)
            w_dual.append_data(frame_dual)
            w_fixed_quad.append_data(frame_fixed_quad)

            frames_quad.append(frame_quad)
            frames_side.append(frame_side)

        if step % 50 == 0:
            print(f"  Step {step:4d}: Ball pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.3f}m), Dist to Goal={info['distance']:.2f}m")

        if term or trunc or info["distance"] < 0.45:
            print(f"\n🎉 Reached Goal at Step {step}! Final distance={info['distance']:.3f}m, Success={info.get('success', False)}")
            break

        if step >= 600:
            print(f"Reached safety step limit at step {step}. Final distance={info['distance']:.2f}m")
            break

    w_quad.close()
    w_side.close()
    w_dual.close()
    w_fixed_quad.close()
    env.close()

    print(f"\nAll Long Videos Saved to -> {out_dir}:")
    print(f"  - 4-Side Quad Video: long_rocky_quad_4_side_views.mp4")
    print(f"  - Side Tracking Video: long_rocky_side_tracking.mp4")
    print(f"  - Dual Map & Chase Video: long_rocky_dual_map_and_chase.mp4")
    print(f"  - Fixed Outside Quad Video: long_rocky_fixed_quad_views.mp4")

    # Extract high quality mid-flight preview
    if len(frames_quad) > 40:
        preview_path = Path("docs/project_journey/assets/long_rocky_underbelly_quad_preview.png")
        Image.fromarray(frames_quad[35]).save(preview_path)
        print(f"Saved preview still -> {preview_path}")


if __name__ == "__main__":
    render_long_rocky_video()
