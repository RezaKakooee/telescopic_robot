"""Render Long 25-Meter Mountain Expedition across 900-Boulder Rocky Terrain.

Features:
1. 25.0-meter long expansive mountain rock field with 900 procedural boulders & stone slabs.
2. Underbelly Ground-Touching Stance active (downward rods continuously support & touch ground/rocks).
3. Produces a full-length, cinematic video (40-50+ seconds duration).
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


def render_epic_rocky_expedition():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__epic_rocky_expedition_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering 25m Epic Rocky Mountain Expedition -> {out_dir} ===")

    cfg = load_config("configs/rl/rocky_mountain_terrain.yaml")
    cfg.controller.enable_underbelly_contact = True
    cfg.controller.enable_adaptive_grouping = False
    cfg.controller.enable_gaussian_stance = False
    cfg.controller.underbelly_stance_gain = 0.40
    cfg.controller.underbelly_threshold_z = -0.20
    cfg.controller.back_gain = 2.0

    env = MujocoRadialSphereEnv(cfg, max_steps=3500)
    obs, info = env.reset(seed=42)

    # Video Writers for High-Definition Long Expedition Videos
    v_quad = out_dir / "epic_rocky_quad_4_side_views.mp4"
    v_side = out_dir / "epic_rocky_side_tracking.mp4"
    v_dual = out_dir / "epic_rocky_dual_map_and_chase.mp4"

    w_quad = imageio.get_writer(str(v_quad), fps=24, codec="libx264")
    w_side = imageio.get_writer(str(v_side), fps=24, codec="libx264")
    w_dual = imageio.get_writer(str(v_dual), fps=24, codec="libx264")

    print(f"Spawn: {info['ball_xy']}, Goal: {env.scenario.goal}")
    print(f"Course Length: {env.scenario.path_length:.2f} meters!")
    print(f"Total Rocks: {env.scenario.stones[0][4]} procedural boulders & rock slabs")

    ctrl = env.cfg.controller
    step = 0
    frames_quad = []
    frames_dual = []

    print("\nSimulating full 25m mountain expedition...")
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

        # Render frame
        frame_quad = env.render(camera_name="quad")
        frame_side = env.render(camera_name="side_right_30deg")
        frame_bird = env.render(camera_name="bird_fixed")
        frame_chase = env.render(camera_name="bird_chase")
        frame_dual = np.concatenate([frame_bird, frame_chase], axis=1)

        w_quad.append_data(frame_quad)
        w_side.append_data(frame_side)
        w_dual.append_data(frame_dual)

        frames_quad.append(frame_quad)
        frames_dual.append(frame_dual)

        if step % 100 == 0:
            print(f"  Step {step:4d}: Ball pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.3f}m), Dist to Goal={info['distance']:.2f}m")

        if term or trunc or info["distance"] < 0.45:
            print(f"\n🎉 Reached Goal at Step {step}! Final distance={info['distance']:.3f}m, Success={info.get('success', False)}")
            break

        if step >= 2500:
            print(f"Reached step limit at step {step}. Final distance={info['distance']:.2f}m")
            break

    w_quad.close()
    w_side.close()
    w_dual.close()
    env.close()

    duration_sec = step / 24.0
    print(f"\n25-Meter Mountain Expedition Complete!")
    print(f"  - Total Video Frames: {step} frames")
    print(f"  - Video Duration: {duration_sec:.1f} seconds! (Full-Length Video)")
    print(f"  - 4-Side Quad Video: {v_quad}")
    print(f"  - Side Tracking Video: {v_side}")
    print(f"  - Dual Map & Chase Video: {v_dual}")

    # Extract preview stills
    if len(frames_quad) > 100:
        p1 = Path("docs/project_journey/assets/epic_rocky_25m_quad_midpoint.png")
        Image.fromarray(frames_quad[len(frames_quad) // 2]).save(p1)
        p2 = Path("docs/project_journey/assets/epic_rocky_25m_dual_overview.png")
        Image.fromarray(frames_dual[len(frames_dual) // 2]).save(p2)
        print(f"Saved preview stills in docs/project_journey/assets/!")


if __name__ == "__main__":
    render_epic_rocky_expedition()
