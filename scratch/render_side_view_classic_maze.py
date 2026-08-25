"""Render Side-Profile Views of the Previous Classic 7x6 Large Maze with Blockers.

Validates & Records:
1. Exact previous 7x6 maze (seed=42) from configs/rl/maze_level3_large_blockers.yaml.
2. High-definition lateral side-profile camera ('side_profile') showing rod extensions & ground contacts.
3. Dual-view composite (Maze overview on left, Side-profile tracking on right).
"""
import datetime
from pathlib import Path
import numpy as np
import imageio
from PIL import Image

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.controller import bar_targets, desired_direction


def render_classic_maze_side_view():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__classic_maze_side_view")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering Side Profile of Classic 7x6 Maze -> {out_dir} ===")

    cfg = load_config("configs/rl/maze_level3_large_blockers.yaml")
    env = MujocoRadialSphereEnv(cfg, max_steps=800)
    obs, info = env.reset(seed=42)

    video_side_path = out_dir / "classic_maze_side_profile.mp4"
    video_dual_path = out_dir / "classic_maze_side_dual_composite.mp4"

    w_side = imageio.get_writer(str(video_side_path), fps=24, codec="libx264")
    w_dual = imageio.get_writer(str(video_dual_path), fps=24, codec="libx264")

    print(f"Maze Grid: 7 cols x 6 rows (Classic Level 3 Large Maze)")
    print(f"Spawn: {info['ball_xy']}, Goal: {env.scenario.goal}")
    print(f"Bollards: {len(env.scenario.obstacles) if env.scenario.obstacles is not None else 0}")
    print(f"Steps/Wood: {len(env.scenario.steps) if env.scenario.steps is not None else 0}")
    print(f"Gaps/Holes: {len(env.scenario.gaps) if env.scenario.gaps is not None else 0}")

    ctrl = env.cfg.controller
    frames_side = []
    frames_dual = []

    print("\nSimulating navigation with side-profile camera...")
    for step in range(320):
        ball_xy = env.data.qpos[0:2]
        quat = env.data.qpos[3:7]

        d_hat, drive = desired_direction(ball_xy, env.path_pts, lookahead=float(ctrl.lookahead))
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
            frame_side = env.render(camera_name="side_profile")
            composite = np.concatenate([frame_bird, frame_side], axis=1)

            w_side.append_data(frame_side)
            w_dual.append_data(composite)
            frames_side.append(frame_side)
            frames_dual.append(composite)

        if term or trunc:
            print(f"Episode completed at step {step + 1}. Success={info.get('success', False)}")
            break

    w_side.close()
    w_dual.close()
    env.close()

    print(f"\nRender Complete:")
    print(f"  - Side Profile Video: {video_side_path}")
    print(f"  - Side Dual Composite: {video_dual_path}")

    # Extract key preview stills
    if len(frames_dual) > 40:
        Image.fromarray(frames_dual[35]).save("docs/project_journey/assets/classic_maze_side_dual_preview.png")
        Image.fromarray(frames_side[35]).save("docs/project_journey/assets/classic_maze_side_profile_still.png")
        print("Saved preview images in docs/project_journey/assets/!")


if __name__ == "__main__":
    render_classic_maze_side_view()
