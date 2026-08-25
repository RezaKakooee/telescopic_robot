"""Evaluation and Video Recording Suite for Extra-Long Mega-Labyrinth Maze.

Tests & Renders:
1. 10x8 Grid Mega-Labyrinth (80 cells, ~70m - 90m corridor route).
2. Embedded Hazards: 6 Industrial Bollards, 3 Floor Pit Holes, 3 Wooden Timber Curbs, Stone Fields.
3. Dual-View Video Recording (Full 15m x 12m labyrinth overview + Close tracking chase).
"""
import datetime
from pathlib import Path
import numpy as np
import imageio
from PIL import Image

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.controller import bar_targets, desired_direction


def run_extra_long_maze():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__extra_long_maze_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Testing Extra-Long Mega-Labyrinth Maze -> {out_dir} ===")

    cfg = load_config("configs/rl/maze_level3_extra_long.yaml")
    env = MujocoRadialSphereEnv(cfg, max_steps=4500)
    obs, info = env.reset(seed=42)

    video_dual_path = out_dir / "extra_long_maze_dual_view.mp4"
    writer = imageio.get_writer(str(video_dual_path), fps=24, codec="libx264")

    print(f"Labyrinth Dimensions: {cfg.scenario.maze.cols} cols x {cfg.scenario.maze.rows} rows (80 cells)")
    print(f"Spawn: {info['ball_xy']}, Goal: {env.scenario.goal}")
    print(f"Total Corridor Route Length: {env.scenario.path_length:.2f} meters!")
    print(f"Embedded Obstacles & Hazards:")
    print(f"  - Industrial Bollards: {len(env.scenario.obstacles) if env.scenario.obstacles is not None else 0}")
    print(f"  - Floor Pit Holes: {len(env.scenario.gaps) if env.scenario.gaps is not None else 0}")
    print(f"  - Wooden Curbs: {len(env.scenario.steps) if env.scenario.steps is not None else 0}")
    print(f"  - Stone Fields: {len(env.scenario.stones) if env.scenario.stones is not None else 0}")

    ctrl = env.cfg.controller
    total_r = 0.0
    wall_contacts = 0
    frames = []

    print("\nSimulating navigation through extra-long labyrinth...")
    for step in range(1200):
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
        total_r += rew

        if info.get("wall_contact", False) or info.get("obstacle_contact", False):
            wall_contacts += 1

        if step % 4 == 0:
            frame = env.render(mode="dual_bird_chase")
            writer.append_data(frame)
            frames.append(frame)

        if step % 300 == 0:
            print(f"  Step {step:4d}: Ball pos=({ball_xy[0]:.2f}, {ball_xy[1]:.2f}), Dist to Goal={info['distance']:.2f}m")

        if term or trunc:
            print(f"Episode terminated at step {step + 1}. Success={info.get('success', False)}")
            break

    writer.close()
    env.close()

    print(f"\nExtra-Long Mega-Labyrinth Results:")
    print(f"  - Video saved: {video_dual_path}")
    print(f"  - Route Length: {env.scenario.path_length:.2f} meters")
    print(f"  - Final Distance to Goal: {info['distance']:.3f} meters")
    print(f"  - Wall/Obstacle Contacts: {wall_contacts} steps")

    # Save overview frame stills
    if len(frames) > 20:
        Image.fromarray(frames[10]).save("docs/project_journey/assets/extra_long_maze_overview_dual.png")
        if len(frames) > 100:
            Image.fromarray(frames[100]).save("docs/project_journey/assets/extra_long_maze_midpoint_dual.png")
        print("Saved preview images in docs/project_journey/assets/!")


if __name__ == "__main__":
    run_extra_long_maze()
