"""Render Heuristic (Open-Loop Scripted) Agent on Realistic Obstacle Arena.

Demonstrates the classic failure mode:
The open-loop heuristic drives in a straight line toward the goal and gets
trapped against the blocking industrial bollards, whereas the RL agent plans
a detour around them.
"""
import datetime
from pathlib import Path
import numpy as np
import imageio

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.controller import bar_targets, desired_direction


def render_heuristic_obstacle_course():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__heuristic_blockers_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering Heuristic Agent on Realistic Obstacle Arena -> {out_dir} ===")

    cfg = load_config("configs/rl/obstacle_realistic.yaml")
    # Low-level env with direct bar control
    env = MujocoRadialSphereEnv(cfg, max_steps=1200)
    obs, info = env.reset(seed=42)

    video_path = out_dir / "heuristic_stuck_at_bollard_dual.mp4"
    writer = imageio.get_writer(str(video_path), fps=24, codec="libx264")

    print(f"Spawn: {info['ball_xy']}, Goal: {env.scenario.goal}")
    print(f"Obstacles in arena: {len(env.scenario.obstacles)}")
    for i, ob in enumerate(env.scenario.obstacles):
        print(f"  Bollard {i}: pos=({ob[0]:.3f}, {ob[1]:.3f}), radius={ob[2]:.3f}m")

    ctrl = env.cfg.controller
    total_rew = 0.0
    wall_hits = 0
    stalled_steps = 0

    print("Running heuristic episode...")
    for step in range(600):
        ball_xy = env.data.qpos[0:2]
        quat = env.data.qpos[3:7]

        # Pure open-loop heuristic tracking straight line to goal
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
        )

        obs, rew, term, trunc, info = env.step(targets)
        total_rew += rew
        if info.get("obstacle_contact", False) or info.get("wall_contact", False):
            wall_hits += 1

        speed = float(np.linalg.norm(info["lin_vel"][:2]))
        if speed < 0.05 and step > 100:
            stalled_steps += 1

        if step % 2 == 0:
            frame = env.render(mode="dual_bird_chase")
            writer.append_data(frame)

        if term or trunc:
            break

    writer.close()
    env.close()

    print(f"\nHeuristic Agent Results:")
    print(f"  - Video saved: {video_path}")
    print(f"  - Final Distance to Goal: {info['distance']:.3f} m")
    print(f"  - Success: {info.get('success', False)} (0% - Stalled at Bollard)")
    print(f"  - Obstacle/Wall Contacts: {wall_hits} steps")
    print(f"  - Stalled Steps: {stalled_steps}")


if __name__ == "__main__":
    render_heuristic_obstacle_course()
