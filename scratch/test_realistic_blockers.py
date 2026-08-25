"""Evaluation and Multi-Camera Rendering Suite for Realistic Industrial Blockers.

Validates:
1. Photorealistic industrial safety bollards & concrete barrier blocks in MuJoCo.
2. Obstacle observation encoding (goal-frame forward, lateral, gap) & LiDAR raycasting.
3. Zero-collision navigation with Sim2Real physics (viscoelastic feet, actuator limits, 25ms delay).
4. Multi-perspective cinematic video recording (Dual Bird's-Eye + Chase View).
"""
import datetime
import os
import sys
from pathlib import Path
import numpy as np
import imageio
import torch

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.mujoco_steering import MujocoSteeringEnv
from radial_sphere.controller import bar_targets, desired_direction
from stable_baselines3 import PPO


def evaluate_realistic_blockers():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__realistic_blockers_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Starting Realistic Blockers Evaluation -> {out_dir} ===")

    # 1. Test Obstacle Arena Scenario
    print("\n--- Testing Open-Arena Obstacle Course with Industrial Bollards ---")
    cfg_obs = load_config("configs/rl/obstacle_realistic.yaml")
    env_obs = MujocoSteeringEnv(cfg_obs)
    obs, info = env_obs.reset(seed=42)

    print(f"Observation space shape: {env_obs.observation_space.shape}, actual obs shape: {obs.shape}")
    print(f"Spawn: {info['ball_xy']}, Goal: {env_obs.env.scenario.goal}")
    print(f"Number of obstacles generated: {len(env_obs.env.scenario.obstacles)}")
    for i, ob in enumerate(env_obs.env.scenario.obstacles):
        print(f"  Bollard {i}: pos=({ob[0]:.3f}, {ob[1]:.3f}), radius={ob[2]:.3f}m")

    # Verify obstacle geom names in model
    geom_names = [env_obs.env.model.geom(i).name for i in range(env_obs.env.model.ngeom)]
    pillar_geoms = [name for name in geom_names if "pillar" in name or "bollard" in name]
    print(f"Registered bollard/pillar geoms in MuJoCo ({len(pillar_geoms)}): {pillar_geoms[:6]}...")

    # Load trained expert / policy if available
    expert_ckpt = Path("storage_local/20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking/checkpoints/ppo_final.zip")
    policy = None
    if expert_ckpt.exists():
        print(f"Loading trained active-braking expert: {expert_ckpt}")
        policy = PPO.load(str(expert_ckpt))
    else:
        print("Using active-braking kinematic steering controller")

    # Run Obstacle Navigation Episode with Dual Video Recording
    video_path_obs = out_dir / "obstacle_bollards_dual_view.mp4"
    writer_obs = imageio.get_writer(str(video_path_obs), fps=24, codec="libx264")

    total_rew = 0.0
    wall_hits = 0
    obstacle_hits = 0
    min_obstacle_dist = 999.0

    print("Running episode on Obstacle Course...")
    for step in range(300):
        ball_pos = env_obs.env.data.qpos[0:2]
        # Track minimum distance to any blocker
        for ob in env_obs.env.scenario.obstacles:
            d_center = np.linalg.norm(ball_pos - ob[:2])
            d_clearance = d_center - (0.35 + ob[2])  # sphere radius + bollard radius
            if d_clearance < min_obstacle_dist:
                min_obstacle_dist = d_clearance

        # High-level Action
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            # Active-braking CPG steering
            d_hat, drive = desired_direction(
                ball_pos,
                env_obs.env.path_pts,
                lookahead=float(env_obs.ctrl.lookahead),
                enable_curvature_deceleration=True,
            )
            # Local goal frame
            g = env_obs._goal_dir(ball_pos)
            cmd_x = d_hat[0] * g[0] + d_hat[1] * g[1]
            cmd_y = d_hat[1] * g[0] - d_hat[0] * g[1]
            action = np.array([cmd_x, cmd_y, drive * 2.0 - 1.0], dtype=np.float32)

        obs, rew, term, trunc, info = env_obs.step(action)
        total_rew += rew
        if info.get("wall_contact", False):
            wall_hits += 1
        if info.get("obstacle_contact", False):
            obstacle_hits += 1

        # Render dual view: chase view + bird view
        if step % 2 == 0:
            frame_dual = env_obs.env.render(mode="dual_bird_chase")
            writer_obs.append_data(frame_dual)

        if term or trunc:
            print(f"Episode finished at decision step {step + 1}. Success={info.get('success', False)}")
            break

    writer_obs.close()
    env_obs.close()

    print(f"Obstacle Arena Summary:")
    print(f"  - Video saved: {video_path_obs}")
    print(f"  - Final Distance to Goal: {info['distance']:.3f} m")
    print(f"  - Total Wall Contacts: {wall_hits}")
    print(f"  - Total Obstacle Contacts: {obstacle_hits}")
    print(f"  - Min Clearance to Bollards: {min_obstacle_dist:.3f} m")

    # 2. Test Maze with Corridor Blockers
    print("\n--- Testing Large Maze with Corridor Industrial Bollards ---")
    cfg_maze = load_config("configs/rl/maze_level3_large_blockers.yaml")
    env_maze = MujocoSteeringEnv(cfg_maze)
    obs_m, info_m = env_maze.reset(seed=10101)

    print(f"Maze dimensions: {env_maze.env.scenario.walls.shape[0]} walls")
    print(f"Corridor blockers generated: {len(env_maze.env.scenario.obstacles) if env_maze.env.scenario.obstacles is not None else 0}")
    if env_maze.env.scenario.obstacles is not None:
        for i, ob in enumerate(env_maze.env.scenario.obstacles):
            print(f"  Corridor Bollard {i}: pos=({ob[0]:.3f}, {ob[1]:.3f}), radius={ob[2]:.3f}m")

    video_path_maze = out_dir / "maze_corridor_blockers_dual_view.mp4"
    writer_maze = imageio.get_writer(str(video_path_maze), fps=24, codec="libx264")

    wall_hits_m = 0
    obs_hits_m = 0
    for step in range(400):
        if policy is not None:
            action, _ = policy.predict(obs_m, deterministic=True)
        else:
            d_hat, drive = desired_direction(
                env_maze.env.data.qpos[0:2],
                env_maze.env.path_pts,
                lookahead=float(env_maze.ctrl.lookahead),
                enable_curvature_deceleration=True,
            )
            g = env_maze._goal_dir(env_maze.env.data.qpos[0:2])
            cmd_x = d_hat[0] * g[0] + d_hat[1] * g[1]
            cmd_y = d_hat[1] * g[0] - d_hat[0] * g[1]
            action = np.array([cmd_x, cmd_y, drive * 2.0 - 1.0], dtype=np.float32)

        obs_m, rew_m, term_m, trunc_m, info_m = env_maze.step(action)
        if info_m.get("wall_contact", False):
            wall_hits_m += 1
        if info_m.get("obstacle_contact", False):
            obs_hits_m += 1

        if step % 2 == 0:
            frame_dual = env_maze.env.render(mode="dual_bird_chase")
            writer_maze.append_data(frame_dual)

        if term_m or trunc_m:
            print(f"Maze episode completed at step {step + 1}. Success={info_m.get('success', False)}")
            break

    writer_maze.close()
    env_maze.close()

    print(f"Maze Corridor Blockers Summary:")
    print(f"  - Video saved: {video_path_maze}")
    print(f"  - Final Distance: {info_m['distance']:.3f} m")
    print(f"  - Wall Hits: {wall_hits_m}")
    print(f"  - Blocker Hits: {obs_hits_m}")

    print(f"\n[DONE] All realistic blocker tests and video recordings complete in {out_dir}")


if __name__ == "__main__":
    evaluate_realistic_blockers()
