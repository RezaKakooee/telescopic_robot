"""Collision-Free Expert Controller Test Script."""
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
import rootutils

rootutils.setup_root("/home/azureuser/telescopic_robot", pythonpath=True)

from radial_sphere import (
    MujocoSteeringEnv,
    bar_targets,
    desired_direction,
    generate_scenario,
    load_config_cli,
)
from radial_sphere.geometry import quat_to_rotmat


def compute_collision_free_expert(raw_env, path_pts, cfg):
    ball_xy = raw_env.data.qpos[:2]
    quat = raw_env.data.qpos[3:7]
    rotmat = quat_to_rotmat(quat)

    # 1. Base lookahead path direction
    g_path, base_drive = desired_direction(ball_xy, path_pts, lookahead=float(cfg.controller.lookahead))
    if base_drive < 1e-3:
        # At goal
        act_60 = np.full(raw_env.n_bars, raw_env.base_offset, dtype=np.float32)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), act_60

    # 2. 24-ray LiDAR world scan & Artificial Potential Field
    lidar_24 = raw_env.raycast_lidar(n_rays=24, max_range=3.0, g=g_path)
    # Reconstruct absolute world angles for each ray
    angles_rel = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    g_angle = np.arctan2(g_path[1], g_path[0])
    angles_world = g_angle + angles_rel

    d_safe = 0.60
    f_rep = np.zeros(2, dtype=np.float32)
    min_front_d = 3.0

    for i in range(24):
        d_i = float(lidar_24[i]) * 3.0  # Actual metres
        ray_dir = np.array([np.cos(angles_world[i]), np.sin(angles_world[i])])
        
        # Check forward arc (+/- 45 deg)
        if abs(angles_rel[i]) < np.pi / 4 or abs(angles_rel[i] - 2 * np.pi) < np.pi / 4:
            min_front_d = min(min_front_d, d_i)

        if d_i < d_safe:
            push_mag = ((d_safe - d_i) / d_safe) ** 2
            f_rep += push_mag * (-ray_dir)

    # 3. Blended repulsive navigation vector
    d_steer = g_path + 1.8 * f_rep
    s_norm = np.linalg.norm(d_steer)
    d_world = d_steer / s_norm if s_norm > 1e-5 else g_path

    # Transform d_world into goal frame for highlevel action
    cmd_gf_x = float(d_world[0] * g_path[0] + d_world[1] * g_path[1])
    cmd_gf_y = float(g_path[0] * d_world[1] - g_path[1] * d_world[0])

    # 4. Corner Pre-Braking Throttle
    if min_front_d < 0.85:
        drive_throttle = float(np.clip((min_front_d - 0.20) / 0.65, 0.20, 1.0))
    else:
        drive_throttle = 1.0

    act_high = np.array([cmd_gf_x, cmd_gf_y, drive_throttle], dtype=np.float32)

    min_offset = float(getattr(cfg.controller, "min_offset", 0.025))
    # 5. Base wave targets
    base_targets = bar_targets(
        quat=quat,
        dirs_body=raw_env.dirs_body,
        max_extend=raw_env.max_extend,
        d_hat=d_world,
        drive=drive_throttle,
        min_offset=min_offset,
        back_gain=float(cfg.controller.back_gain),
    )

    # 6. Active Rod Clearance Retraction (Tuck rods facing walls)
    dirs_world = raw_env.dirs_body @ rotmat.T
    clean_targets = base_targets.copy()

    for k, (ux, uy, uz) in enumerate(dirs_world):
        rod_angle = np.arctan2(uy, ux)
        diffs = np.abs((angles_world - rod_angle + np.pi) % (2 * np.pi) - np.pi)
        closest_ray = int(np.argmin(diffs))
        d_wall = float(lidar_24[closest_ray]) * 3.0

        # If wall is within 0.42m and rod points towards wall, tuck it
        if d_wall < 0.42 and abs(uz) < 0.70:
            clean_targets[k] = min(clean_targets[k], min_offset + 0.005)

    return act_high, clean_targets


# Test across the 3 test episodes
cfg = load_config_cli(name="maze_level3_random_endpoints")
OmegaConf.set_struct(cfg, False)
cfg.scenario.maze.level = 1
cfg.scenario.maze.random_endpoints = True
cfg.scenario.maze.endpoint_min_route = 6.0

sc = generate_scenario("maze", cfg, seed=9901)
env = MujocoSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=2000)
obs, _ = env.reset(seed=9901)
raw = env.env

step = 0
done = False
wall_contacts = 0

print("Testing Collision-Free Expert on Level 1 Orthogonal Maze (Seed 9901)...", flush=True)

while not done and step < 1500:
    act_high, act_low = compute_collision_free_expert(raw, sc.path_pts, cfg)
    obs, r, terminated, truncated, info = env.step(act_high)
    done = terminated or truncated
    
    if info.get("wall_contact", False):
        wall_contacts += 1
    step += 1

final_dist = float(np.linalg.norm(raw.data.qpos[:2] - sc.goal[:2]))
success = bool(final_dist < 0.50 or info.get("success", False))

print(f"\n[COLLISION-FREE EXPERT RESULT]")
print(f"Steps: {step} | Success: {success} | GoalDist: {final_dist:.2f}m | WallHits: {wall_contacts} ({wall_contacts/step*100:.1f}%)")
env.close()
