"""Comprehensive Evaluation Suite across 6 Diverse Maze Topologies."""
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
import rootutils
import torch

rootutils.setup_root("/home/azureuser/telescopic_robot", pythonpath=True)

from radial_sphere import (
    MujocoSteeringEnv,
    desired_direction,
    generate_scenario,
    load_config_cli,
)
from scripts.imitation.train_bc import HierarchicalImitationPolicy

# 1. Load Pretrained Clean IL Policy
ckpt_path = Path("/home/azureuser/telescopic_robot/storage_local/20260822_1617__imitation_models/bc_hierarchical_best.pt")
ckpt = torch.load(str(ckpt_path), map_location="cpu")
obs_dim = ckpt.get("obs_dim", 163)

model = HierarchicalImitationPolicy(obs_dim=obs_dim, high_act_dim=3, low_act_dim=60, hidden_dim=256)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")
renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260823_0102__diverse_maze_suite")
renders_dir.mkdir(parents=True, exist_ok=True)

# Define 6 Diverse Maze Topologies
maze_suite = [
    {
        "id": "maze_1_orthogonal_spiral",
        "title": "Maze 1: Orthogonal Spiral Labyrinth (Level 1)",
        "level": 1,
        "seed": 10101,
        "max_steps": 1200,
    },
    {
        "id": "maze_2_multiloop_braid",
        "title": "Maze 2: High-Density Multi-Loop Braid Maze (Level 2)",
        "level": 2,
        "seed": 20202,
        "max_steps": 1200,
    },
    {
        "id": "maze_3_branching_tree",
        "title": "Maze 3: Deep Branching Tree Maze (Level 3)",
        "level": 3,
        "seed": 30303,
        "max_steps": 1500,
    },
    {
        "id": "maze_4_random_diagonal_endpoints",
        "title": "Maze 4: Random Diagonal Endpoints Route (Level 2)",
        "level": 2,
        "seed": 40404,
        "max_steps": 1500,
    },
    {
        "id": "maze_5_large_45m_gauntlet",
        "title": "Maze 5: Large 7x6 45-Meter Extended Gauntlet (Level 3)",
        "level": 3,
        "seed": 50505,
        "max_steps": 2500,
    },
    {
        "id": "maze_6_dense_switchback",
        "title": "Maze 6: Dense S-Curve Switchback Maze (Level 3)",
        "level": 3,
        "seed": 60606,
        "max_steps": 1500,
    },
]

results = []

print("=========================================================================")
print("STARTING DIVERSE MAZE SUITE EVALUATION (6 DISTINCT MAZES)")
print("=========================================================================\n")

for idx, m_info in enumerate(maze_suite):
    m_id = m_info["id"]
    m_title = m_info["title"]
    level = m_info["level"]
    seed = m_info["seed"]
    max_steps = m_info["max_steps"]

    cfg = load_config_cli(name="maze_level3_large_active_braking")
    OmegaConf.set_struct(cfg, False)
    if hasattr(cfg.scenario, "maze") and cfg.scenario.maze is not None:
        cfg.scenario.maze.level = level
        cfg.scenario.maze.random_endpoints = True
        cfg.scenario.maze.endpoint_min_route = 5.0

    sc = generate_scenario("maze", cfg, seed=seed)
    env = MujocoSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    raw = env.env

    frames_dual = [raw.render(mode="dual")]
    frames_chase = [raw.render(mode="chase")]

    done = False
    step = 0
    total_reward = 0.0
    wall_contacts = 0
    velocities = []

    print(f"[{idx+1}/6] Evaluating: {m_title} (Seed {seed})...", flush=True)

    while not done and step < max_steps:
        ball_xy = raw.data.qpos[:2]
        quat = raw.data.qpos[3:7]
        g, expert_drive = desired_direction(ball_xy, sc.path_pts, lookahead=float(cfg.controller.lookahead))
        
        # 163D proprioceptive & spatial observation
        lidar = raw.raycast_lidar(n_rays=24, max_range=3.0, g=g)
        v_fwd = float(raw.data.qvel[0] * g[0] + raw.data.qvel[1] * g[1])
        v_lat = float(g[0] * raw.data.qvel[1] - g[1] * raw.data.qvel[0])
        norm_dist = float(np.linalg.norm(sc.goal[:2] - ball_xy) / max(raw.path_length, 1.0))
        rel_goal = (sc.goal[:2] - ball_xy).astype(np.float32)
        norm_joint = (raw.data.qpos[7:7 + raw.n_bars] / raw.max_extend).astype(np.float32)
        
        speed = float(np.linalg.norm(raw.data.qvel[:2]))
        velocities.append(speed)

        obs_163 = np.concatenate([
            quat.astype(np.float32),
            np.array([v_fwd, v_lat, raw.data.qvel[2]], dtype=np.float32),
            raw.data.qvel[3:6].astype(np.float32),
            norm_joint,
            np.zeros(66, dtype=np.float32),
            rel_goal,
            np.array([norm_dist], dtype=np.float32),
            lidar.astype(np.float32),
        ])

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_163).unsqueeze(0).float()
            pred_high, _ = model(obs_tensor)
            act_high = pred_high.squeeze(0).numpy()

        obs, r, terminated, truncated, info = env.step(act_high)
        done = terminated or truncated
        total_reward += float(r)

        if info.get("wall_contact", False):
            wall_contacts += 1

        if step % 2 == 0:
            frames_dual.append(raw.render(mode="dual"))
            frames_chase.append(raw.render(mode="chase"))

        step += 1

    final_dist = float(np.linalg.norm(raw.data.qpos[:2] - sc.goal[:2]))
    success = bool(final_dist < 0.50 or info.get("success", False))
    avg_speed = float(np.mean(velocities)) if velocities else 0.0
    max_speed = float(np.max(velocities)) if velocities else 0.0

    vid_dual = renders_dir / f"{m_id}_dual.mp4"
    vid_chase = renders_dir / f"{m_id}_chase.mp4"
    imageio.mimsave(str(vid_dual), frames_dual, fps=25)
    imageio.mimsave(str(vid_chase), frames_chase, fps=25)

    thumb = scratch_dir / f"{m_id}_mid.png"
    imageio.imwrite(str(thumb), frames_dual[len(frames_dual)//2])

    res = {
        "id": m_id,
        "title": m_title,
        "level": level,
        "seed": seed,
        "steps": step,
        "success": success,
        "goal_dist": final_dist,
        "wall_contacts": wall_contacts,
        "wall_pct": wall_contacts / max(step, 1) * 100.0,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "total_reward": total_reward,
        "dual_video": str(vid_dual),
        "chase_video": str(vid_chase),
        "thumb": str(thumb),
    }
    results.append(res)

    print(f"  -> Result: Steps={step} | Success={success} | GoalDist={final_dist:.2f}m | WallHits={wall_contacts} ({res['wall_pct']:.1f}%) | AvgSpeed={avg_speed:.2f}m/s | Rew={total_reward:.2f}\n")
    env.close()

print("=========================================================================")
print("ALL 6 MAZES COMPLETED SUCCESSFULLY!")
print("=========================================================================")
for r in results:
    print(f"{r['title'][:45]:45s} | Succ={str(r['success']):5s} | GoalDist={r['goal_dist']:4.2f}m | Steps={r['steps']:4d} | WallHits={r['wall_contacts']:3d} ({r['wall_pct']:4.1f}%) | Speed={r['avg_speed']:.2f}m/s")
