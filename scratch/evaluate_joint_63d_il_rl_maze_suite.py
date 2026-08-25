"""Evaluation of Joint IL + RL Policy (simultaneous Low-Level 60D + High-Level 3D Control) across 6 Diverse Mazes."""
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
import rootutils
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

rootutils.setup_root("/home/azureuser/telescopic_robot", pythonpath=True)

from radial_sphere import (
    MujocoLowLevelEnv,
    generate_scenario,
    load_config_cli,
)

# 1. Load the Fine-Tuned Joint IL + RL Model (60D Low-Level + High-Level)
base_dir = Path("/home/azureuser/telescopic_robot/storage_local")
matching = sorted(base_dir.glob("*__finetune_rl_from_bc__ppo__lowlevel*"))
assert len(matching) > 0, "No fine-tuned BC->RL runs found"
exp_dir = matching[-1]

model_path = exp_dir / "checkpoints" / "ppo_final.zip"
norm_path = exp_dir / "checkpoints" / "vecnormalize_final.pkl"

print(f"Loaded Joint 60D Low-Level + High-Level IL+RL Checkpoint from:\n  {exp_dir}")

scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")
renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260823_0104__joint_63d_il_rl_suite")
renders_dir.mkdir(parents=True, exist_ok=True)

# 6 Diverse Maze Topologies
maze_suite = [
    {
        "id": "joint_maze_1_spiral",
        "title": "Maze 1: Orthogonal Spiral Labyrinth (Level 1)",
        "level": 1,
        "seed": 10101,
        "max_steps": 1200,
    },
    {
        "id": "joint_maze_2_braid",
        "title": "Maze 2: Multi-Loop Braid Maze (Level 2)",
        "level": 2,
        "seed": 20202,
        "max_steps": 1200,
    },
    {
        "id": "joint_maze_3_tree",
        "title": "Maze 3: Deep Branching Tree Maze (Level 3)",
        "level": 3,
        "seed": 30303,
        "max_steps": 1500,
    },
    {
        "id": "joint_maze_4_diagonal",
        "title": "Maze 4: Random Diagonal Endpoints Route (Level 2)",
        "level": 2,
        "seed": 40404,
        "max_steps": 1500,
    },
    {
        "id": "joint_maze_5_large_45m",
        "title": "Maze 5: Large 7x6 45-Meter Extended Gauntlet (Level 3)",
        "level": 3,
        "seed": 50505,
        "max_steps": 2500,
    },
    {
        "id": "joint_maze_6_switchback",
        "title": "Maze 6: Dense S-Curve Switchback Maze (Level 3)",
        "level": 3,
        "seed": 60606,
        "max_steps": 1500,
    },
]

results = []

print("\n=========================================================================")
print("EVALUATING JOINT IL+RL POLICY (LOW-LEVEL 60D + HIGH-LEVEL CONTROL)")
print("=========================================================================\n")

for idx, m_info in enumerate(maze_suite):
    m_id = m_info["id"]
    m_title = m_info["title"]
    level = m_info["level"]
    seed = m_info["seed"]
    max_steps = m_info["max_steps"]

    cfg = load_config_cli(name="maze_level3_large_active_braking_multiaxis")
    OmegaConf.set_struct(cfg, False)
    if hasattr(cfg.scenario, "maze") and cfg.scenario.maze is not None:
        cfg.scenario.maze.level = level
        cfg.scenario.maze.random_endpoints = True
        cfg.scenario.maze.endpoint_min_route = 5.0

    sc = generate_scenario("maze", cfg, seed=seed)

    def make_env():
        return MujocoLowLevelEnv(cfg, scenario=sc, randomize=False, max_steps=max_steps)

    vec_env = DummyVecEnv([make_env])
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env, device="cpu")

    obs = vec_env.reset()
    raw_env = vec_env.envs[0]

    frames_dual = [raw_env.render(mode="dual")]
    frames_chase = [raw_env.render(mode="chase")]

    done = False
    step = 0
    total_reward = 0.0
    wall_contacts = 0
    velocities = []

    print(f"[{idx+1}/6] Rolling out Joint IL+RL on: {m_title} (Seed {seed})...", flush=True)

    while not done and step < max_steps:
        # RL Policy predicts joint 63D action (3D High-Level Steering/Brake + 60D Low-Level Extension Trims)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        
        total_reward += float(reward[0])
        speed = float(np.linalg.norm(raw_env.env.data.qvel[:2]))
        velocities.append(speed)

        if infos[0].get("wall_contact", False):
            wall_contacts += 1

        if step % 2 == 0:
            frames_dual.append(raw_env.render(mode="dual"))
            frames_chase.append(raw_env.render(mode="chase"))

        done = dones[0]
        step += 1

    info = infos[0]
    ball_pos = info.get("ball_xy", raw_env.env.data.qpos[:2])
    final_dist = float(np.linalg.norm(ball_pos - sc.goal[:2]))
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
    vec_env.close()

print("=========================================================================")
print("JOINT IL+RL 63D ACTION SPACE EVALUATION COMPLETE!")
print("=========================================================================")
for r in results:
    print(f"{r['title'][:45]:45s} | Succ={str(r['success']):5s} | GoalDist={r['goal_dist']:4.2f}m | Steps={r['steps']:4d} | WallHits={r['wall_contacts']:3d} ({r['wall_pct']:4.1f}%) | Speed={r['avg_speed']:.2f}m/s")
