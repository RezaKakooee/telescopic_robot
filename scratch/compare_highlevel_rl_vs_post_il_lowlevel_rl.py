"""Rigorous Head-to-Head Benchmark: High-Level RL Only vs. Post-IL Joint 60D Low-Level RL."""
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
    MujocoSteeringEnv,
    generate_scenario,
    load_config_cli,
)

scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")
renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/comparison_highlevel_vs_post_il_lowlevel")
renders_dir.mkdir(parents=True, exist_ok=True)

# 1. Candidate 1: High-Level RL Only (Active Braking Steering)
highlevel_dir = Path("/home/azureuser/telescopic_robot/storage_local/radial__20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
hl_model_path = highlevel_dir / "checkpoints" / "ppo_final.zip"
hl_norm_path = highlevel_dir / "checkpoints" / "vecnormalize_final.pkl"

# 2. Candidate 2: Post-IL Joint 60D Low-Level + High-Level RL Policy
post_il_dir = Path("/home/azureuser/telescopic_robot/storage_local/radial__20260822_1619__local__finetune_rl_from_bc__ppo__lowlevel__maze_level3_large_active_braking_multiaxis")
post_il_model_path = post_il_dir / "checkpoints" / "ppo_final.zip"
post_il_norm_path = post_il_dir / "checkpoints" / "vecnormalize_final.pkl"

test_mazes = [
    {
        "id": "ep1_level1_spiral",
        "title": "Level 1: Orthogonal Spiral Maze",
        "level": 1,
        "seed": 10101,
        "max_steps": 1200,
    },
    {
        "id": "ep2_level2_braid",
        "title": "Level 2: Multi-Loop Braid Maze",
        "level": 2,
        "seed": 20202,
        "max_steps": 1200,
    },
    {
        "id": "ep3_level3_tree",
        "title": "Level 3: Deep Branching Tree Maze",
        "level": 3,
        "seed": 30303,
        "max_steps": 1500,
    },
    {
        "id": "ep4_large_45m",
        "title": "Level 3: Large 7x6 45-Meter Gauntlet",
        "level": 3,
        "seed": 50505,
        "max_steps": 2500,
    },
]

candidates = [
    {
        "id": "highlevel_rl",
        "name": "High-Level RL Only (3D Action Space)",
        "model_path": hl_model_path,
        "norm_path": hl_norm_path,
        "env_type": "steering",
        "cfg_name": "maze_level3_large_active_braking",
    },
    {
        "id": "post_il_lowlevel_rl",
        "name": "Post-IL Joint 60D Low-Level RL (63D Action Space)",
        "model_path": post_il_model_path,
        "norm_path": post_il_norm_path,
        "env_type": "lowlevel",
        "cfg_name": "maze_level3_large_active_braking_multiaxis",
    },
]

results = {c["id"]: [] for c in candidates}

print("=========================================================================================")
print("STARTING RIGOROUS COMPARISON: HIGH-LEVEL RL vs. POST-IL JOINT 60D LOW-LEVEL RL")
print("=========================================================================================\n")

for m_idx, m_info in enumerate(test_mazes):
    print(f"\n--- Benchmark Scenario [{m_idx+1}/4]: {m_info['title']} (Seed {m_info['seed']}) ---")
    
    for cand in candidates:
        c_id = cand["id"]
        c_name = cand["name"]
        
        cfg = load_config_cli(name=cand["cfg_name"])
        OmegaConf.set_struct(cfg, False)
        if hasattr(cfg.scenario, "maze") and cfg.scenario.maze is not None:
            cfg.scenario.maze.level = m_info["level"]
            cfg.scenario.maze.random_endpoints = True
            cfg.scenario.maze.endpoint_min_route = 5.0
            
        sc = generate_scenario("maze", cfg, seed=m_info["seed"])
        
        if cand["env_type"] == "steering":
            def make_env():
                return MujocoSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=m_info["max_steps"])
        else:
            def make_env():
                return MujocoLowLevelEnv(cfg, scenario=sc, randomize=False, max_steps=m_info["max_steps"])
                
        vec_env = DummyVecEnv([make_env])
        if cand["norm_path"].exists():
            vec_env = VecNormalize.load(str(cand["norm_path"]), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            
        model = PPO.load(str(cand["model_path"]), env=vec_env, device="cpu")
        
        obs = vec_env.reset()
        raw_env = vec_env.envs[0]
        
        frames_dual = [raw_env.render(mode="dual")]
        frames_chase = [raw_env.render(mode="chase")]
        
        done = False
        step = 0
        total_r = 0.0
        wall_contacts = 0
        velocities = []
        actuator_efforts = []
        prev_joints = None
        
        while not done and step < m_info["max_steps"]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            total_r += float(reward[0])
            
            raw = raw_env.env if hasattr(raw_env, "env") else raw_env
            curr_joints = raw.data.qpos[7:7 + raw.n_bars].copy()
            if prev_joints is not None:
                actuator_efforts.append(float(np.sum(np.abs(curr_joints - prev_joints))))
            prev_joints = curr_joints
            
            speed = float(np.linalg.norm(raw.data.qvel[:2]))
            velocities.append(speed)
            
            if infos[0].get("wall_contact", False):
                wall_contacts += 1
                
            if step % 2 == 0:
                frames_dual.append(raw_env.render(mode="dual"))
                frames_chase.append(raw_env.render(mode="chase"))
                
            done = dones[0]
            step += 1
            
        info = infos[0]
        ball_pos = info.get("ball_xy", raw.data.qpos[:2])
        final_dist = float(np.linalg.norm(ball_pos - sc.goal[:2]))
        success = bool(final_dist < 0.50 or info.get("success", False))
        avg_speed = float(np.mean(velocities)) if velocities else 0.0
        peak_speed = float(np.max(velocities)) if velocities else 0.0
        total_actuation_effort = float(np.sum(actuator_efforts)) if actuator_efforts else 0.0
        
        vid_dual = renders_dir / f"{m_info['id']}_{c_id}_dual.mp4"
        vid_chase = renders_dir / f"{m_info['id']}_{c_id}_chase.mp4"
        imageio.mimsave(str(vid_dual), frames_dual, fps=25)
        imageio.mimsave(str(vid_chase), frames_chase, fps=25)
        
        thumb = scratch_dir / f"{m_info['id']}_{c_id}_mid.png"
        imageio.imwrite(str(thumb), frames_dual[len(frames_dual)//2])
        
        res = {
            "maze_id": m_info["id"],
            "maze_title": m_info["title"],
            "candidate_id": c_id,
            "candidate_name": c_name,
            "steps": step,
            "success": success,
            "final_dist": final_dist,
            "wall_contacts": wall_contacts,
            "wall_pct": wall_contacts / max(step, 1) * 100.0,
            "avg_speed": avg_speed,
            "peak_speed": peak_speed,
            "total_effort": total_actuation_effort,
            "total_reward": total_r,
            "dual_video": str(vid_dual),
            "chase_video": str(vid_chase),
            "thumb": str(thumb),
        }
        results[c_id].append(res)
        
        print(f"  [{c_name}]")
        print(f"    -> Steps: {step} | Success: {success} | GoalDist: {final_dist:.2f}m | WallHits: {wall_contacts} ({res['wall_pct']:.1f}%) | AvgSpeed: {avg_speed:.2f}m/s | PeakSpeed: {peak_speed:.2f}m/s | Effort: {total_actuation_effort:.1f}")
        vec_env.close()

print("\n=========================================================================================")
print("BENCHMARK SUMMARY COMPARISON TABLE")
print("=========================================================================================")
print(f"{'Maze Scenario':35s} | {'High-Level RL Only (3D)':32s} | {'Post-IL Joint 60D RL (63D)':32s}")
print("-" * 105)
for i in range(len(test_mazes)):
    hl = results["highlevel_rl"][i]
    ll = results["post_il_lowlevel_rl"][i]
    hl_str = f"Succ={str(hl['success']):5s} | Hits={hl['wall_contacts']:3d} ({hl['wall_pct']:4.1f}%) | {hl['avg_speed']:.2f}m/s"
    ll_str = f"Succ={str(ll['success']):5s} | Hits={ll['wall_contacts']:3d} ({ll['wall_pct']:4.1f}%) | {ll['avg_speed']:.2f}m/s"
    print(f"{hl['maze_title'][:35]:35s} | {hl_str:32s} | {ll_str:32s}")
