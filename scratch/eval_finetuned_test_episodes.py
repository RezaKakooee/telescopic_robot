"""Evaluate Fine-Tuned BC->RL policy across the 3 procedural test mazes."""
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

base_dir = Path("/home/azureuser/telescopic_robot/storage_local")
matching = sorted(base_dir.glob("radial__*__finetune_rl_from_bc__ppo__lowlevel*"))
assert len(matching) > 0, "No fine-tuned BC RL runs found"
exp_dir = matching[-1]

model_path = exp_dir / "checkpoints" / "ppo_final.zip"
norm_path = exp_dir / "checkpoints" / "vecnormalize_final.pkl"

scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")
renders_dir = exp_dir / "renders"
renders_dir.mkdir(parents=True, exist_ok=True)

test_scenarios = [
    {"name": "Test Ep 1: Level 1 Orthogonal Maze", "level": 1, "seed": 9901},
    {"name": "Test Ep 2: Level 2 Multi-Loop Braid Maze", "level": 2, "seed": 9902},
    {"name": "Test Ep 3: Level 3 Twisty Tree Maze", "level": 3, "seed": 9903},
]

results = []

for idx, sc_info in enumerate(test_scenarios):
    level = sc_info["level"]
    seed = sc_info["seed"]
    
    cfg = load_config_cli(name="maze_level3_large_active_braking_multiaxis")
    OmegaConf.set_struct(cfg, False)
    if hasattr(cfg.scenario, "maze") and cfg.scenario.maze is not None:
        cfg.scenario.maze.level = level
        cfg.scenario.maze.random_endpoints = True
        cfg.scenario.maze.endpoint_min_route = 6.0

    sc = generate_scenario("maze", cfg, seed=seed)

    def make_env():
        return MujocoLowLevelEnv(cfg, scenario=sc, randomize=False, max_steps=2000)

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
    
    print(f"\nEvaluating Fine-Tuned Policy on {sc_info['name']} (Seed {seed})...", flush=True)

    while not done and step < 1500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        total_reward += float(reward[0])

        if infos[0].get("wall_contact", False):
            wall_contacts += 1

        if step % 2 == 0:
            frames_dual.append(raw_env.render(mode="dual"))
            frames_chase.append(raw_env.render(mode="chase"))

        done = dones[0]
        step += 1

    info = infos[0]
    ball_pos = info.get("ball_xy", raw_env.env.data.qpos[:2])
    dist = float(np.linalg.norm(ball_pos - sc.goal[:2]))
    success = bool(dist < 0.50 or info.get("success", False))

    vid_dual = renders_dir / f"finetuned_test_ep_{idx+1}_dual.mp4"
    vid_chase = renders_dir / f"finetuned_test_ep_{idx+1}_chase.mp4"
    imageio.mimsave(str(vid_dual), frames_dual, fps=25)
    imageio.mimsave(str(vid_chase), frames_chase, fps=25)

    thumb = scratch_dir / f"finetuned_test_ep_{idx+1}_mid.png"
    imageio.imwrite(str(thumb), frames_dual[len(frames_dual)//2])

    results.append({
        "name": sc_info["name"],
        "level": level,
        "seed": seed,
        "steps": step,
        "success": success,
        "goal_dist": dist,
        "wall_contacts": wall_contacts,
        "wall_pct": wall_contacts / max(step, 1) * 100.0,
        "total_reward": total_reward,
        "dual_video": str(vid_dual),
        "chase_video": str(vid_chase),
        "thumb": str(thumb),
    })

    print(f"[{sc_info['name']}] Steps={step} | Success={success} | GoalDist={dist:.2f}m | WallHits={wall_contacts} ({wall_contacts/max(step,1)*100:.1f}%) | Rew={total_reward:.2f}")
    vec_env.close()

print("\n=========================================================================")
print("FINE-TUNED BC->RL TEST EVALUATION RESULTS")
print("=========================================================================")
for r in results:
    print(f"{r['name']:35s} | Success={str(r['success']):5s} | GoalDist={r['goal_dist']:4.2f}m | Steps={r['steps']:4d} | WallHits={r['wall_contacts']:3d} ({r['wall_pct']:4.1f}%) | Rew={r['total_reward']:7.2f}")
