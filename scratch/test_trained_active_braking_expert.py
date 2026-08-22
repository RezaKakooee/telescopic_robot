import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import rootutils
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

rootutils.setup_root("/home/azureuser/telescopic_robot", pythonpath=True)

from radial_sphere import (
    MujocoSteeringEnv,
    generate_scenario,
    load_config_cli,
)

model_dir = Path("/home/azureuser/telescopic_robot/storage_local/radial__20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
model_path = model_dir / "checkpoints" / "ppo_final.zip"
norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

cfg = load_config_cli(name="maze_level3_large_active_braking")
OmegaConf.set_struct(cfg, False)
cfg.scenario.maze.level = 1
cfg.scenario.maze.random_endpoints = True
cfg.scenario.maze.endpoint_min_route = 6.0

sc = generate_scenario("maze", cfg, seed=9901)

def make_env():
    return MujocoSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=2000)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load(str(norm_path), vec_env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load(str(model_path), env=vec_env, device="cpu")

obs = vec_env.reset()
raw_env = vec_env.envs[0]

done = False
step = 0
wall_contacts = 0
total_r = 0.0

print("Testing Trained Active-Braking RL Model on Level 1 Orthogonal Maze (Seed 9901)...", flush=True)

while not done and step < 1500:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = vec_env.step(action)
    total_r += float(reward[0])
    if infos[0].get("wall_contact", False):
        wall_contacts += 1
    done = dones[0]
    step += 1

ball_pos = infos[0].get("ball_xy", raw_env.env.data.qpos[:2])
dist = float(np.linalg.norm(ball_pos - sc.goal))
success = bool(dist < 0.50 or infos[0].get("success", False))

print(f"\n[TRAINED ACTIVE-BRAKING EXPERT RESULT]")
print(f"Steps: {step} | Success: {success} | GoalDist: {dist:.2f}m | WallHits: {wall_contacts} ({wall_contacts/max(step,1)*100:.1f}%) | Reward: {total_r:.2f}")
vec_env.close()
