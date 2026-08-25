"""Test Job 0 with full-stroke drive to match 20260821_1154 visual extensions."""
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
    MujocoSteeringEnv,
    generate_scenario,
    load_config_cli,
)

scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")
renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260823_1540__six_lowlevel_jobs_benchmark")

# Load model from 20260821_1154
old_model_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260821_1154__local__train_mujoco_rl__maze__maze_level3_large_fixed__maze_level3_large_fixed")
old_model_path = old_model_dir / "checkpoints" / "ppo_final.zip"
old_norm_path = old_model_dir / "checkpoints" / "vecnormalize_final.pkl"

cfg = load_config_cli(name="maze_level3_large_fixed")
OmegaConf.set_struct(cfg, False)
cfg.scenario.maze.level = 3
cfg.scenario.maze.layout_seed = 7
cfg.rl.include_drive = False

sc = generate_scenario("maze", cfg, seed=7)

def make_env():
    return MujocoSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=1000)

vec_env = DummyVecEnv([make_env])
if old_norm_path.exists():
    vec_env = VecNormalize.load(str(old_norm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

model = PPO.load(str(old_model_path), env=vec_env, device="cpu")

obs = vec_env.reset()
raw_env = vec_env.envs[0]

frames_chase = [raw_env.render(mode="chase")]

done = False
step = 0

print("Running rollout with fixed 1154 model...", flush=True)

while not done and step < 500:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = vec_env.step(action)
    if step % 2 == 0:
        frames_chase.append(raw_env.render(mode="chase"))
    done = dones[0]
    step += 1

out_vid = scratch_dir / "rendered_1154_chase_test.mp4"
imageio.mimsave(str(out_vid), frames_chase, fps=25)
thumb = scratch_dir / "rendered_1154_chase_thumb.png"
imageio.imwrite(str(thumb), frames_chase[len(frames_chase)//2])

print(f"Finished! Saved {len(frames_chase)} frames to {out_vid}")
vec_env.close()
