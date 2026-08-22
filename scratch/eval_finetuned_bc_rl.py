import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import rootutils
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

rootutils.setup_root("/home/azureuser/telescopic_robot", pythonpath=True)

from radial_sphere import (
    MujocoLowLevelEnv,
    load_config_cli,
    generate_scenario,
)

base_dir = Path("/home/azureuser/telescopic_robot/storage_local")
matching = sorted(base_dir.glob("radial__*__finetune_rl_from_bc__ppo__lowlevel*"))
assert len(matching) > 0, "No fine-tuned BC RL runs found"
exp_dir = matching[-1]

print(f"Evaluating Fine-Tuned BC -> RL Model from: {exp_dir}")

model_path = exp_dir / "checkpoints" / "ppo_final.zip"
norm_path = exp_dir / "checkpoints" / "vecnormalize_final.pkl"
if not model_path.exists():
    ckpts = sorted((exp_dir / "checkpoints").glob("ppo_*_steps.zip"))
    if ckpts:
        model_path = ckpts[-1]
        norm_path = exp_dir / "checkpoints" / f"ppo_vecnormalize_{model_path.stem.split('_')[1]}_steps.pkl"

cfg = load_config_cli(name="maze_level3_large_active_braking_multiaxis")
sc = generate_scenario("maze", cfg, seed=42)

def make_env():
    return MujocoLowLevelEnv(cfg, scenario=sc, randomize=False, max_steps=4000)

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

total_r = 0.0
done = False
step = 0
wall_contacts = 0

print("\nRolling out Fine-Tuned BC -> RL Policy on Large 7x6 Maze (45m path)...", flush=True)

while not done and step < 3000:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = vec_env.step(action)
    total_r += float(reward[0])
    
    if infos[0].get("wall_contact", False):
        wall_contacts += 1
        
    if step % 2 == 0:
        frames_dual.append(raw_env.render(mode="dual"))
        frames_chase.append(raw_env.render(mode="chase"))
        
    done = dones[0]
    step += 1

info = infos[0]
ball_pos = info.get("ball_xy", raw_env.env.data.qpos[:2])
dist = float(np.linalg.norm(ball_pos - sc.goal))
success = bool(dist < 0.50 or info.get("success", False))

print(f"\n[FINE-TUNED BC->RL EVAL COMPLETE] Steps: {step} | Success: {success} | Goal Dist: {dist:.2f}m | Wall Contacts: {wall_contacts} ({wall_contacts/max(step,1)*100:.1f}%) | Reward: {total_r:.2f}")

renders_dir = exp_dir / "renders"
renders_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")

dual_vid = renders_dir / "finetuned_bc_rl_dual.mp4"
chase_vid = renders_dir / "finetuned_bc_rl_chase.mp4"

imageio.mimsave(str(dual_vid), frames_dual, fps=25)
imageio.mimsave(str(chase_vid), frames_chase, fps=25)

thumb = scratch_dir / "finetuned_bc_rl_mid.png"
imageio.imwrite(str(thumb), frames_dual[len(frames_dual)//2])
print(f"Saved evaluation renders to: {renders_dir}")
