"""Combined High-Performance Smooth Navigation: Force Compliance + Actuator Slew Rate + Steadicam."""
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

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260823_1732__perfect_smooth_suite")
renders_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")

model_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
model_path = model_dir / "checkpoints" / "ppo_final.zip"
norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

cfg = load_config_cli(name="maze_level3_large_active_braking")
OmegaConf.set_struct(cfg, False)

# Set the optimal smooth navigation parameters
cfg.controller.enable_contact_compliance = True
cfg.controller.compliance_gain = 0.0006
cfg.controller.max_contact_force = 35.0

cfg.controller.enable_actuator_slew_rate = True
cfg.controller.actuator_max_vel = 0.40

cfg.controller.enable_camber_banking = True
cfg.controller.camber_bank_gain = 0.025

cfg.scenario.maze.level = 3
cfg.scenario.maze.layout_seed = 42

sc = generate_scenario("maze", cfg, seed=42)

class SmoothNavEnv(MujocoSteeringEnv):
    def step(self, action):
        act = np.array(action, copy=True).reshape(-1)
        if len(act) > 2:
            act[2] = 1.0  # Full drive for full visible trailing extensions
        return super().step(act)

vec_env = DummyVecEnv([lambda: SmoothNavEnv(cfg, scenario=sc, randomize=False, max_steps=1500)])
if norm_path.exists():
    vec_env = VecNormalize.load(str(norm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

model = PPO.load(str(model_path), env=vec_env, device="cpu")

obs = vec_env.reset()
raw_env = vec_env.envs[0]

frames_dual = []
frames_chase = []

done = False
step = 0
wall_contacts = 0
print("Running Combined High-Performance Smooth Navigation...", flush=True)

while not done and step < 500:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = vec_env.step(action)
    
    if infos[0].get("wall_contact", False):
        wall_contacts += 1
        
    frames_dual.append(raw_env.render(mode="dual"))
    frames_chase.append(raw_env.render(mode="chase"))
    
    done = dones[0]
    step += 1

success = infos[0].get("success", False) or np.linalg.norm(raw_env.env.data.qpos[:2] - sc.goal[:2]) < 0.50
print(f"Navigation Complete! Success={success} | Steps={step} | Wall Contacts={wall_contacts} ({wall_contacts/step*100:.1f}%)")

out_dual = renders_dir / "perfect_smooth_navigation_dual.mp4"
out_chase = renders_dir / "perfect_smooth_navigation_chase.mp4"
imageio.mimsave(str(out_dual), frames_dual, fps=30)
imageio.mimsave(str(out_chase), frames_chase, fps=30)

thumb_dual = scratch_dir / "perfect_smooth_dual_thumb.png"
thumb_end = scratch_dir / "perfect_smooth_dual_end_thumb.png"
imageio.imwrite(str(thumb_dual), frames_dual[len(frames_dual)//2])
imageio.imwrite(str(thumb_end), frames_dual[-1])

print(f"Saved {len(frames_dual)} frames to {out_dual}")
vec_env.close()
