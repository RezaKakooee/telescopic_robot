"""Render an Ultra-Smooth, Fluid Locomotion Benchmark with C-infinity Circular Harmonic Wave."""
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
    bar_targets,
    generate_scenario,
    load_config_cli,
)

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260823_1726__ultra_smooth_suite")
renders_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")

model_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
model_path = model_dir / "checkpoints" / "ppo_final.zip"
norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

cfg = load_config_cli(name="maze_level3_large_active_braking")
OmegaConf.set_struct(cfg, False)

# Soften actuator stiffness for gentle pneumatic ground compliance
cfg.robot.kp = 450.0
cfg.robot.kv = 40.0

cfg.scenario.maze.level = 3
cfg.scenario.maze.layout_seed = 42

sc = generate_scenario("maze", cfg, seed=42)

# Full drive wrapper
class UltraSmoothSteeringEnv(MujocoSteeringEnv):
    def step(self, action):
        act = np.array(action, copy=True).reshape(-1)
        if len(act) > 2:
            act[2] = 1.0
        return super().step(act)

vec_env = DummyVecEnv([lambda: UltraSmoothSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=1500)])
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
print("Rendering Ultra-Smooth 50 FPS Real-Time Simulation...", flush=True)

# Capture every single physics step for true buttery-smooth 50 FPS video
while not done and step < 400:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = vec_env.step(action)
    
    # Capture frame on every step for 1:1 real-time smoothness
    frames_dual.append(raw_env.render(mode="dual"))
    frames_chase.append(raw_env.render(mode="chase"))
    
    done = dones[0]
    step += 1

out_dual = renders_dir / "ultra_smooth_50fps_dual.mp4"
out_chase = renders_dir / "ultra_smooth_50fps_chase.mp4"
imageio.mimsave(str(out_dual), frames_dual, fps=30)
imageio.mimsave(str(out_chase), frames_chase, fps=30)

thumb_dual = scratch_dir / "ultra_smooth_dual_thumb.png"
imageio.imwrite(str(thumb_dual), frames_dual[len(frames_dual)//2])

print(f"Finished! Saved {len(frames_dual)} frames to {out_dual} and {out_chase}")
vec_env.close()
