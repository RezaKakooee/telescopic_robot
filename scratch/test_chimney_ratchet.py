import numpy as np
import mujoco
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
obs, info = env.reset(seed=42)

# Start in mid-air
env.data.qpos[2] = 1.0
mujoco.mj_forward(env.model, env.data)

print("Rod directions:")
for i, d in enumerate(env.dirs_body):
    print(f"Rod {i:2d}: {d}")
