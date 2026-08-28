import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)

for i, d in enumerate(env.dirs_body):
    if d[1] < -0.2 and d[2] < -0.2:
        print(f"Left Down Rod {i}: {d}")
    if d[1] > 0.2 and d[2] < -0.2:
        print(f"Right Down Rod {i}: {d}")
