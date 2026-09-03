import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)

print("Searching for rods with |d_y| > 0.571 AND |d_z| > |d_y|...")
found = 0
for i, d in enumerate(env.dirs_body):
    if abs(d[1]) > 0.571 and abs(d[2]) > abs(d[1]):
        print(f"Rod {i:2d}: dx={d[0]:.3f}, dy={d[1]:.3f}, dz={d[2]:.3f} (Lift: {abs(d[2]) - abs(d[1]):.3f})")
        found += 1
print(f"Found {found} rods.")
