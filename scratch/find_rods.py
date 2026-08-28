import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)

print("Downward rods for W=0.32 (W/2=0.16):")
for i, d in enumerate(env.dirs_body):
    uy, uz = d[1], d[2]
    if uz < -0.2:
        if abs(uy) >= 0.16 / 0.31:
            tan_theta = abs(uz) / abs(uy)
            print(f"Rod {i:2d}: uy={uy:+.3f}, uz={uz:+.3f}, tan={tan_theta:.3f} {'SLIP' if tan_theta > 1.0 else 'STICK'}")
