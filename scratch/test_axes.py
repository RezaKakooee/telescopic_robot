import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
for b in scenario.mocap_bodies:
    print(b["name"], b["pos"], b["geom"]["size"])
