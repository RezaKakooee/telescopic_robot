import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
import mujoco

cfg = load_config("configs/rl/terrain_transparent_glass_pipe.yaml")
scenario = generate_scenario("glass_pipe", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100)
obs, info = env.reset(seed=42)

for step in range(10):
    targets = np.zeros(env.n_bars)
    env.step(targets)

print(f"Number of contacts: {env.data.ncon}")
for ci in range(min(env.data.ncon, 20)):
    con = env.data.contact[ci]
    g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
    g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
    print(f"Contact {ci}: {g1} <-> {g2}, dist={con.dist:.4f}, pos={con.pos}")
