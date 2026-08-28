"""Inspect contacts in chimney scenario."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
env.reset(seed=42)

print("Number of geoms:", env.model.ngeom)
for g in range(env.model.ngeom):
    name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g)
    pos = env.model.geom_pos[g]
    size = env.model.geom_size[g]
    gtype = env.model.geom_type[g]
    print(f"Geom {g:2d}: {name:25s} | type={gtype} | pos={pos} | size={size}")

env.close()
