import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100)

mujoco.mj_resetData(env.model, env.data)
env.data.qpos[0:3] = [0.0, 0.0, 0.25]
env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
env.data.qpos[7:7 + env.n_bars] = 0.02
env.data.ctrl[:] = 0.02
mujoco.mj_forward(env.model, env.data)

print(f"Step 0: z = {env.data.qpos[2]:.4f}")
for s in range(1, 60):
    mujoco.mj_step(env.model, env.data)
    if s % 10 == 0:
        print(f"Substep {s:2d}: z = {env.data.qpos[2]:.4f}, ncon = {env.data.ncon}")
        for c in range(env.data.ncon):
            con = env.data.contact[c]
            g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            print(f"   Contact {c}: {g1} <-> {g2} (dist={con.dist:.4f}, pos={con.pos})")

env.close()
