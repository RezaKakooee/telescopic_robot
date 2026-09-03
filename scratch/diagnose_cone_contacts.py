import os
os.environ["MUJOCO_GL"] = "egl"
import mujoco
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move

cfg = load_config("configs/rl/training_cones.yaml")
cfg.scenario.spacing = 2.20
scenario = generate_scenario("training_cones", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=1000)
obs, info = env.reset(seed=42)

cones = np.asarray(scenario.cones, dtype=float)
goal = np.asarray(scenario.goal, dtype=float)

contact_geom_names = set()

for step in range(800):
    quat = env.data.qpos[3:7].copy()
    pos = env.data.qpos[0:3].copy()
    vel = env.data.qvel[0:3].copy()

    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        n1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        n2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        if "cone" in n1 or "cone" in n2:
            contact_geom_names.add(f"{n1} <-> {n2}")

    # Weave steering
    x_target = pos[0] + 0.65
    first_cone_x = float(cones[0, 0])
    last_cone_x = float(cones[-1, 0])
    spacing = float(cones[1, 0] - cones[0, 0])
    
    if x_target < first_cone_x - 0.4:
        y_target = 0.70 * 0.5
    elif x_target > last_cone_x + 0.6:
        y_target = 0.0
    else:
        phase = np.pi * (x_target - (first_cone_x - spacing / 2.0)) / spacing
        y_target = 0.75 * np.sin(phase)

    heading_vec = np.array([0.65, (y_target - pos[1]) * 1.5], dtype=np.float64)
    d_hat = heading_vec / np.linalg.norm(heading_vec)

    targets = move(quat, env.dirs_body, env.max_extend, d_hat=d_hat, speed=1.2)
    obs, r, term, trunc, info = env.step(targets)

    if pos[0] >= goal[0]:
        break

env.close()
print(f"Detected cone contact pairs ({len(contact_geom_names)} unique):")
for name in sorted(contact_geom_names):
    print(f"  {name}")
