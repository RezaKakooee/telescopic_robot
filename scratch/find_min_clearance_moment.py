import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.slalom import slalom
from scripts.skills.run_training_cones import _cone_contact

cfg = load_config("configs/rl/training_cones.yaml")
scenario = generate_scenario("training_cones", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=2000)
obs, info = env.reset(seed=42)

cones = np.asarray(scenario.cones, dtype=float)
goal = np.asarray(scenario.goal, dtype=float)

closest_info = []

for step in range(1500):
    pos = env.data.qpos[0:3].copy()
    quat = env.data.qpos[3:7].copy()
    vel = env.data.qvel[0:3].copy()

    for ci, c in enumerate(cones):
        dist = float(np.linalg.norm(pos[:2] - c[:2]))
        if dist < 0.40:
            closest_info.append((step, ci, float(c[0]), float(pos[0]), float(pos[1]), dist, _cone_contact(env)))

    targets = slalom(quat, env.dirs_body, env.max_extend, ball_xy=pos[:2], cones=cones, speed=1.4)
    obs, r, term, trunc, info = env.step(targets)
    if pos[0] >= goal[0]:
        break

env.close()
print(f"Moments with dist < 0.40m ({len(closest_info)} steps):")
for item in closest_info[:20]:
    print(f"  step {item[0]:4d}: cone {item[1]} (at x={item[2]:.1f}) -> ball at x={item[3]:.2f}, y={item[4]:.2f}, dist={item[5]:.3f}m, contacts={item[6]}")
