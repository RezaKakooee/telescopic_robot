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

print("Testing Wall Jump 2 (No Unclamp)...")

targets = np.zeros(60)

for step in range(200):
    
    # Phase 0-50: Clamp to stabilize
    if step < 50:
        for i, d in enumerate(env.dirs_body):
            if abs(d[1]) > 0.6 and abs(d[2]) < 0.2:
                targets[i] = 0.16
    
    # Phase 50-60: VIOLENTLY push downward rods, but DO NOT retract clamp yet!
    elif step < 60:
        for i, d in enumerate(env.dirs_body):
            if abs(d[1]) > 0.6 and d[2] < -0.2:
                targets[i] = 0.16
                
    # Phase 60-70: Retract clamp (let the downward rods propel it)
    elif step < 70:
        for i, d in enumerate(env.dirs_body):
            if abs(d[1]) > 0.6 and abs(d[2]) < 0.2:
                targets[i] = 0.0 # retract horizontal
                
    # Phase 70-150: Clamp again, retract downward
    else:
        for i, d in enumerate(env.dirs_body):
            if abs(d[1]) > 0.6 and abs(d[2]) < 0.2:
                targets[i] = 0.16
            if abs(d[1]) > 0.6 and d[2] < -0.2:
                targets[i] = 0.0
                
    env.step(targets)
    if step % 10 == 0:
        pos = env.data.qpos[:3]
        vel = env.data.qvel[:3]
        print(f"Step {step:3d}: z={pos[2]:.3f}m, vz={vel[2]:.3f}m/s")

