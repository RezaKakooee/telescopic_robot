import numpy as np
import mujoco
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
obs, info = env.reset(seed=42)

env.data.qpos[2] = 0.5
mujoco.mj_forward(env.model, env.data)

for step in range(1000):
    targets = np.zeros(60)
    
    # 0.1s period (10 Hz)
    t = step * 0.01
    phase = (t % 0.1) / 0.1
    
    for i, d in enumerate(env.dirs_body):
        if abs(d[1]) < 0.6: continue
        
        if d[2] < -0.2:
            targets[i] = 0.16 if phase < 0.5 else 0.0
        elif abs(d[2]) < 0.2:
            targets[i] = 0.0 if phase < 0.5 else 0.16
            
    env.step(targets)
    if step % 50 == 0:
        pos = env.data.qpos[:3]
        print(f"Step {step:3d}: z={pos[2]:.3f}m, y={pos[1]:.3f}m")

