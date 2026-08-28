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
env.data.qpos[2] = 0.5
mujoco.mj_forward(env.model, env.data)

for step in range(500):
    targets = np.zeros(60)
    
    # 0.5s period
    t = step * 0.01
    phase = (t % 0.5) / 0.5
    
    # We want to climb UP. 
    # Phase 0-0.5: clamped with horizontal, retract downward.
    # Phase 0.5-1.0: clamped with downward, retract horizontal.
    
    for i, d in enumerate(env.dirs_body):
        if abs(d[1]) < 0.6: continue # only side rods
        
        is_horiz = abs(d[2]) < 0.2
        is_down = d[2] < -0.2
        
        if is_horiz:
            if phase < 0.5:
                targets[i] = 0.16 # clamp
            else:
                targets[i] = 0.0 # retract
        elif is_down:
            if phase < 0.5:
                targets[i] = 0.0 # retract
            else:
                targets[i] = 0.16 # clamp & push
                
    env.step(targets)
    if step % 25 == 0:
        pos = env.data.qpos[:3]
        print(f"Step {step:3d} [ph={phase:.2f}]: z={pos[2]:.3f}m, y={pos[1]:.3f}m")

