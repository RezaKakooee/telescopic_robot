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

def inchworm_step(env, t):
    # time period T = 1.0s
    # phase 0 - 0.25: Clamp with horizontal rods. Retract downward rods.
    # phase 0.25 - 0.5: Clamp with downward rods (reach).
    # phase 0.5 - 0.75: Extend downward rods, retract horizontal rods (Push UP)
    # phase 0.75 - 1.0: Clamp with horizontal rods (stabilize)
    
    period = 1.0
    phase = (t % period) / period
    
    targets = np.zeros(60)
    dirs = env.dirs_body
    
    for i, d in enumerate(dirs):
        ux, uy, uz = d
        if abs(uy) < 0.6:
            continue # Only side rods
            
        is_horiz = abs(uz) < 0.2
        is_down = uz < -0.2
        
        if is_horiz:
            # Clamp phase: 0.0 - 0.5 and 0.75 - 1.0
            if phase < 0.5 or phase > 0.75:
                targets[i] = 0.16
            else:
                # Retract slightly to allow moving up
                targets[i] = 0.05
        elif is_down:
            # Downward rods
            if phase < 0.25:
                targets[i] = 0.05 # retracted
            elif phase < 0.5:
                targets[i] = 0.16 # reach
            elif phase < 0.75:
                targets[i] = 0.16 # push (horizontal is retracting, so we push up)
            else:
                targets[i] = 0.05 # start retracting
                
    return targets

print("Starting inchworm climb test...")
for step in range(500):
    t = step * 0.01
    targets = inchworm_step(env, t)
    env.step(targets)
    if step % 25 == 0:
        pos = env.data.qpos[:3]
        print(f"Step {step:3d} (t={t:.2f}s, z={pos[2]:.3f}m, y={pos[1]:.3f}m)")

