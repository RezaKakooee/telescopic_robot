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

print("Testing Narrow Slipping Inchworm...")

max_z = 0.5

for step in range(2000):
    targets = np.zeros(60)
    
    # 0.5s period
    t = step * 0.01
    phase = (t % 0.5) / 0.5
    
    pos = env.data.qpos[:3]
    max_z = max(max_z, pos[2])
    
    # Rod 51: uy = 0.614, uz = -0.717 (Right down, slip capable!)
    # Left counterpart? 
    # Let's find all steep downward rods that can reach the wall (|uy| > 0.16/0.31 = 0.516)
    for i, d in enumerate(env.dirs_body):
        # We need rods pointing sideways and down
        if abs(d[1]) < 0.5: continue
        
        is_horiz = abs(d[2]) < 0.2
        is_down = d[2] < -0.2
        
        # If it's too steep (uy < 0.516), it can't reach the wall, but we filtered by abs(d[1]) > 0.5
        
        if is_horiz:
            # Clamp phase: 0.0 to 0.5
            if phase < 0.5:
                targets[i] = 0.16
            else:
                targets[i] = 0.05
        elif is_down:
            # Push phase: 0.5 to 1.0
            if phase < 0.5:
                targets[i] = 0.05
            else:
                targets[i] = 0.16
                
    env.step(targets)
    if step % 100 == 0:
        print(f"Step {step:4d} [ph={phase:.2f}]: z={pos[2]:.3f}m, y={pos[1]:.3f}m, x={pos[0]:.3f}m")
        
print(f"Max Z achieved: {max_z:.3f}m (Net climb: {max_z - 0.5:.3f}m)")
