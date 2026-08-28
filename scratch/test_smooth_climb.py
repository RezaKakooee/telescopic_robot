import numpy as np
import mujoco
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
obs, info = env.reset(seed=42)

env.data.qpos[1] = -0.05
env.data.qpos[2] = 0.5
mujoco.mj_forward(env.model, env.data)

targets = np.zeros(60)
max_z = 0.5

print("Testing Smooth Inchworm Climb...")

for step in range(3000):
    pos = env.data.qpos[:3].copy()
    vel = env.data.qvel[:3].copy()
    max_z = max(max_z, pos[2])
    
    # CHEAT: Zero out X and orientation for pure 2D analysis
    env.data.qpos[0] = 0.0
    env.data.qvel[0] = 0.0
    env.data.qpos[3:7] = [1, 0, 0, 0]
    env.data.qvel[3:6] = [0, 0, 0]
    
    # Smooth Inchworm Phase: Period = 2.0 seconds (200 steps)
    t = step * 0.01
    phase = (t % 2.0) / 2.0
    
    targets[:] = 0.01 # Baseline tuck
    
    # Identify horizontal and downward rods that point strongly in +/- Y
    # And have very little X component to minimize twist
    left_horiz = []
    right_horiz = []
    left_down = []
    right_down = []
    
    for i, d in enumerate(env.dirs_body):
        if abs(d[0]) > 0.3: continue # Skip rods with large X component
        
        if d[1] < -0.6 and abs(d[2]) < 0.3:
            right_horiz.append((i, d)) # -Y is right side? Wait. +Y is left box.
        elif d[1] > 0.6 and abs(d[2]) < 0.3:
            left_horiz.append((i, d))
            
        if d[1] < -0.5 and d[2] < -0.4:
            right_down.append((i, d))
        elif d[1] > 0.5 and d[2] < -0.4:
            left_down.append((i, d))
            
    # Phase 0.0 - 0.4: PUSH UP
    # Extend downward rods from 0.05 to 0.20
    # Retract horizontal rods to 0.01
    
    # Phase 0.4 - 0.5: CLAMP
    # Downward rods stay at 0.20
    # Extend horizontal rods to 0.20
    
    # Phase 0.5 - 0.9: RETRACT
    # Horizontal rods stay at 0.20
    # Retract downward rods from 0.20 to 0.05
    
    # Phase 0.9 - 1.0: UNCLAMP
    # Horizontal rods retract to 0.01
    # Downward rods stay at 0.05
    
    down_ext = 0.05
    horiz_ext = 0.05
    
    if phase < 0.4:
        # PUSH UP
        prog = phase / 0.4
        down_ext = 0.05 + 0.15 * prog
        horiz_ext = 0.01
    elif phase < 0.5:
        # CLAMP
        prog = (phase - 0.4) / 0.1
        down_ext = 0.20
        horiz_ext = 0.01 + 0.19 * prog
    elif phase < 0.9:
        # RETRACT
        prog = (phase - 0.5) / 0.4
        down_ext = 0.20 - 0.15 * prog
        horiz_ext = 0.20
    else:
        # UNCLAMP
        prog = (phase - 0.9) / 0.1
        down_ext = 0.05
        horiz_ext = 0.20 - 0.19 * prog
        
    for i, d in left_down + right_down:
        targets[i] = down_ext
        
    for i, d in left_horiz + right_horiz:
        targets[i] = horiz_ext
        
    env.step(targets)
    if step % 200 == 0:
        print(f"Step {step:4d} [ph={phase:.2f}]: z={pos[2]:.3f}m, y={pos[1]:.3f}m")

print(f"Max Z achieved: {max_z:.3f}m (Net climb: {max_z - 0.5:.3f}m)")
