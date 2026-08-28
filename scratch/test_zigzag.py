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

print("Testing Zig-Zag Jump...")

state = "fall"
timer = 0
targets = np.zeros(60)

for step in range(300):
    pos = env.data.qpos[:3].copy()
    vel = env.data.qvel[:3].copy()
    
    if step == 0:
        state = "push_left"
        timer = 0
        
    targets[:] = 0.05 # default retract
    
    if state == "push_left":
        # push against right wall to move left
        for i, d in enumerate(env.dirs_body):
            if d[1] > 0.5 and abs(d[2]) < 0.3:
                targets[i] = 0.16
        if pos[1] < -0.05:
            state = "jump_up_right"
            timer = 0
            
    elif state == "jump_up_right":
        # we are on left wall. push down-left rods to jump up-right
        for i, d in enumerate(env.dirs_body):
            if d[1] < -0.5 and d[2] < -0.2:
                targets[i] = 0.16
        timer += 1
        if timer > 15 and pos[1] > 0.05:
            state = "jump_up_left"
            timer = 0
            
    elif state == "jump_up_left":
        # we are on right wall. push down-right rods to jump up-left
        for i, d in enumerate(env.dirs_body):
            if d[1] > 0.5 and d[2] < -0.2:
                targets[i] = 0.16
        timer += 1
        if timer > 15 and pos[1] < -0.05:
            state = "jump_up_right"
            timer = 0
            
    env.step(targets)
    if step % 10 == 0:
        print(f"Step {step:3d} [{state:15s}]: z={pos[2]:.3f}m, y={pos[1]:.3f}m, vz={vel[2]:.3f}m/s")

