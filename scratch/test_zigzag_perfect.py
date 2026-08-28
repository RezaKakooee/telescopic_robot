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

state = "jump_up_right"
max_z = 0.5
targets = np.zeros(60)

for step in range(2000):
    pos = env.data.qpos[:3]
    vel = env.data.qvel[:3]
    max_z = max(max_z, pos[2])
    
    targets[:] = 0.0
    
    if state == "jump_up_right":
        # We are at left wall (y < 0). Push down-left rods to jump up and right.
        for i, d in enumerate(env.dirs_body):
            if d[1] < -0.3 and d[2] < -0.2:
                targets[i] = 0.16
        # Switch when we have enough rightward velocity AND we are moving away
        if vel[1] > 0.8 and pos[1] > -0.02:
            state = "fly_right"
            
    elif state == "fly_right":
        # Flying towards right wall. Keep rods retracted to avoid snagging.
        # Wait until we hit the right wall (velocity reverses or we go past y=0.04)
        if pos[1] > 0.04 and vel[1] < 0.2:
            state = "jump_up_left"
            
    elif state == "jump_up_left":
        # We are at right wall (y > 0). Push down-right rods to jump up and left.
        for i, d in enumerate(env.dirs_body):
            if d[1] > 0.3 and d[2] < -0.2:
                targets[i] = 0.16
        if vel[1] < -0.8 and pos[1] < 0.02:
            state = "fly_left"
            
    elif state == "fly_left":
        if pos[1] < -0.04 and vel[1] > -0.2:
            state = "jump_up_right"
            
    env.step(targets)
    if step % 100 == 0:
        print(f"Step {step:4d} [{state:15s}]: z={pos[2]:.3f}m, y={pos[1]:.3f}m, vy={vel[1]:.3f}, vz={vel[2]:.3f}")
        
    if pos[2] < 0.1:
        print("Fell!")
        break

print(f"Max Z achieved: {max_z:.3f}m")
