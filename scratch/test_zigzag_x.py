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
timer = 0

for step in range(2000):
    pos = env.data.qpos[:3].copy()
    vel = env.data.qvel[:3].copy()
    max_z = max(max_z, pos[2])
    
    targets[:] = 0.0
    
    # Simple X stabilization: if x > 0.1, push back slightly with forward rods
    if pos[0] > 0.1:
        for i, d in enumerate(env.dirs_body):
            if d[0] > 0.5 and abs(d[2]) < 0.2:
                targets[i] = 0.05
    elif pos[0] < -0.1:
        for i, d in enumerate(env.dirs_body):
            if d[0] < -0.5 and abs(d[2]) < 0.2:
                targets[i] = 0.05
    
    if state == "jump_up_right":
        for i, d in enumerate(env.dirs_body):
            if d[1] < -0.2 and d[2] < -0.2:
                targets[i] = 0.16
        timer += 1
        if timer > 4 and vel[1] > 0.4:
            state = "fly_right"
            timer = 0
            
    elif state == "fly_right":
        if pos[1] > 0.03:
            state = "jump_up_left"
            timer = 0
            
    elif state == "jump_up_left":
        for i, d in enumerate(env.dirs_body):
            if d[1] > 0.2 and d[2] < -0.2:
                targets[i] = 0.16
        timer += 1
        if timer > 4 and vel[1] < -0.4:
            state = "fly_left"
            timer = 0
            
    elif state == "fly_left":
        if pos[1] < -0.03:
            state = "jump_up_right"
            timer = 0
            
    env.step(targets)
    if step % 100 == 0:
        print(f"Step {step:4d} [{state:15s}]: z={pos[2]:.3f}m, y={pos[1]:.3f}m, x={pos[0]:.3f}m")

print(f"Max Z achieved: {max_z:.3f}m (Net climb: {max_z - 0.5:.3f}m)")
