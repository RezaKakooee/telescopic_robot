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
env.data.qpos[2] = 4.0
mujoco.mj_forward(env.model, env.data)

state = "jump_up_right"
targets = np.zeros(60)
timer = 0

for step in range(3000):
    pos = env.data.qpos[:3].copy()
    vel = env.data.qvel[:3].copy()
    
    # CHEAT: Zero out X and orientation
    env.data.qpos[0] = 0.0
    env.data.qvel[0] = 0.0
    env.data.qpos[3:7] = [1, 0, 0, 0] # Reset quaternion
    env.data.qvel[3:6] = [0, 0, 0] # Reset angular velocity
    
    targets[:] = 0.0
    
    if state == "jump_up_right":
        for i, d in enumerate(env.dirs_body):
            if d[1] < -0.2 and d[2] < -0.2:
                targets[i] = 0.10
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
                targets[i] = 0.10
        timer += 1
        if timer > 4 and vel[1] < -0.4:
            state = "fly_left"
            timer = 0
            
    elif state == "fly_left":
        if pos[1] < -0.03:
            state = "jump_up_right"
            timer = 0
            
    env.step(targets)
    if step % 200 == 0:
        print(f"Step {step:4d} [{state:15s}]: z={pos[2]:.3f}m, y={pos[1]:.3f}m")

print(f"Final Z: {pos[2]:.3f}m")
