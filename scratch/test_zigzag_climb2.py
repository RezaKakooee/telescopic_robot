import numpy as np
import mujoco
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
obs, info = env.reset(seed=42)

# Start in mid-air, slightly off-center
env.data.qpos[1] = -0.05
env.data.qpos[2] = 1.0
mujoco.mj_forward(env.model, env.data)

print("Testing Zig-Zag Climb 2...")

state = "jump_up_right"
timer = 0
targets = np.zeros(60)

max_z = 1.0

for step in range(500):
    pos = env.data.qpos[:3].copy()
    vel = env.data.qvel[:3].copy()
    
    max_z = max(max_z, pos[2])
    
    targets[:] = 0.0 # FULLY retract all rods!
    
    if state == "jump_up_right":
        # we are on left wall (y < 0). push down-left rods to jump up-right
        for i, d in enumerate(env.dirs_body):
            if d[1] < -0.3 and d[2] < -0.2:
                targets[i] = 0.16
        timer += 1
        if timer > 8 and pos[1] > 0.02:
            state = "jump_up_left"
            timer = 0
            
    elif state == "jump_up_left":
        # we are on right wall (y > 0). push down-right rods to jump up-left
        for i, d in enumerate(env.dirs_body):
            if d[1] > 0.3 and d[2] < -0.2:
                targets[i] = 0.16
        timer += 1
        if timer > 8 and pos[1] < -0.02:
            state = "jump_up_right"
            timer = 0
            
    env.step(targets)
    if step % 20 == 0:
        print(f"Step {step:3d} [{state:15s}]: z={pos[2]:.3f}m, y={pos[1]:.3f}m, vz={vel[2]:.3f}m/s")

print(f"Max Z achieved: {max_z:.3f}m (Net climb: {max_z - 1.0:.3f}m)")

