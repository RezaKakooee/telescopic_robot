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
env.data.qpos[2] = 0.5
mujoco.mj_forward(env.model, env.data)

print("Testing Zig-Zag Climb 3...")

state = "jump_up_right"
timer = 0
targets = np.zeros(60)

max_z = 0.5
z_history = []

for step in range(1000):
    pos = env.data.qpos[:3].copy()
    vel = env.data.qvel[:3].copy()
    
    max_z = max(max_z, pos[2])
    
    targets[:] = 0.05 # retract, but not fully, to keep some stability?
    # No, fully retract is better for jumping clearance
    targets[:] = 0.0
    
    if state == "jump_up_right":
        for i, d in enumerate(env.dirs_body):
            # push left wall to jump up-right
            if d[1] < -0.3 and d[2] < -0.2:
                targets[i] = 0.16
        timer += 1
        if timer > 6 and pos[1] > 0.03:
            state = "jump_up_left"
            timer = 0
            
    elif state == "jump_up_left":
        for i, d in enumerate(env.dirs_body):
            # push right wall to jump up-left
            if d[1] > 0.3 and d[2] < -0.2:
                targets[i] = 0.16
        timer += 1
        if timer > 6 and pos[1] < -0.03:
            state = "jump_up_right"
            timer = 0
            
    env.step(targets)
    if step % 50 == 0:
        print(f"Step {step:4d} [{state:15s}]: z={pos[2]:.3f}m, y={pos[1]:.3f}m, vz={vel[2]:.3f}m/s")
    
    # Failsafe restart
    if pos[2] < 0.2:
        break

print(f"Max Z achieved: {max_z:.3f}m (Net climb: {max_z - 0.5:.3f}m)")

