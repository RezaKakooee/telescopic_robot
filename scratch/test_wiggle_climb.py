import numpy as np
import mujoco
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
obs, info = env.reset(seed=42)

env.data.qpos[1] = 0.0
env.data.qpos[2] = 0.5
mujoco.mj_forward(env.model, env.data)

print("Testing Wiggle Climb...")

max_z = 0.5

# Find best rods for left/right downward and left/right horizontal
left_down = []
right_down = []
left_horiz = []
right_horiz = []

for i, d in enumerate(env.dirs_body):
    if abs(d[0]) > 0.3: continue # minimize x twist
    if d[1] > 0.5 and d[2] < -0.3:
        left_down.append(i)  # Pushes against Left Wall
    elif d[1] < -0.5 and d[2] < -0.3:
        right_down.append(i) # Pushes against Right Wall
    elif d[1] > 0.8 and abs(d[2]) <= 0.3:
        left_horiz.append(i) # Pushes against Left Wall
    elif d[1] < -0.8 and abs(d[2]) <= 0.3:
        right_horiz.append(i) # Pushes against Right Wall

print("LD:", left_down)
print("RD:", right_down)
print("LH:", left_horiz)
print("RH:", right_horiz)

state = "push_left"
timer = 0
targets = np.zeros(60)

for step in range(3000):
    pos = env.data.qpos[:3].copy()
    vel = env.data.qvel[:3].copy()
    max_z = max(max_z, pos[2])
    
    # CHEAT: Zero out X and orientation for pure 2D testing
    env.data.qpos[0] = 0.0
    env.data.qvel[0] = 0.0
    env.data.qpos[3:7] = [1, 0, 0, 0]
    env.data.qvel[3:6] = [0, 0, 0]
    
    # Wiggle state machine
    # push_left: Extend LD (Left Down), retract RH (Right Horiz). Keep LH and RD retracted.
    # swap_to_right: Plant RD and LH.
    # push_right: Extend RD, retract LH. Keep RH and LD retracted.
    # swap_to_left: Plant LD and RH.
    
    targets[:] = 0.01
    
    if state == "push_left":
        # LD extends to push UP and RIGHT
        for i in left_down: targets[i] = 0.20
        # RH retracts to accommodate rightward motion, but maintains contact (spring-like)
        for i in right_horiz: targets[i] = 0.05
        # Keep other rods out of the way
        
        timer += 1
        # Switch when core has moved sufficiently right or timer expires
        if timer > 50 or pos[1] < -0.05:
            state = "swap_to_right"
            timer = 0
            
    elif state == "swap_to_right":
        # Plant all 4 sets to transfer weight safely
        for i in left_down: targets[i] = 0.20
        for i in right_horiz: targets[i] = 0.05
        for i in right_down: targets[i] = 0.05 # plant RD
        for i in left_horiz: targets[i] = 0.20 # plant LH
        
        timer += 1
        if timer > 20:
            state = "push_right"
            timer = 0
            
    elif state == "push_right":
        # RD extends to push UP and LEFT
        for i in right_down: targets[i] = 0.20
        # LH retracts to accommodate leftward motion
        for i in left_horiz: targets[i] = 0.05
        
        timer += 1
        if timer > 50 or pos[1] > 0.05:
            state = "swap_to_left"
            timer = 0
            
    elif state == "swap_to_left":
        # Plant all 4 sets
        for i in right_down: targets[i] = 0.20
        for i in left_horiz: targets[i] = 0.05
        for i in left_down: targets[i] = 0.05 # plant LD
        for i in right_horiz: targets[i] = 0.20 # plant RH
        
        timer += 1
        if timer > 20:
            state = "push_left"
            timer = 0
            
    env.step(targets)
    if step % 50 == 0:
        print(f"Step {step:4d} [{state:15s}]: z={pos[2]:.3f}m, y={pos[1]:.3f}m")

print(f"Max Z achieved: {max_z:.3f}m (Net climb: {max_z - 0.5:.3f}m)")
