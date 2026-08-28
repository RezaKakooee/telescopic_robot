import numpy as np
import mujoco
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from scipy.spatial.transform import Rotation

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
obs, info = env.reset(seed=42)

# Start in mid-air
env.data.qpos[2] = 0.5
mujoco.mj_forward(env.model, env.data)

print("Testing Traveling Wave Climb...")

max_z = 0.5

for step in range(1000):
    targets = np.zeros(60)
    
    # Wave phase
    t = step * 0.01
    wave_phase = t * 2 * np.pi * 1.0 # 1 Hz
    
    # Get current core orientation
    quat = env.data.qpos[3:7]
    r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]) # scipy uses x,y,z,w
    
    pos = env.data.qpos[:3]
    max_z = max(max_z, pos[2])
    
    for i, d_body in enumerate(env.dirs_body):
        # transform rod to world frame
        d_world = r.apply(d_body)
        uy, uz = d_world[1], d_world[2]
        
        # We only care about side rods
        if abs(uy) < 0.3:
            continue
            
        # Angle in the y-z plane
        angle = np.arctan2(uz, uy)
        
        # We want the 'bulge' to travel DOWN the left side, and DOWN the right side?
        # If the bulge travels DOWN the wall, it pushes the robot UP!
        # Left wall (uy < 0): angle is near PI or -PI.
        # Right wall (uy > 0): angle is near 0.
        
        if uy < 0:
            # Left wall. We want bulge to move from +z to -z.
            # Angle goes from PI/2 to PI to -PI/2.
            # So angle should decrease.
            target_angle = np.pi - wave_phase
        else:
            # Right wall. Bulge moves from +z to -z.
            # Angle goes from PI/2 to 0 to -PI/2.
            # So angle should decrease.
            target_angle = -wave_phase
            
        # Angle difference
        diff = (angle - target_angle + np.pi) % (2 * np.pi) - np.pi
        
        # If rod is near the target angle, extend it!
        if abs(diff) < 0.5:
            targets[i] = 0.16
        elif abs(diff) < 1.0:
            targets[i] = 0.08
        else:
            targets[i] = 0.0
            
    env.step(targets)
    if step % 50 == 0:
        print(f"Step {step:3d}: z={pos[2]:.3f}m, y={pos[1]:.3f}m")
        
print(f"Max Z achieved: {max_z:.3f}m (Net climb: {max_z - 0.5:.3f}m)")

