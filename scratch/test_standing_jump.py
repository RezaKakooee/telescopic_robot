import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/config.yaml")
scenario = generate_scenario("goal", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=500)
obs, info = env.reset(seed=42)

dirs_body = env.dirs_body
max_extend = env.max_extend

print(f"Spawn: {info['ball_xy']}, Max extend: {max_extend}m")

# State machine for standing jump
# 0: Standing rest (steps 0-40)
# 1: Deep Crouch / Preload (steps 41-60)
# 2: Explosive Takeoff Thrust (steps 61-75)
# 3: Airborne Flight (until landing)
# 4: Compliant Touchdown & Settle

jump_start_step = 60
crouch_start_step = 35

z_history = []
vz_history = []

for step in range(1, 201):
    ball_pos = env.data.qpos[0:3]
    ball_vel = env.data.qvel[0:3]
    quat = env.data.qpos[3:7]

    z_history.append(ball_pos[2])
    vz_history.append(ball_vel[2])

    w, x, y, z = quat
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    dirs_world = dirs_body @ R.T
    u_z = dirs_world[:, 2]

    targets = np.zeros(env.n_bars, dtype=np.float32)

    if step < crouch_start_step:
        # Phase 0: Standing Still
        # Bottom rods maintain steady standing height
        bottom_mask = (u_z < -0.3)
        targets[bottom_mask] = 0.05
        state_str = "STAND"
    elif step < jump_start_step:
        # Phase 1: Deep Crouch Preload
        # Retract bottom rods so core drops low to ground
        targets[:] = 0.00
        state_str = "CROUCH"
    elif step < jump_start_step + 10:
        # Phase 2: EXPLOSIVE TAKEOFF
        # Fire ALL downward rods simultaneously to 100% full stroke!
        ground_thrust_mask = (u_z < 0.10)
        # Weight thrust by downward alignment for maximum vertical launch
        downwardness = np.clip(-u_z, 0.0, 1.0)
        targets[ground_thrust_mask] = max_extend
        # Top rods stay retracted
        targets[u_z > 0.15] = 0.0
        state_str = "🚀 TAKEOFF"
    elif ball_pos[2] > 0.28:
        # Phase 3: Airborne Flight
        # In mid-air, tuck rods cleanly
        targets[:] = 0.015
        state_str = "✈️ AIRBORNE"
    else:
        # Phase 4: Soft Landing Touchdown
        bottom_mask = (u_z < -0.2)
        # Compliant landing extension
        targets[bottom_mask] = 0.06
        state_str = "🛬 LANDING"

    obs, rew, term, trunc, info = env.step(targets)

    if step % 10 == 0 or state_str.startswith("🚀") or (state_str.startswith("✈️") and step % 5 == 0):
        print(f"Step {step:3d} [{state_str:12s}]: z={ball_pos[2]:.3f}m, vz={ball_vel[2]:+5.2f}m/s, pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f})")

env.close()

z_arr = np.array(z_history)
vz_arr = np.array(vz_history)
baseline_z = z_arr[10]
peak_z = float(np.max(z_arr))
peak_vz = float(np.max(vz_arr))
net_jump_cm = (peak_z - baseline_z) * 100.0

print(f"\n=== Standing Jump Results ===")
print(f"  - Standing Baseline Height: {baseline_z:.3f} m")
print(f"  - Peak Airborne Altitude:   {peak_z:.3f} m")
print(f"  - Net Jump Height:          +{net_jump_cm:.1f} cm (Airborne Clearance)")
print(f"  - Peak Vertical Velocity:   {peak_vz:+.2f} m/s")
