import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/standing_jump_showcase.yaml")
scenario = generate_scenario("goal", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=500)
obs, info = env.reset(seed=42)

dirs_body = env.dirs_body
max_extend = env.max_extend

print(f"Spawn: {info['ball_xy']}, Max extend: {max_extend}m")

# Forward Jump State Machine:
# Phase 0: Stand (steps 0-35)
# Phase 1: Deep Crouch Preload (steps 35-55) -> full compression to z ~ 0.156m
# Phase 2: Directional Forward Impulse (steps 55-68) -> Rear-biased explosive thrust
# Phase 3: Airborne Parabolic Flight (until landing) -> mid-air tuck
# Phase 4: Forward Compliant Touchdown & Rollout

takeoff_step = 55
crouch_step = 35

x_history = []
z_history = []
vx_history = []
vz_history = []

for step in range(1, 200):
    ball_pos = env.data.qpos[0:3]
    ball_vel = env.data.qvel[0:3]
    quat = env.data.qpos[3:7]

    x_history.append(ball_pos[0])
    z_history.append(ball_pos[2])
    vx_history.append(ball_vel[0])
    vz_history.append(ball_vel[2])

    w, x, y, z = quat
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    dirs_world = dirs_body @ R.T

    # Travel coordinate along +X
    u_long = dirs_world[:, 0]
    u_lat = dirs_world[:, 1]
    u_z = dirs_world[:, 2]

    targets = np.zeros(env.n_bars, dtype=np.float32)

    if step < crouch_step:
        # Phase 0: Standing Still
        bottom_mask = (u_z < -0.3)
        targets[bottom_mask] = 0.045
        phase_str = "STAND"
    elif step < takeoff_step:
        # Phase 1: Deep Crouch Preload (all rods retract to store full 16cm stroke!)
        targets[:] = 0.00
        phase_str = "CROUCH"
    elif step < takeoff_step + 12:
        # Phase 2: DIRECTIONAL FORWARD TAKEOFF IMPULSE
        # All ground rods fire to max_extend with strong rear-thrust bias
        ground_mask = (u_z < 0.10)
        # Rear-downward bias: rods with u_long < 0 push with 100% stroke, front rods with 40%
        forward_bias = np.clip(1.0 - 0.90 * np.maximum(u_long, -0.3), 0.35, 1.0)
        targets[ground_mask] = max_extend * forward_bias[ground_mask]
        targets[u_long > 0.15] = 0.0
        targets[u_z > 0.15] = 0.0
        phase_str = "🚀 FORWARD LAUNCH"
    elif ball_pos[2] > 0.28:
        # Phase 3: Airborne Parabolic Leap (Mid-air tuck)
        targets[:] = 0.015
        phase_str = "✈️ AIRBORNE LEAP"
    else:
        # Phase 4: Forward Landing Touchdown & Rollout
        bottom_mask = (u_z < -0.20)
        targets[bottom_mask] = 0.055
        # Rear pushers continue rolling forward upon landing
        rear_pusher = (u_long < -0.15) & (u_z < 0.0)
        targets[rear_pusher] = 0.12
        phase_str = "🛬 TOUCHDOWN ROLLOUT"

    obs, rew, term, trunc, info = env.step(targets)

    if step % 10 == 0 or phase_str.startswith("🚀") or (phase_str.startswith("✈️") and step % 5 == 0):
        print(f"Step {step:3d} [{phase_str:20s}]: pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.3f}m), vx={ball_vel[0]:+5.2f}m/s, vz={ball_vel[2]:+5.2f}m/s")

env.close()

x_arr = np.array(x_history)
z_arr = np.array(z_history)
vx_arr = np.array(vx_history)
vz_arr = np.array(vz_history)

takeoff_x = float(x_arr[takeoff_step])
landing_indices = np.where((z_arr[takeoff_step+12:] <= 0.24))[0]
if len(landing_indices) > 0:
    landing_idx = takeoff_step + 12 + landing_indices[0]
    landing_x = float(x_arr[landing_idx])
    airborne_dist = landing_x - takeoff_x
else:
    landing_idx = len(x_arr) - 1
    landing_x = float(x_arr[-1])
    airborne_dist = landing_x - takeoff_x

peak_z = float(np.max(z_arr))
peak_vx = float(np.max(vx_arr))
peak_vz = float(np.max(vz_arr))

print(f"\n=== Forward Jump Results ===")
print(f"  - Takeoff Position:         x = {takeoff_x:.3f} m")
print(f"  - Landing Position:         x = {landing_x:.3f} m")
print(f"  - Peak Airborne Altitude:   z = {peak_z:.3f} m (Net Height: +{(peak_z-z_arr[10])*100:.1f} cm)")
print(f"  - Peak Forward Velocity:    vx = {peak_vx:+.2f} m/s")
print(f"  - Peak Vertical Velocity:   vz = {peak_vz:+.2f} m/s")
print(f"  - Airborne Leap Distance:   +{airborne_dist:.2f} m (Forward Flight in 1 Jump!)")
