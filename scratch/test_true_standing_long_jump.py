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

print(f"Testing True Standing Long Jump with Tilted Full-Cluster Impulse...")

# Standing long jump routine:
# Steps 1-30: Complete standstill standing at x=0.0m
# Steps 30-42: Forward tilt preload (retract front rods, lower COM, tip body forward)
# Steps 42-54: FULL CLUSTER EXPLOSIVE LAUNCH (All 15 ground contact rods explode at 100% stroke!)
# Steps 55-90: Mid-air flight (tuck rods)
# Steps 90+: Touchdown

takeoff_step = 42
crouch_step = 30

x_history = []
z_history = []
vx_history = []
vz_history = []

for step in range(1, 130):
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

    u_long = dirs_world[:, 0]  # +X is forward
    u_lat = dirs_world[:, 1]
    u_z = dirs_world[:, 2]

    targets = np.zeros(env.n_bars, dtype=np.float32)

    if step < crouch_step:
        # Stationary rest standing at (0,0)
        bottom_mask = (u_z < -0.3)
        targets[bottom_mask] = 0.045
        phase = "STAND"
    elif step < takeoff_step:
        # Forward tilt preload: front bottom rods 0.0, rear bottom rods 0.04
        is_front_crouch = (u_long >= 0.0) & (u_z < 0.0)
        is_rear_crouch = (u_long < 0.0) & (u_z < 0.0)
        targets[is_front_crouch] = 0.00
        targets[is_rear_crouch] = 0.035
        phase = "TILT_PRELOAD"
    elif step < takeoff_step + 10:
        # FULL CLUSTER EXPLOSIVE LAUNCH (All ground contact rods fire to 100% stroke!)
        ground_mask = (u_z < 0.10)
        targets[ground_mask] = max_extend
        # Top rods stay 0
        targets[u_z > 0.15] = 0.0
        phase = "🚀 FORWARD_LAUNCH"
    elif ball_pos[2] > 0.28:
        # Airborne flight - tuck all rods
        targets[:] = 0.015
        phase = "✈️ AIRBORNE"
    else:
        # Touchdown landing
        bottom_mask = (u_z < -0.20)
        targets[bottom_mask] = 0.055
        phase = "🛬 LANDING"

    obs, rew, term, trunc, info = env.step(targets)

    if step % 5 == 0 or phase.startswith("🚀"):
        print(f"Step {step:3d} [{phase:18s}]: pos=(x={ball_pos[0]:.3f}m, z={ball_pos[2]:.3f}m), vx={ball_vel[0]:+5.2f}m/s, vz={ball_vel[2]:+5.2f}m/s")

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
else:
    landing_x = float(x_arr[-1])

flight_dist = landing_x - takeoff_x
peak_z = float(np.max(z_arr))
peak_vx = float(np.max(vx_arr))
peak_vz = float(np.max(vz_arr))

print(f"\n=== Standing Long Jump Evaluation ===")
print(f"  - Initial Stand:      x = {x_arr[0]:.3f} m")
print(f"  - Takeoff Position:   x = {takeoff_x:.3f} m")
print(f"  - Landing Position:   x = {landing_x:.3f} m")
print(f"  - Total Jump Distance: +{flight_dist:.2f} m (Flight Distance in Mid-Air from Standstill!)")
print(f"  - Peak Altitude:      z = {peak_z:.3f} m (Net Height: +{(peak_z-z_arr[10])*100:.1f} cm)")
print(f"  - Peak Forward Speed: vx = {peak_vx:+.2f} m/s")
print(f"  - Peak Upward Speed:  vz = {peak_vz:+.2f} m/s")
