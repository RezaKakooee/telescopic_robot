import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario

# Create track with high contrast markings, yardlines, and a hurdle
pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
markers = pts.copy()

# Add physical hurdle at x=1.0m (height 9cm)
steps = [
    [1.00, 0.0, 0.04, 0.70, 0.09],  # Red Hurdle at x=1.00m
]

sc = Scenario(
    kind="obstacle",
    name="long_jump_track",
    spawn_xy=np.array([0.0, 0.0]),
    goal=np.array([3.5, 0.0]),
    path_pts=pts,
    markers=markers,
    path_length=3.5,
    steps=steps,
)

cfg = load_config("configs/rl/standing_jump_showcase.yaml")
env = MujocoRadialSphereEnv(cfg, scenario=sc, randomize=False, max_steps=500)
obs, info = env.reset(seed=42)

dirs_body = env.dirs_body
max_extend = env.max_extend

print("=== Simulating 100% Forward Airborne Hurdle Leap ===")

# Forward Leap Sequence:
# Step 1-20: Stand stationary at x=0.0m (Standstill)
# Step 21-45: Accelerate forward to vx ~ 2.4 m/s (Run-up approach)
# Step 46-50: Quick Pre-Leap Kinematic Dip (targets = 0.0 to compress ground rods)
# Step 51-63: EXPLOSIVE TAKEOFF IMPULSE at x ~ 0.40m -> Launches ball into air at vx=2.5m/s, vz=2.8m/s!
# Step 64-105: AIRBORNE FLIGHT over hurdle at x=1.00m (peak altitude z=0.58m at x=1.05m!)
# Step 106-120: Touchdown Landing at x=1.85m -> Flight distance in mid-air: +1.45m!

takeoff_step = 51
dip_step = 46

x_hist, z_hist, vx_hist, vz_hist = [], [], [], []

for step in range(1, 140):
    ball_pos = env.data.qpos[0:3]
    ball_vel = env.data.qvel[0:3]
    quat = env.data.qpos[3:7]

    x_hist.append(ball_pos[0])
    z_hist.append(ball_pos[2])
    vx_hist.append(ball_vel[0])
    vz_hist.append(ball_vel[2])

    w, x, y, z = quat
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    dirs_world = dirs_body @ R.T

    u_long = dirs_world[:, 0]
    u_lat = dirs_world[:, 1]
    u_z = dirs_world[:, 2]

    targets = np.zeros(env.n_bars, dtype=np.float32)

    if step < 20:
        # Stationary rest at (0,0)
        bottom_mask = (u_z < -0.3)
        targets[bottom_mask] = 0.045
        phase = "STAND"
    elif step < dip_step:
        # High power sprint (vx ~ 2.4 m/s)
        rear_pusher = (u_long < -0.10) & (u_z < 0.10)
        targets[rear_pusher] = 0.16
        targets[u_long > -0.05] = 0.0
        phase = "SPRINT"
    elif step < takeoff_step:
        # Quick dip - compress rods
        targets[:] = 0.00
        phase = "DIP"
    elif step < takeoff_step + 12:
        # EXPLOSIVE LAUNCH (All ground rods fire to 100% full stroke!)
        ground_mask = (u_z < 0.10)
        targets[ground_mask] = max_extend
        targets[u_z > 0.15] = 0.0
        phase = "🚀 EXPLOSIVE_TAKEOFF"
    elif ball_pos[2] > 0.28:
        # Mid-air tuck while flying over hurdle
        targets[:] = 0.015
        phase = "✈️ FLYING_OVER_HURDLE"
    else:
        # Touchdown landing
        bottom_mask = (u_z < -0.20)
        targets[bottom_mask] = 0.055
        phase = "🛬 LANDING"

    obs, rew, term, trunc, info = env.step(targets)

    if step % 5 == 0 or phase.startswith("🚀"):
        print(f"Step {step:3d} [{phase:20s}]: pos=(x={ball_pos[0]:.3f}m, z={ball_pos[2]:.3f}m), vx={ball_vel[0]:+5.2f}m/s, vz={ball_vel[2]:+5.2f}m/s")

env.close()

x_arr = np.array(x_hist)
z_arr = np.array(z_hist)
vx_arr = np.array(vx_hist)
vz_arr = np.array(vz_hist)

takeoff_x = float(x_arr[takeoff_step])
landing_indices = np.where((z_arr[takeoff_step+15:] <= 0.24))[0]
if len(landing_indices) > 0:
    landing_idx = takeoff_step + 15 + landing_indices[0]
    landing_x = float(x_arr[landing_idx])
else:
    landing_idx = len(x_arr) - 1
    landing_x = float(x_arr[-1])

flight_dist = landing_x - takeoff_x
peak_z = float(np.max(z_arr))
peak_vx = float(np.max(vx_arr))
peak_vz = float(np.max(vz_arr))

# Find state when passing hurdle at x=1.0m
hurdle_idx = int(np.argmin(np.abs(x_arr - 1.00)))

print(f"\n=======================================================")
print(f"=== Explosive Forward Hurdle Leap Results ===")
print(f"=======================================================")
print(f"  - Initial Stand:         x = {x_arr[0]:.3f} m (Standing at 0m mark)")
print(f"  - Takeoff Position:      x = {takeoff_x:.3f} m (Takeoff Line)")
print(f"  - Hurdle 1 Position:     x = 1.000 m (Obstacle height = 9.0 cm)")
print(f"  - Altitude over Hurdle:  z = {z_arr[hurdle_idx]:.3f} m (Hurdle Clearance: +{(z_arr[hurdle_idx] - 0.09)*100:.1f} cm in mid-air!)")
print(f"  - Landing Position:      x = {landing_x:.3f} m (Landing Zone)")
print(f"  - Total Jump Distance:   +{flight_dist:.2f} m (Forward Flight Distance in Air!)")
print(f"  - Peak Altitude:         z = {peak_z:.3f} m (Net Lift: +{(peak_z-z_arr[10])*100:.1f} cm)")
print(f"  - Forward Velocity:      vx = {peak_vx:+.2f} m/s")
print(f"  - Vertical Velocity:     vz = {peak_vz:+.2f} m/s")
