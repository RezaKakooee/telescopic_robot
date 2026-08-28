import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario

# Create a scenario with a high-contrast track, meter lines, and a hurdle
pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
markers = pts.copy()

# Add a hurdle at x=0.85m (height 8cm) and x=1.50m (height 8cm)
steps = [
    [0.85, 0.0, 0.03, 0.60, 0.08],  # Hurdle 1 at x=0.85m
    [1.60, 0.0, 0.03, 0.60, 0.08],  # Hurdle 2 at x=1.60m
]

sc = Scenario(
    kind="obstacle",
    name="jump_hurdle_track",
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

print(f"Testing Forward Jump over Hurdles...")

# Jump State Machine:
# 1. Stand at start line (x=0m): steps 1-25
# 2. Forward pre-pitch stride: steps 26-38 (rear pushers provide forward lean and vx ~ 1.2 m/s)
# 3. Explosive Full-Cluster Takeoff: steps 39-50 (all ground rods explode at 100% stroke, launching over the hurdle!)
# 4. Airborne Flight over Hurdle (x=0.85m): steps 51-85 (peak altitude ~ 0.55m)
# 5. Compliant Touchdown Landing at x=1.8m: steps 86-120

takeoff_step = 38
stride_step = 25

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

    if step < stride_step:
        # Stationary rest standing at x=0.0m
        bottom_mask = (u_z < -0.3)
        targets[bottom_mask] = 0.045
        phase = "STAND"
    elif step < takeoff_step:
        # Pre-launch forward acceleration lean (rear pushers extend to 0.16m, front rods retract)
        rear_pusher = (u_long < -0.10) & (u_z < 0.10)
        targets[rear_pusher] = max_extend
        targets[u_long > 0.0] = 0.0
        phase = "FORWARD_LEAN"
    elif step < takeoff_step + 10:
        # EXPLOSIVE FULL CLUSTER LAUNCH (All ground contact rods fire at 100% stroke!)
        ground_mask = (u_z < 0.12)
        targets[ground_mask] = max_extend
        targets[u_z > 0.15] = 0.0
        phase = "🚀 HURDLE_LAUNCH"
    elif ball_pos[2] > 0.28:
        # Airborne flight over hurdle - tuck rods
        targets[:] = 0.015
        phase = "✈️ FLYING_OVER_HURDLE"
    else:
        # Touchdown landing & rollout
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

print(f"\n=== Hurdle Jump Evaluation ===")
print(f"  - Initial Stand:         x = {x_arr[0]:.3f} m")
print(f"  - Takeoff Position:      x = {takeoff_x:.3f} m")
print(f"  - Hurdle 1 Position:     x = 0.850 m (Hurdle height = 8.0 cm)")
print(f"  - Altitude over Hurdle:  z = {z_arr[int(np.argmin(np.abs(x_arr - 0.85)))]:.3f} m (Clearance: +{z_arr[int(np.argmin(np.abs(x_arr - 0.85)))] - 0.08:.3f} m)")
print(f"  - Landing Position:      x = {landing_x:.3f} m")
print(f"  - Total Jump Distance:   +{flight_dist:.2f} m (Flight Distance in Mid-Air!)")
print(f"  - Peak Altitude:         z = {peak_z:.3f} m (Net Height: +{(peak_z-z_arr[10])*100:.1f} cm)")
print(f"  - Peak Forward Speed:    vx = {peak_vx:+.2f} m/s")
print(f"  - Peak Upward Speed:     vz = {peak_vz:+.2f} m/s")
