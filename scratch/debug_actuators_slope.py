import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/terrain_slopes_and_ramps.yaml")
scenario = generate_scenario("slopes", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=300)
obs, info = env.reset(seed=42)

for step in range(1, 201):
    ball_pos = env.data.qpos[0:3]
    quat = env.data.qpos[3:7]

    w, x, y, z = quat
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    dirs_world = env.dirs_body @ R.T

    d_hat = np.array([1.0, 0.0])
    u_long = dirs_world[:, 0]
    u_lat = dirs_world[:, 1]
    u_z = dirs_world[:, 2]

    ideal_push = np.array([-0.707, 0.0, -0.707])
    align = dirs_world @ ideal_push
    wave = np.clip(np.maximum(align, 0.0) * 1.8, 0.0, 1.0)
    wave = wave * np.clip(1.0 - 1.8 * (u_lat ** 2), 0.0, 1.0)

    is_underbelly = (u_z < -0.25) & (u_long <= 0.02)
    depth_frac = np.clip((-u_z - 0.25) / 0.75, 0.0, 1.0)
    support_stance = depth_frac * 0.40 * np.clip(1.0 - 1.5 * (u_lat ** 2), 0.0, 1.0)
    wave = np.where(is_underbelly, np.maximum(wave, support_stance), wave)

    wave[u_long > 0.02] = 0.0
    wave[u_z > 0.15] = 0.0

    targets = env.max_extend * wave
    env.step(targets)

print(f"\n--- Diagnostic at Step 200 (pos={ball_pos}) ---")
joint_pos = env.data.qpos[7:]
joint_vel = env.data.qvel[6:]
act_ctrl = env.data.ctrl[:]
print(f"Active targets (> 0.02m): {np.sum(targets > 0.02)} / 60")
print(f"Max target: {targets.max():.4f}m, Min target: {targets.min():.4f}m")
print(f"Actual extensions: min={joint_pos.min():.4f}m, max={joint_pos.max():.4f}m, mean={joint_pos.mean():.4f}m")
print(f"Ball angular vel (deg/s): {np.degrees(env.data.qvel[3:6])}")

# Check which rods are touching the ground
for k in range(60):
    if targets[k] > 0.03:
        u = dirs_world[k]
        print(f"Rod {k:2d}: target={targets[k]:.3f}m, actual={joint_pos[k]:.3f}m, world_dir=({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f})")
