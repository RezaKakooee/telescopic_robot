import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/terrain_slopes_and_ramps.yaml")
scenario = generate_scenario("slopes", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=1200)
obs, info = env.reset(seed=42)

dirs_body = env.dirs_body
max_extend = env.max_extend

print(f"Spawn: {info['ball_xy']} -> Goal: {env.scenario.goal}")

for step in range(1, 1001):
    ball_pos = env.data.qpos[0:3]
    ball_vel = env.data.qvel[0:3]
    quat = env.data.qpos[3:7]

    # Compute rotation matrix
    w, x, y, z = quat
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    dirs_world = dirs_body @ R.T

    # Desired direction with cross-track correction
    cross_track_y = float(ball_pos[1])
    target_heading_y = np.clip(-3.5 * cross_track_y, -0.6, 0.6)
    d_hat = np.array([np.sqrt(max(1.0 - target_heading_y**2, 0.1)), target_heading_y])
    d_hat /= np.linalg.norm(d_hat)

    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
    u_z = dirs_world[:, 2]

    # High-power rear thrust on trailing rods (u_long in [-1.0, -0.15])
    rear_factor = np.clip((-u_long - 0.12) / 0.88, 0.0, 1.0)
    down_factor = np.clip(1.0 - abs(u_z + 0.35) / 0.85, 0.0, 1.0)

    wave = (rear_factor ** 1.1) * down_factor * 2.8
    wave = np.clip(wave * np.clip(1.0 - 1.8 * (u_lat ** 2), 0.0, 1.0), 0.0, 1.0)

    # Gentle underbelly support (0.04m) so feet lift the core
    is_underbelly = (u_z < -0.30) & (u_long <= -0.05)
    depth_frac = np.clip((-u_z - 0.30) / 0.70, 0.0, 1.0)
    support_stance = depth_frac * 0.25 * np.clip(1.0 - 1.5 * (u_lat ** 2), 0.0, 1.0)
    wave = np.where(is_underbelly, np.maximum(wave, support_stance), wave)

    # Strictly retract forward & top rods
    wave[u_long > -0.05] = 0.0
    wave[u_z > 0.10] = 0.0

    targets = max_extend * wave

    obs, rew, term, trunc, info = env.step(targets)

    if step % 25 == 0 or term or trunc or info["distance"] < 0.45:
        print(f"Step {step:4d}: Ball pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.2f}), vx={ball_vel[0]:.2f}m/s, dist={info['distance']:.2f}m")

    if term or trunc or info["distance"] < 0.45:
        print(f"🎉 REACHED GOAL at step {step}! Final pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f})")
        break

env.close()
