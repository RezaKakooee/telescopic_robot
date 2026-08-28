import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/terrain_transparent_glass_pipe.yaml")
scenario = generate_scenario("glass_pipe", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=1000)
obs, info = env.reset(seed=42)

dirs_body = env.dirs_body
max_extend = env.max_extend

print(f"Spawn: {info['ball_xy']} -> Goal: {env.scenario.goal}")

for step in range(1, 601):
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

    # Desired direction with strong lateral centering towards y=0
    cross_track_y = float(ball_pos[1])
    target_heading_y = np.clip(-3.5 * cross_track_y, -0.6, 0.6)
    d_hat = np.array([np.sqrt(max(1.0 - target_heading_y**2, 0.1)), target_heading_y])
    d_hat /= np.linalg.norm(d_hat)

    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
    u_z = dirs_world[:, 2]

    # Ideal rear-downward push vector
    ideal_push = -0.707 * np.array([d_hat[0], d_hat[1], 0.0]) + np.array([0.0, 0.0, -0.707])
    ideal_push /= np.linalg.norm(ideal_push)
    align = dirs_world @ ideal_push

    # Push wave with lateral centering bias
    wave = np.clip((align ** 2) * 2.8, 0.0, 1.0)
    # Lateral flank tucking: keep side rods tucked inside ball radius
    wave = wave * np.clip(1.0 - 2.0 * (u_lat ** 2), 0.0, 1.0)
    # Strictly retract rods outside rear-down quadrant
    wave[u_long > -0.05] = 0.0
    wave[u_z > 0.05] = 0.0

    targets = max_extend * wave

    obs, rew, term, trunc, info = env.step(targets)

    if step % 25 == 0 or term or trunc or info["distance"] < 0.45:
        print(f"Step {step:4d}: Ball pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.2f}), vx={ball_vel[0]:.2f}m/s, dist={info['distance']:.2f}m")

    if term or trunc or info["distance"] < 0.45:
        print(f"🎉 REACHED GOAL at step {step}! Final pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f})")
        break

env.close()
