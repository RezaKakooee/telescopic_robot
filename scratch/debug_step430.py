import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
import mujoco

cfg = load_config("configs/rl/terrain_slopes_and_ramps.yaml")
scenario = generate_scenario("slopes", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=450)
obs, info = env.reset(seed=42)

for step in range(1, 430):
    ball_pos = env.data.qpos[0:3]
    quat = env.data.qpos[3:7]

    w, x, y, z = quat
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    dirs_world = env.dirs_body @ R.T

    cross_track_y = float(ball_pos[1])
    target_heading_y = np.clip(-3.5 * cross_track_y, -0.6, 0.6)
    d_hat = np.array([np.sqrt(max(1.0 - target_heading_y**2, 0.1)), target_heading_y])
    d_hat /= np.linalg.norm(d_hat)

    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
    u_z = dirs_world[:, 2]

    rear_factor = np.clip((-u_long - 0.12) / 0.88, 0.0, 1.0)
    down_factor = np.clip(1.0 - abs(u_z + 0.35) / 0.85, 0.0, 1.0)
    wave = (rear_factor ** 1.1) * down_factor * 2.8
    wave = np.clip(wave * np.clip(1.0 - 1.8 * (u_lat ** 2), 0.0, 1.0), 0.0, 1.0)

    is_underbelly = (u_z < -0.30) & (u_long <= -0.05)
    depth_frac = np.clip((-u_z - 0.30) / 0.70, 0.0, 1.0)
    support_stance = depth_frac * 0.25 * np.clip(1.0 - 1.5 * (u_lat ** 2), 0.0, 1.0)
    wave = np.where(is_underbelly, np.maximum(wave, support_stance), wave)

    wave[u_long > -0.05] = 0.0
    wave[u_z > 0.10] = 0.0

    targets = env.max_extend * wave
    env.step(targets)

print(f"\n--- Contacts at step 430 (pos={ball_pos}) ---")
for ci in range(env.data.ncon):
    con = env.data.contact[ci]
    g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
    g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
    print(f"Contact {ci}: {g1} <-> {g2}, dist={con.dist:.4f}, pos={con.pos}")
