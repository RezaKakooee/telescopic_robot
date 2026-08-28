import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
import mujoco

cfg = load_config("configs/rl/terrain_slopes_and_ramps.yaml")
scenario = generate_scenario("slopes", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=300)
obs, info = env.reset(seed=42)

for step in range(1, 220):
    ball_pos = env.data.qpos[0:3]
    ball_vel = env.data.qvel[0:3]
    quat = env.data.qpos[3:7]

    w, x, y, z = quat
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    dirs_world = env.dirs_body @ R.T

    slope_angle = np.radians(9.3) if 2.5 <= ball_pos[0] <= 6.8 else 0.0
    slope_tangent = np.array([np.cos(slope_angle), 0.0, np.sin(slope_angle)])
    slope_normal = np.array([-np.sin(slope_angle), 0.0, np.cos(slope_angle)])
    ideal_push = -0.707 * slope_tangent - 0.707 * slope_normal
    ideal_push /= np.linalg.norm(ideal_push)

    u_long = dirs_world @ slope_tangent
    u_lat = dirs_world[:, 1]
    u_norm = dirs_world @ slope_normal

    align = dirs_world @ ideal_push
    wave = np.clip((np.maximum(align, 0.0) ** 1.5) * 3.5, 0.0, 1.0)
    wave = wave * np.clip(1.0 - 2.0 * (u_lat ** 2), 0.0, 1.0)
    wave[u_long > -0.05] = 0.0
    wave[u_norm > 0.05] = 0.0

    targets = env.max_extend * wave
    obs, rew, term, trunc, info = env.step(targets)

print(f"\n--- Contacts at step 220 (pos={ball_pos}) ---")
print(f"Number of contacts: {env.data.ncon}")
for ci in range(env.data.ncon):
    con = env.data.contact[ci]
    g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
    g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
    print(f"Contact {ci}: {g1} <-> {g2}, dist={con.dist:.4f}, pos={con.pos}")

print(f"qpos qvel: pos={env.data.qpos[:3]}, vel={env.data.qvel[:3]}, angvel={env.data.qvel[3:6]}")
