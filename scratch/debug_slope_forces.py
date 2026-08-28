import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
import mujoco

cfg = load_config("configs/rl/terrain_slopes_and_ramps.yaml")
scenario = generate_scenario("slopes", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=250)
obs, info = env.reset(seed=42)

for step in range(1, 230):
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

    ideal_push = np.array([-0.60, 0.0, -0.80])
    ideal_push /= np.linalg.norm(ideal_push)
    align = dirs_world @ ideal_push

    wave = np.clip((np.maximum(align, 0.0) ** 1.0) * 2.5, 0.0, 1.0)
    wave = wave * np.clip(1.0 - 2.0 * (u_lat ** 2), 0.0, 1.0)
    wave[u_long > 0.02] = 0.0
    wave[u_z > 0.05] = 0.0

    targets = env.max_extend * wave
    env.step(targets)

print(f"\n--- Contacts & Forces at step 230 (pos={ball_pos}) ---")
c_force = np.zeros(6)
for ci in range(env.data.ncon):
    con = env.data.contact[ci]
    g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
    g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
    mujoco.mj_contactForce(env.model, env.data, ci, c_force)
    print(f"Contact {ci}: {g1} <-> {g2}, normal_force={c_force[0]:.2f}N, friction_force={np.linalg.norm(c_force[1:3]):.2f}N, pos={con.pos}")

print(f"\nActuator control signals for extended rods:")
for k in range(60):
    if targets[k] > 0.03:
        act_force = env.data.actuator_force[k]
        q_pos = env.data.qpos[7+k]
        print(f"Rod {k:2d}: target={targets[k]:.3f}m, actual_q={q_pos:.3f}m, force={act_force:.2f}N, dir=({dirs_world[k,0]:.2f}, {dirs_world[k,1]:.2f}, {dirs_world[k,2]:.2f})")
