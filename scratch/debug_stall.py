"""Debug why ball stalls at x=1.37m when straddling gap."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill

cfg = load_config("configs/rl/gap_bridge.yaml")
scenario = generate_scenario("gap_bridge", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
env.reset(seed=42)

box_height = 0.25
env.data.qpos[0] = 0.0
env.data.qpos[1] = 0.0
env.data.qpos[2] = box_height + 0.19
env.data.qvel[:] = 0
mujoco.mj_forward(env.model, env.data)

for _ in range(25):
    t = execute_skill("straddle_gap", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, speed=0.0)
    env.step(t)

d_fwd = np.array([1.0, 0.0])
for step in range(1, 200):
    pos = env.data.qpos[0:3].copy()
    quat = env.data.qpos[3:7].copy()
    targets = execute_skill("straddle_gap", quat, env.dirs_body, env.max_extend, d_hat=d_fwd, speed=1.3, lateral_offset=float(pos[1]))
    env.step(targets)

    if step in (95, 100, 105, 120, 150):
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T
        u_long = dirs_world[:, 0]
        u_lat = dirs_world[:, 1]
        u_z = dirs_world[:, 2]

        print(f"\n--- STEP {step} (x={pos[0]:.2f}m, y={pos[1]:+.3f}m, z={pos[2]:.3f}m, vx={env.data.qvel[0]:.3f}) ---")
        active = np.where(targets > 0.03)[0]
        print(f"Active rods ({len(active)}):")
        for idx in active:
            print(f"  rod {idx:2d}: u_long={u_long[idx]:+.2f} u_lat={u_lat[idx]:+.2f} u_z={u_z[idx]:+.2f} target={targets[idx]:.3f}m actual={env.data.qpos[7+idx]:.3f}m")

env.close()
