"""Test vertical jump launch and apex wall clamp with clean flat chimney config."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario


def test():
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = generate_scenario("chimney", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn resting on floor at z = 0.22m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Initial pos:", env.data.qpos[0:3])

    # 1. Rest standing (10 steps)
    for _ in range(15):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        t = np.zeros(env.n_bars, dtype=np.float32)
        t[dirs[:, 2] < -0.3] = 0.045
        env.step(t)

    print(f"Pre-jump rest: z = {env.data.qpos[2]:.3f}m")

    # 2. Crouch (10 steps)
    for _ in range(10):
        env.step(np.zeros(env.n_bars, dtype=np.float32))

    print(f"Crouch: z = {env.data.qpos[2]:.3f}m")

    # 3. Explosive Launch (12 steps): all bottom rods fire 100% stroke
    for _ in range(12):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_z = dirs[:, 2]
        t = np.zeros(env.n_bars, dtype=np.float32)
        t[u_z < 0.10] = env.max_extend
        t[u_z > 0.15] = 0.0
        env.step(t)

    print(f"🚀 Launch: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # 4. Flight into Chimney (15 steps)
    for step in range(15):
        env.step(np.full(env.n_bars, 0.015, dtype=np.float32))
        if step % 5 == 0:
            print(f"  Ascent step {step:2d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # 5. Apex Wall Catch & Clamp (60 steps)
    print("🔒 Clamping against Vertical Box 1 & Box 2...")
    for step in range(60):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.090
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)
        if step % 20 == 0:
            print(f"  Suspended step {step:2d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_apex = float(env.data.qpos[2])
    print(f"Apex Locked: z = {z_apex:.3f}m (Climbed +{z_apex - 0.22:.2f}m up!)")

    # 6. Controlled Glide Descent (50 steps)
    print("🪂 Controlled Glide Descent...")
    for step in range(60):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.038
        if env.data.qpos[2] < 0.26:
            t[u_z < -0.20] = 0.055
        env.step(t)
        if step % 20 == 0:
            print(f"  Descent step {step:2d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    print(f"🛬 Landed: z = {env.data.qpos[2]:.3f}m")
    env.close()

test()
