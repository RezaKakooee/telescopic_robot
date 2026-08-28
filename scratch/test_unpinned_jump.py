"""Test unpinned launch inside chimney: bottom rods only, lateral rods tucked!"""
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

    # 1. Rest standing: bottom downward rods only (u_z < -0.35, |u_lat| < 0.20)
    for _ in range(15):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]
        t = np.zeros(env.n_bars, dtype=np.float32)
        bottom_ground = (u_z < -0.35) & (np.abs(u_lat) < 0.25)
        t[bottom_ground] = 0.045
        env.step(t)

    print(f"Pre-jump rest: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # 2. Crouch (10 steps)
    for _ in range(10):
        env.step(np.zeros(env.n_bars, dtype=np.float32))

    print(f"Crouch: z = {env.data.qpos[2]:.3f}m")

    # 3. Explosive Launch (12 steps): ONLY downward ground rods fire! Lateral rods STAY TUCKED!
    for _ in range(12):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]
        t = np.zeros(env.n_bars, dtype=np.float32)
        # Ground launch cluster: pointing down, NOT into the side walls!
        ground_cluster = (u_z < -0.20) & (np.abs(u_lat) < 0.30)
        t[ground_cluster] = env.max_extend
        env.step(t)

    print(f"🚀 Takeoff: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # 4. Flight up through Chimney (20 steps)
    for step in range(20):
        env.step(np.zeros(env.n_bars, dtype=np.float32))
        if step % 5 == 0:
            print(f"  Ascent step {step:2d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # 5. Apex Wall Catch & Lock: now that ball is at apex in the air, clamp lateral rods!
    print("🔒 Apex Wall Lock: Clamping against Box 1 & Box 2...")
    for step in range(60):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.088
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)
        if step % 20 == 0:
            print(f"  Suspended step {step:2d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_apex = float(env.data.qpos[2])
    print(f"Apex Locked: z = {z_apex:.3f}m (Climbed +{z_apex - 0.22:.2f}m up!)")
    env.close()

test()
