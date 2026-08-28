"""Test multi-meter chimney vertical climbing using dynamic stemming jumps."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5, shaft_length=1.2):
    half_gap = shaft_width / 2.0  # 0.18m
    box_w = 0.60
    box_y_left = half_gap + box_w / 2.0    # +0.48m
    box_y_right = -half_gap - box_w / 2.0  # -0.48m

    steps = [
        # Box 1 (Left vertical box): pos_x, pos_y, hx, hy, height
        [0.0, box_y_left, shaft_length / 2.0, box_w / 2.0, shaft_height],
        # Box 2 (Right vertical box)
        [0.0, box_y_right, shaft_length / 2.0, box_w / 2.0, shaft_height],
    ]

    walls = np.array([
        [-shaft_length / 2.0 - 0.2, -1.2, shaft_length / 2.0 + 0.2, -1.2],
        [shaft_length / 2.0 + 0.2, -1.2, shaft_length / 2.0 + 0.2, 1.2],
        [shaft_length / 2.0 + 0.2, 1.2, -shaft_length / 2.0 - 0.2, 1.2],
        [-shaft_length / 2.0 - 0.2, 1.2, -shaft_length / 2.0 - 0.2, -1.2],
    ])

    yardlines = []
    for h in np.arange(0.5, shaft_height, 0.5):
        color = "0.96 0.78 0.08 1.0" if int(round(h * 10)) % 10 == 0 else "0.95 0.95 0.95 0.8"
        yardlines.append([0.0, half_gap + 0.01, shaft_length / 2.0 * 0.9, 0.02, color])
        yardlines.append([0.0, -half_gap - 0.01, shaft_length / 2.0 * 0.9, 0.02, color])

    return Scenario(
        kind="chimney", name="chimney_climb",
        spawn_xy=np.array([0.0, 0.0]), goal=np.array([0.0, 0.0]),
        path_pts=np.array([[0.0, 0.0], [0.0, 0.0]]), markers=np.array([[0.0, 0.0]]),
        path_length=shaft_height,
        walls=walls,
        steps=steps,
        yardlines=yardlines,
    )


def test_jumps():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn resting on floor at z = 0.25m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.25
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Initial pos:", env.data.qpos[0:3])

    # Jump 1: Launch from floor up into chimney
    # 1. Takeoff blast (10 steps)
    for _ in range(12):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_z = dirs[:, 2]
        t = np.zeros(len(env.dirs_body), dtype=np.float32)
        t[u_z < 0.15] = env.max_extend
        env.step(t)

    print(f"Post takeoff 1: z = {env.data.qpos[2]:.3f}m, vz = {env.data.qvel[2]:+.2f}m/s")

    # 2. Coast up & Catch at apex (25 steps)
    for step in range(25):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        # Catch and clamp
        t = flank * 0.085
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)

    z1 = float(env.data.qpos[2])
    print(f"Jump 1 Lock: z = {z1:.3f}m, vz = {env.data.qvel[2]:+.2f}m/s")

    # Jump 2: Stem jump off vertical walls from z1
    for _ in range(12):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)

        t = np.zeros(len(env.dirs_body), dtype=np.float32)
        # Blast downward-lateral rods into the walls
        blast = (u_z < 0.10) & (flank > 0.3)
        t[blast] = env.max_extend
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)

    # Catch at Jump 2 apex
    for _ in range(30):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.085
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)

    z2 = float(env.data.qpos[2])
    print(f"Jump 2 Lock: z = {z2:.3f}m, vz = {env.data.qvel[2]:+.2f}m/s")

    # Jump 3: Stem jump off vertical walls from z2
    for _ in range(12):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)

        t = np.zeros(len(env.dirs_body), dtype=np.float32)
        blast = (u_z < 0.10) & (flank > 0.3)
        t[blast] = env.max_extend
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)

    # Catch at Jump 3 apex
    for _ in range(30):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.085
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)

    z3 = float(env.data.qpos[2])
    print(f"Jump 3 Lock: z = {z3:.3f}m, vz = {env.data.qvel[2]:+.2f}m/s")
    env.close()

test_jumps()
