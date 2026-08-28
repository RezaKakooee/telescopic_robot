"""Test vertical climbing force balance."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5, shaft_length=1.2):
    half_gap = shaft_width / 2.0
    box_w = 0.60
    box_y_left = half_gap + box_w / 2.0
    box_y_right = -half_gap - box_w / 2.0

    steps = [
        [0.0, box_y_left, shaft_length / 2.0, box_w / 2.0, shaft_height],
        [0.0, box_y_right, shaft_length / 2.0, box_w / 2.0, shaft_height],
    ]

    walls = np.array([
        [-shaft_length / 2.0, -1.0, shaft_length / 2.0, -1.0],
        [shaft_length / 2.0, -1.0, shaft_length / 2.0, 1.0],
        [shaft_length / 2.0, 1.0, -shaft_length / 2.0, 1.0],
        [-shaft_length / 2.0, 1.0, -shaft_length / 2.0, -1.0],
    ])

    return Scenario(
        kind="chimney", name="chimney_climb",
        spawn_xy=np.array([0.0, 0.0]), goal=np.array([0.0, 0.0]),
        path_pts=np.array([[0.0, 0.0], [0.0, 0.0]]), markers=np.array([[0.0, 0.0]]),
        path_length=shaft_height,
        walls=walls,
        steps=steps,
    )


def test_lift():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn at z = 1.0m (mid-air between walls)
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 1.0
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Spawned at z = 1.0m (mid-air between walls)")

    # 1. Test vertical lift:
    # Continuously push downward-flank rods (u_z < 0.0) at full stroke
    # while upper-flank rods maintain clamping grip
    for step in range(1, 300):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T
        u_lat = dirs_world[:, 1]
        u_z = dirs_world[:, 2]

        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        y_now = float(env.data.qpos[1])
        side_centering = np.where(u_lat * y_now > 0, 1.15, 0.85)

        targets = np.zeros(len(env.dirs_body), dtype=np.float32)

        # Upper grip rods (u_z in [0.0, 0.5])
        upper = (u_z >= 0.0) & (flank > 0.3)
        # Lower thrust rods (u_z < 0.0)
        lower = (u_z < 0.0) & (flank > 0.3)

        targets[upper] = 0.075
        targets[lower] = env.max_extend

        targets *= side_centering
        targets = np.clip(targets, 0.0, env.max_extend)
        targets[np.abs(u_lat) < 0.22] = 0.0

        env.step(targets)

        if step % 50 == 0:
            print(f"Step {step:3d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_mid = float(env.data.qpos[2])
    print(f"\nHeight under upward push: z = {z_mid:.3f}m")

    # 2. Test downward push (moving down):
    # Lower rods hold grip, upper rods push upward (u_z > 0.0)
    for step in range(1, 300):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T
        u_lat = dirs_world[:, 1]
        u_z = dirs_world[:, 2]

        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        y_now = float(env.data.qpos[1])
        side_centering = np.where(u_lat * y_now > 0, 1.15, 0.85)

        targets = np.zeros(len(env.dirs_body), dtype=np.float32)

        upper = (u_z >= 0.0) & (flank > 0.3)
        lower = (u_z < 0.0) & (flank > 0.3)

        targets[lower] = 0.075
        targets[upper] = env.max_extend  # push up against wall -> core moves down

        targets *= side_centering
        targets = np.clip(targets, 0.0, env.max_extend)
        targets[np.abs(u_lat) < 0.22] = 0.0

        env.step(targets)

        if step % 50 == 0:
            print(f"Step {step:3d} (DOWN): z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_down = float(env.data.qpos[2])
    print(f"\nHeight under downward push: z = {z_down:.3f}m")
    env.close()

test_lift()
