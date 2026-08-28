"""Test clean barrier boxes for chimney climbing."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_clean_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.0, shaft_length=1.2):
    half_gap = shaft_width / 2.0
    box_w = 0.50
    box_y_left = half_gap + box_w / 2.0    # +0.43m
    box_y_right = -half_gap - box_w / 2.0  # -0.43m

    # Use obstacles (clean rectangular barrier boxes without protruding brackets)
    # [px, py, phx, phy, phz] -> size=(phx, phy, phz), pos=(px, py, phz)
    obstacles = np.array([
        [0.0, box_y_left, shaft_length / 2.0, box_w / 2.0, shaft_height / 2.0],
        [0.0, box_y_right, shaft_length / 2.0, box_w / 2.0, shaft_height / 2.0],
    ])

    walls = np.array([
        [-shaft_length / 2.0 - 0.2, -1.2, shaft_length / 2.0 + 0.2, -1.2],
        [shaft_length / 2.0 + 0.2, -1.2, shaft_length / 2.0 + 0.2, 1.2],
        [shaft_length / 2.0 + 0.2, 1.2, -shaft_length / 2.0 - 0.2, 1.2],
        [-shaft_length / 2.0 - 0.2, 1.2, -shaft_length / 2.0 - 0.2, -1.2],
    ])

    yardlines = []
    for h in np.arange(0.25, shaft_height, 0.25):
        color = "0.96 0.78 0.08 1.0" if int(round(h * 100)) % 100 == 0 else "0.95 0.95 0.95 0.8"
        yardlines.append([0.0, half_gap + 0.005, shaft_length / 2.0 * 0.9, 0.015, color])
        yardlines.append([0.0, -half_gap - 0.005, shaft_length / 2.0 * 0.9, 0.015, color])

    return Scenario(
        kind="chimney", name="clean_chimney",
        spawn_xy=np.array([0.0, 0.0]), goal=np.array([0.0, 0.0]),
        path_pts=np.array([[0.0, 0.0], [0.0, 0.0]]), markers=np.array([[0.0, 0.0]]),
        path_length=shaft_height,
        walls=walls,
        obstacles=obstacles,
        yardlines=yardlines,
    )


def test_clean():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_clean_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.0)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn resting on floor at z = 0.20m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.20
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Initial pos:", env.data.qpos[0:3])

    # 1. Takeoff jump from ground between the two clean boxes
    for _ in range(12):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_z = dirs[:, 2]
        t = np.zeros(len(env.dirs_body), dtype=np.float32)
        t[u_z < 0.15] = env.max_extend
        env.step(t)

    print(f"Post jump 1: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # 2. Coast & Catch in mid-air
    for _ in range(25):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.085
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)

    print(f"Locked at height: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")
    env.close()

test_clean()
