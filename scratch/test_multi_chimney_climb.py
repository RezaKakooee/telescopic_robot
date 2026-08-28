"""Test multi-stage chimney climbing to reach multi-meter height."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.40, shaft_height=4.0, shaft_length=1.2):
    half_gap = shaft_width / 2.0  # 0.20m
    box_w = 0.50
    box_y_left = half_gap + box_w / 2.0    # +0.45m
    box_y_right = -half_gap - box_w / 2.0  # -0.45m

    steps = [
        [0.0, box_y_left, shaft_length / 2.0, box_w / 2.0, shaft_height],
        [0.0, box_y_right, shaft_length / 2.0, box_w / 2.0, shaft_height],
    ]

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
        kind="chimney", name="chimney_climb",
        spawn_xy=np.array([0.0, 0.0]), goal=np.array([0.0, 10.0]),
        path_pts=np.array([[0.0, 0.0], [0.0, 0.0]]), markers=np.array([[0.0, 0.0]]),
        path_length=shaft_height,
        walls=walls,
        steps=steps,
        yardlines=yardlines,
    )


def test():
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.40, shaft_height=4.0)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    obs, info = env.reset(seed=42)

    # 1. Stand at rest (20 steps)
    for _ in range(20):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        t = np.zeros(env.n_bars, dtype=np.float32)
        t[dirs[:, 2] < -0.3] = 0.045
        env.step(t)

    # 2. Stage 1 Climb: Crouch + Blast
    for _ in range(12):
        env.step(np.zeros(env.n_bars, dtype=np.float32))

    for _ in range(12):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]
        t = np.zeros(env.n_bars, dtype=np.float32)
        ground = (u_z < 0.0) & (np.abs(u_lat) < 0.45)
        t[ground] = env.max_extend
        env.step(t)

    # Flight 1
    for _ in range(18):
        env.step(np.full(env.n_bars, 0.010, dtype=np.float32))

    # Lock 1 at z ~ 0.64m
    for _ in range(40):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.098
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)

    z1 = float(env.data.qpos[2])
    print(f"✅ Stage 1 Height (Apex 1): z = {z1:.3f}m")

    # Controlled Descent all the way down to ground
    print("🪂 Controlled Descent to Floor...")
    for step in range(100):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        # Friction glide braking
        t = flank * 0.025
        if env.data.qpos[2] < 0.22:
            t[u_z < -0.20] = 0.055
        env.step(t)
        if step % 20 == 0:
            print(f"  Descent step {step:2d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    z_end = float(env.data.qpos[2])
    print(f"🛬 Touchdown at Floor: z = {z_end:.3f}m")
    env.close()

test()
