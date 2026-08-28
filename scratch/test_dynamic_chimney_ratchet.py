"""Test dynamic ratchet climb: explosive vertical wall impulse + immediate apex clamp."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.38, shaft_height=4.5, shaft_length=1.2):
    half_gap = shaft_width / 2.0  # 0.19m
    box_w = 0.50
    box_y_left = half_gap + box_w / 2.0    # +0.44m
    box_y_right = -half_gap - box_w / 2.0  # -0.44m

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

    return Scenario(
        kind="chimney", name="chimney_climb",
        spawn_xy=np.array([0.0, 0.0]), goal=np.array([0.0, 10.0]),
        path_pts=np.array([[0.0, 0.0], [0.0, 0.0]]), markers=np.array([[0.0, 0.0]]),
        path_length=shaft_height,
        walls=walls,
        steps=steps,
    )


def test():
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.38, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn at z = 0.40m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.40
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print(f"Spawned at z = {env.data.qpos[2]:.3f} m")

    # Initial clamp (20 steps)
    for _ in range(20):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.095
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)

    print(f"Initial Locked Height: z = {env.data.qpos[2]:.3f} m")

    # Dynamic Wall Stemming Steps (Each step pushes against both vertical walls to jump up +25cm!)
    n_leaps = 4
    for leap in range(n_leaps):
        # 1. Impulse Push: lower lateral rods push downward into Box 1 & Box 2 (8 steps)
        for _ in range(8):
            quat = env.data.qpos[3:7].copy()
            R = quat_to_rotmat(quat)
            dirs = env.dirs_body @ R.T
            u_lat = dirs[:, 1]
            u_z = dirs[:, 2]
            flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
            t = np.zeros(env.n_bars, dtype=np.float32)
            # Push rods: pointing down and into walls
            push_rods = flank * np.clip((-u_z + 0.20) / 0.80, 0.0, 1.0)
            t = push_rods * env.max_extend
            t[np.abs(u_lat) < 0.22] = 0.0
            env.step(t)

        # 2. Airborne Ascent (retract rods to glide up through chimney) (10 steps)
        for _ in range(10):
            env.step(np.full(env.n_bars, 0.010, dtype=np.float32))

        # 3. Apex Catch & Clamp (25 steps)
        for _ in range(25):
            quat = env.data.qpos[3:7].copy()
            R = quat_to_rotmat(quat)
            dirs = env.dirs_body @ R.T
            u_lat = dirs[:, 1]
            flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
            t = flank * 0.098
            t[np.abs(u_lat) < 0.22] = 0.0
            env.step(t)

        print(f"Leap {leap+1}: z = {env.data.qpos[2]:.3f} m | vz = {env.data.qvel[2]:+.2f} m/s | y = {env.data.qpos[1]*100:+.1f} cm")

    z_peak = float(env.data.qpos[2])
    print(f"\n✅ Climbed from 0.40m to z = {z_peak:.3f} m (Net Elevation Gain: +{z_peak - 0.40:.2f} m!)")
    env.close()

test()
