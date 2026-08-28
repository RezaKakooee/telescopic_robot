"""Test step-by-step climbing cycle between two vertical boxes."""
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
        [-shaft_length / 2.0 - 0.2, -1.2, shaft_length / 2.0 + 0.2, -1.2],
        [shaft_length / 2.0 + 0.2, -1.2, shaft_length / 2.0 + 0.2, 1.2],
        [shaft_length / 2.0 + 0.2, 1.2, -shaft_length / 2.0 - 0.2, 1.2],
        [-shaft_length / 2.0 - 0.2, 1.2, -shaft_length / 2.0 - 0.2, -1.2],
    ])

    return Scenario(
        kind="chimney", name="chimney_climb",
        spawn_xy=np.array([0.0, 0.0]), goal=np.array([0.0, 0.0]),
        path_pts=np.array([[0.0, 0.0], [0.0, 0.0]]), markers=np.array([[0.0, 0.0]]),
        path_length=shaft_height,
        walls=walls,
        steps=steps,
    )


def test_climb_step_by_step():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn at z = 0.50m (in the chimney)
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.50
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Initial height: z = 0.50m")

    # 1. First stabilize clamp in mid-air
    for _ in range(30):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.075
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)

    print(f"Stabilized at z = {env.data.qpos[2]:.3f}m")

    # Run 10 climbing cycles
    for cycle in range(1, 11):
        # Phase 1: Upper rods anchor (e=0.085m), lower rods retract to 0.02m (15 steps)
        for _ in range(15):
            quat = env.data.qpos[3:7].copy()
            R = quat_to_rotmat(quat)
            dirs = env.dirs_body @ R.T
            u_lat = dirs[:, 1]
            u_z = dirs[:, 2]
            flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)

            t = np.zeros(len(env.dirs_body), dtype=np.float32)
            upper = (u_z >= -0.05) & (flank > 0.3)
            lower = (u_z < -0.05) & (flank > 0.3)
            t[upper] = 0.085
            t[lower] = 0.020
            t[np.abs(u_lat) < 0.22] = 0.0
            env.step(t)

        # Phase 2: Lower rods push HARD downward (e=0.16m), upper rods slide (15 steps)
        for _ in range(15):
            quat = env.data.qpos[3:7].copy()
            R = quat_to_rotmat(quat)
            dirs = env.dirs_body @ R.T
            u_lat = dirs[:, 1]
            u_z = dirs[:, 2]
            flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)

            t = np.zeros(len(env.dirs_body), dtype=np.float32)
            upper = (u_z >= -0.05) & (flank > 0.3)
            lower = (u_z < -0.05) & (flank > 0.3)
            t[lower] = env.max_extend  # 0.16m push
            t[upper] = 0.035           # light grip
            t[np.abs(u_lat) < 0.22] = 0.0
            env.step(t)

        # Phase 3: Lock upper rods at higher position (10 steps)
        for _ in range(10):
            quat = env.data.qpos[3:7].copy()
            R = quat_to_rotmat(quat)
            dirs = env.dirs_body @ R.T
            u_lat = dirs[:, 1]
            flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)

            t = flank * 0.080
            t[np.abs(u_lat) < 0.22] = 0.0
            env.step(t)

        print(f"Cycle {cycle:2d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_end = float(env.data.qpos[2])
    print(f"\nFinal height: z = {z_end:.3f}m (Δz = {z_end - 0.50:+.3f}m)")
    env.close()

test_climb_step_by_step()
