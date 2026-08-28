"""Test high-traction climbing robot between vertical boxes."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.34, shaft_height=4.0, shaft_length=0.9):
    half_gap = shaft_width / 2.0  # 0.17m
    box_w = 0.40
    box_y_left = half_gap + box_w / 2.0
    box_y_right = -half_gap - box_w / 2.0

    steps = [
        # Box 1 (Left vertical box): pos_x, pos_y, hx, hy, height
        [0.0, box_y_left, shaft_length / 2.0, box_w / 2.0, shaft_height],
        # Box 2 (Right vertical box)
        [0.0, box_y_right, shaft_length / 2.0, box_w / 2.0, shaft_height],
    ]

    # Yardlines every 25cm
    yardlines = []
    for h in np.arange(0.25, shaft_height, 0.25):
        color = "0.96 0.78 0.08 1.0" if int(round(h * 100)) % 100 == 0 else "0.95 0.95 0.95 0.7"
        yardlines.append([0.0, half_gap + 0.005, shaft_length / 2.0 * 0.95, 0.015, color])
        yardlines.append([0.0, -half_gap - 0.005, shaft_length / 2.0 * 0.95, 0.015, color])

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
        yardlines=yardlines,
    )


def test_climb():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.34, shaft_height=4.0)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn at z = 0.25m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.25
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("=== Testing Vertical Climbing Propulsion ===")
    for step in range(1, 401):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_x = dirs[:, 0]
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]

        flank = np.clip((np.abs(u_lat) - 0.18) / 0.30, 0.0, 1.0)
        y_now = float(env.data.qpos[1])
        side_centering = np.where(u_lat * y_now > 0, 1.20, 0.80)

        # Traveling wave: downward moving wave along z
        omega = 2.0 * np.pi * 3.0
        phase = omega * (step * 0.01) + 3.0 * u_z
        wave = np.clip(0.5 + 0.5 * np.sin(phase), 0.0, 1.0) ** 0.8

        clamp = 0.055
        targets = flank * side_centering * (clamp + (env.max_extend - clamp) * wave)
        targets = np.clip(targets, 0.0, env.max_extend)
        targets[np.abs(u_lat) < 0.20] = 0.0

        env.step(targets)

        if step % 50 == 0:
            print(f"Step {step:3d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    env.close()

test_climb()
