"""Scratch test: Vertical chimney climbing between two vertical boxes."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=3.5, shaft_length=1.2):
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

    # Front and back retaining guide plates to keep test focused in the slot
    walls = np.array([
        [-shaft_length / 2.0, -1.0, shaft_length / 2.0, -1.0],
        [shaft_length / 2.0, -1.0, shaft_length / 2.0, 1.0],
        [shaft_length / 2.0, 1.0, -shaft_length / 2.0, 1.0],
        [-shaft_length / 2.0, 1.0, -shaft_length / 2.0, -1.0],
    ])

    # Vertical height stripes every 0.5m on both vertical box faces
    yardlines = []
    for h in np.arange(0.5, shaft_height, 0.5):
        # Stripe on left box face
        yardlines.append([0.0, half_gap + 0.01, shaft_length / 2.0 * 0.9, 0.02, "0.96 0.78 0.08 1.0"])
        # Stripe on right box face
        yardlines.append([0.0, -half_gap - 0.01, shaft_length / 2.0 * 0.9, 0.02, "0.96 0.78 0.08 1.0"])

    return Scenario(
        kind="chimney", name="chimney_test",
        spawn_xy=np.array([0.0, 0.0]), goal=np.array([0.0, 0.0]),
        path_pts=np.array([[0.0, 0.0], [0.0, 0.0]]), markers=np.array([[0.0, 0.0]]),
        path_length=shaft_height,
        walls=walls,
        steps=steps,
        yardlines=yardlines,
    )


def test_climb():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=3.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn ball between the two vertical boxes at z = 0.50m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.50
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Initial pos:", env.data.qpos[0:3])

    # Test lateral brace force to see if it holds in mid-air
    for step in range(1, 100):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T
        u_lat = dirs_world[:, 1]
        u_z = dirs_world[:, 2]

        # Extend lateral rods into left (+y) and right (-y) walls
        targets = np.zeros(len(env.dirs_body), dtype=np.float32)
        lateral_mask = np.abs(u_lat) > 0.35
        # Extend lateral rods to brace against walls
        targets[lateral_mask] = env.max_extend * 0.75

        env.step(targets)
        if step % 20 == 0:
            print(f"Brace step {step:2d}: z={env.data.qpos[2]:.3f}m vz={env.data.qvel[2]:.3f}m/s")

    env.close()

test_climb()
