"""Test true vertical climbing move between two vertical boxes."""
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


def climb_move(quat, dirs_body, max_extend, direction="up", gain=3.5, y_offset=0.0):
    R = quat_to_rotmat(quat)
    dirs = dirs_body @ R.T
    u_x = dirs[:, 0]
    u_lat = dirs[:, 1]
    u_z = dirs[:, 2]

    # Lateral centering
    side_centering = np.where(u_lat * y_offset > 0, 1.15, 0.85)

    if direction == "up":
        # Trailing direction along vertical wall is DOWN (-z)
        push_down = np.clip((-u_z - 0.05) / 0.90, 0.0, 1.0)
        into_wall = np.clip(1.0 - np.abs(np.abs(u_lat) - 0.70) / 0.85, 0.0, 1.0)
        tuck_long = np.clip(1.0 - 2.0 * (u_x ** 2), 0.0, 1.0)

        wave = np.clip((push_down ** 0.9) * into_wall * gain * tuck_long * side_centering, 0.0, 1.0)
        wave[u_z > 0.05] = 0.0
        wave[np.abs(u_lat) < 0.22] = 0.0

        # Add light lateral clamp so it doesn't lose wall contact
        clamp = np.clip((np.abs(u_lat) - 0.25) / 0.35, 0.0, 1.0) * 0.06
        targets = np.maximum(clamp, max_extend * wave)
    elif direction == "down":
        # Trailing direction along vertical wall is UP (+z)
        push_up = np.clip((u_z - 0.05) / 0.90, 0.0, 1.0)
        into_wall = np.clip(1.0 - np.abs(np.abs(u_lat) - 0.70) / 0.85, 0.0, 1.0)
        tuck_long = np.clip(1.0 - 2.0 * (u_x ** 2), 0.0, 1.0)

        wave = np.clip((push_up ** 0.9) * into_wall * (gain * 0.7) * tuck_long * side_centering, 0.0, 1.0)
        wave[u_z < -0.05] = 0.0
        wave[np.abs(u_lat) < 0.22] = 0.0

        clamp = np.clip((np.abs(u_lat) - 0.25) / 0.35, 0.0, 1.0) * 0.06
        targets = np.maximum(clamp, max_extend * wave)
    else:
        # Hold
        clamp = np.clip((np.abs(u_lat) - 0.25) / 0.35, 0.0, 1.0) * 0.075
        targets = clamp

    targets = np.clip(targets, 0.0, max_extend)
    targets[np.abs(u_lat) < 0.20] = 0.0
    return targets.astype(np.float32)


def test_climb_movement():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.34, shaft_height=4.0)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn ball resting on the floor between the two boxes at z = 0.20m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.20
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("=== Testing Climbing UP from Floor ===")
    for step in range(1, 301):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        t = climb_move(quat, env.dirs_body, env.max_extend, direction="up", gain=3.8, y_offset=float(pos[1]))
        env.step(t)

        if step % 50 == 0:
            print(f"Step {step:3d} (UP):   z = {pos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {pos[1]*100:+.1f}cm")

    z_apex = float(env.data.qpos[2])
    print(f"\nReached Apex Height: z = {z_apex:.3f}m (Climbed +{z_apex - 0.20:.2f}m up!)")

    print("\n=== Testing Climbing DOWN ===")
    for step in range(1, 301):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        t = climb_move(quat, env.dirs_body, env.max_extend, direction="down", gain=3.8, y_offset=float(pos[1]))
        env.step(t)

        if step % 50 == 0:
            print(f"Step {step:3d} (DOWN): z = {pos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {pos[1]*100:+.1f}cm")

    z_down = float(env.data.qpos[2])
    print(f"\nReached Bottom Height: z = {z_down:.3f}m (Descended -{z_apex - z_down:.2f}m down!)")
    env.close()

test_climb_movement()
