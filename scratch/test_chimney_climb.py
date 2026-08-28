"""Test vertical climbing UP and DOWN between two vertical box walls."""
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


def climb_shaft_step(quat, dirs_body, max_extend, direction="up", gain=3.2, brace=0.55, y_offset=0.0, x_offset=0.0):
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T
    u_x = dirs_world[:, 0]
    u_lat = dirs_world[:, 1]
    u_z = dirs_world[:, 2]

    # Contact flank mask: rods pointing laterally into Box 1 (+y) and Box 2 (-y)
    flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)

    # Lateral differential for centering between the two walls
    # If drifting left (y > 0), push more on right wall to push back right
    flank_bias = np.where(u_lat * y_offset > 0, 0.85, 1.15)

    if direction == "up":
        # Push downward on lateral rods to lift body up
        push_z = np.clip((-u_z - 0.02) / 0.85, 0.0, 1.0) ** 0.85
        push = push_z * flank * flank_bias * gain

        # Upper catch & lateral clamp grip
        grip = flank * np.clip(1.0 - np.abs(u_z) / 0.90, 0.0, 1.0) * brace
        wave = np.clip(push + grip, 0.0, 1.0)

        # Retract upper-most rods slightly so they don't jam against wall
        wave[u_z > 0.80] *= 0.3
    else:
        # Descend: push upward or controlled descent
        push_z = np.clip((u_z - 0.02) / 0.85, 0.0, 1.0) ** 0.85
        push = push_z * flank * flank_bias * (gain * 0.7)
        grip = flank * brace * 0.65
        wave = np.clip(push + grip, 0.0, 1.0)

    targets = max_extend * wave

    # Longitudinal stabilization (keep x near 0)
    # Tuck pure front/back rods
    non_flank = (np.abs(u_lat) < 0.22)
    targets[non_flank] = 0.0

    return targets.astype(np.float32)


def run_climbing_test():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=3.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # 1. Climb UP from z = 0.25m to top (z > 2.5m)
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.25
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("=== 1. CLIMBING UP BETWEEN TWO VERTICAL BOXES ===")
    for step in range(1, 351):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        targets = climb_shaft_step(quat, env.dirs_body, env.max_extend, direction="up", y_offset=float(pos[1]), x_offset=float(pos[0]))
        env.step(targets)

        if step % 50 == 0:
            print(f"Step {step:3d} (UP):   z = {pos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {pos[1]*100:+.1f}cm | x = {pos[0]*100:+.1f}cm")

    z_apex = float(env.data.qpos[2])
    print(f"\nReached apex height: z = {z_apex:.3f}m (Climbed +{z_apex - 0.25:.2f}m vertically!)")

    # 2. Climb DOWN from apex
    print("\n=== 2. CLIMBING DOWN BETWEEN TWO VERTICAL BOXES ===")
    for step in range(1, 351):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        targets = climb_shaft_step(quat, env.dirs_body, env.max_extend, direction="down", y_offset=float(pos[1]), x_offset=float(pos[0]))
        env.step(targets)

        if step % 50 == 0:
            print(f"Step {step:3d} (DOWN): z = {pos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {pos[1]*100:+.1f}cm | x = {pos[0]*100:+.1f}cm")

    z_end = float(env.data.qpos[2])
    print(f"\nFinal height after descent: z = {z_end:.3f}m (Descended -{z_apex - z_end:.2f}m!)")
    env.close()


if __name__ == "__main__":
    run_climbing_test()
