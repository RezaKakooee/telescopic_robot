"""Optimize vertical climbing speed and descent between two vertical box walls."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.0, shaft_length=1.2):
    half_gap = shaft_width / 2.0  # 0.18m
    box_w = 0.60
    box_y_left = half_gap + box_w / 2.0
    box_y_right = -half_gap - box_w / 2.0

    steps = [
        # Box 1 (Left vertical box): pos_x, pos_y, hx, hy, height
        [0.0, box_y_left, shaft_length / 2.0, box_w / 2.0, shaft_height],
        # Box 2 (Right vertical box)
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


def climb_shaft(quat, dirs_body, max_extend, step, direction="up", freq=3.5, clamp=0.065, gain=1.0, y_err=0.0):
    t_sec = step * 0.01
    omega = 2.0 * np.pi * freq

    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T
    u_x = dirs_world[:, 0]
    u_lat = dirs_world[:, 1]
    u_z = dirs_world[:, 2]

    # Contact flank mask: rods pointing laterally into Box 1 (+y) and Box 2 (-y)
    flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)

    # Active centering between left and right walls
    # If y > 0 (drifting towards left wall), push harder on left wall to bounce right
    side_centering = np.where(u_lat * y_err > 0, 1.25, 0.75)

    # Direction: +1 for UP, -1 for DOWN
    sign = +1.0 if direction == "up" else -1.0

    # Traveling wave along z
    # Upward climb: phase = omega*t + 3.2*u_z
    # Downward descent: phase = omega*t - 3.2*u_z
    phase = omega * t_sec + sign * 3.2 * u_z
    wave_norm = np.clip(0.5 + 0.5 * np.sin(phase), 0.0, 1.0) ** 0.85

    # Target extension
    targets = flank * side_centering * (clamp + (max_extend - clamp) * wave_norm * gain)
    targets = np.clip(targets, 0.0, max_extend)

    # Retract front/back rods
    targets[np.abs(u_lat) < 0.22] = 0.0

    return targets.astype(np.float32)


def test_up_and_down():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.0)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Start at z = 0.30m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.30
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("=== 1. CLIMBING UP (z = 0.30m -> Top) ===")
    for step in range(1, 401):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        t = climb_shaft(quat, env.dirs_body, env.max_extend, step, direction="up", freq=3.5, y_err=float(pos[1]))
        env.step(t)

        if step % 50 == 0:
            print(f"Step {step:3d} (UP):   z = {pos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {pos[1]*100:+.1f}cm")

    z_peak = float(env.data.qpos[2])
    print(f"\nReached Peak Height: z = {z_peak:.3f}m (Climbed +{z_peak - 0.30:.2f}m vertically!)")

    print("\n=== 2. CLIMBING DOWN (z = Peak -> Bottom) ===")
    for step in range(1, 401):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        t = climb_shaft(quat, env.dirs_body, env.max_extend, step, direction="down", freq=3.5, y_err=float(pos[1]))
        env.step(t)

        if step % 50 == 0:
            print(f"Step {step:3d} (DOWN): z = {pos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {pos[1]*100:+.1f}cm")

    z_bottom = float(env.data.qpos[2])
    print(f"\nReached Bottom Height: z = {z_bottom:.3f}m (Descended -{z_peak - z_bottom:.2f}m vertically!)")
    env.close()


if __name__ == "__main__":
    test_up_and_down()
