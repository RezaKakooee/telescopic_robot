"""Test up-hold-down-up vertical motion cycle between two vertical boxes."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.0, shaft_length=1.0):
    half_gap = shaft_width / 2.0  # 0.18m
    box_w = 0.50
    box_y_left = half_gap + box_w / 2.0    # +0.43m
    box_y_right = -half_gap - box_w / 2.0  # -0.43m

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
        spawn_xy=np.array([0.0, 0.0]), goal=np.array([0.0, 0.0]),
        path_pts=np.array([[0.0, 0.0], [0.0, 0.0]]), markers=np.array([[0.0, 0.0]]),
        path_length=shaft_height,
        walls=walls,
        steps=steps,
        yardlines=yardlines,
    )


def climb_control(quat, dirs_body, max_extend, step, mode="up", y_offset=0.0):
    R = quat_to_rotmat(quat)
    dirs = dirs_body @ R.T
    u_lat = dirs[:, 1]
    u_z = dirs[:, 2]

    flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
    side_centering = np.where(u_lat * y_offset > 0, 1.20, 0.80)

    targets = np.zeros(len(dirs_body), dtype=np.float32)

    if mode == "up":
        # Push downward on lateral rods to lift core UP
        push_down = np.clip((-u_z - 0.02) / 0.85, 0.0, 1.0) ** 0.85
        push = push_down * flank * side_centering * (1.1 * max_extend)
        clamp = flank * side_centering * 0.055
        targets = np.maximum(clamp, push)
    elif mode == "down":
        # Push upward on lateral rods to drive core DOWN
        push_up = np.clip((u_z - 0.02) / 0.85, 0.0, 1.0) ** 0.85
        push = push_up * flank * side_centering * (1.0 * max_extend)
        clamp = flank * side_centering * 0.040
        targets = np.maximum(clamp, push)
    elif mode == "hold":
        # Neutral clamp to hold static in mid-air
        targets = flank * side_centering * 0.065

    targets = np.clip(targets, 0.0, max_extend)
    targets[np.abs(u_lat) < 0.22] = 0.0
    return targets.astype(np.float32)


def run_cycle():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.0)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn at z = 0.40m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.40
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("=== 1. CLIMBING UP (Pushing Both Boxes Downward) ===")
    for step in range(1, 151):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        t = climb_control(quat, env.dirs_body, env.max_extend, step, mode="up", y_offset=float(pos[1]))
        env.step(t)
        if step % 30 == 0:
            print(f"Step {step:3d} (UP):   z = {pos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {pos[1]*100:+.1f}cm")

    print("\n=== 2. MID-AIR SUSPENSION (Holding Both Boxes) ===")
    for step in range(151, 251):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        t = climb_control(quat, env.dirs_body, env.max_extend, step, mode="hold", y_offset=float(pos[1]))
        env.step(t)
        if step % 30 == 0:
            print(f"Step {step:3d} (HOLD): z = {pos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {pos[1]*100:+.1f}cm")

    print("\n=== 3. CLIMBING DOWN (Pushing Both Boxes Upward) ===")
    for step in range(251, 401):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        t = climb_control(quat, env.dirs_body, env.max_extend, step, mode="down", y_offset=float(pos[1]))
        env.step(t)
        if step % 30 == 0:
            print(f"Step {step:3d} (DOWN): z = {pos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {pos[1]*100:+.1f}cm")

    print("\n=== 4. CLIMBING UP AGAIN ===")
    for step in range(401, 551):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        t = climb_control(quat, env.dirs_body, env.max_extend, step, mode="up", y_offset=float(pos[1]))
        env.step(t)
        if step % 30 == 0:
            print(f"Step {step:3d} (UP):   z = {pos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {pos[1]*100:+.1f}cm")

    env.close()

run_cycle()
