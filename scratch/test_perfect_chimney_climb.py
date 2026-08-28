"""Test true multi-meter chimney vertical climbing and descent."""
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


def test_climb_flow():
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.38, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn resting on floor at z = 0.22m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Initial pos:", env.data.qpos[0:3])

    # 1. Stand at rest (15 steps)
    for _ in range(15):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        t = np.zeros(env.n_bars, dtype=np.float32)
        t[dirs[:, 2] < -0.3] = 0.045
        env.step(t)

    print(f"Stand rest: z = {env.data.qpos[2]:.3f}m")

    # 2. Crouch (10 steps)
    for _ in range(10):
        env.step(np.zeros(env.n_bars, dtype=np.float32))

    print(f"Crouch: z = {env.data.qpos[2]:.3f}m")

    # 3. Takeoff Launch 1 (12 steps): blast all ground rods
    for _ in range(12):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_z = dirs[:, 2]
        t = np.zeros(env.n_bars, dtype=np.float32)
        t[u_z < 0.10] = env.max_extend
        t[u_z > 0.15] = 0.0
        env.step(t)

    print(f"🚀 Takeoff: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # 4. Flight into Chimney (tuck rods)
    for step in range(14):
        env.step(np.full(env.n_bars, 0.015, dtype=np.float32))
        if step % 4 == 0:
            print(f"  Ascent: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # 5. Apex Wall Catch & Lock: clamp lateral rods into Box 1 & Box 2
    print("🔒 Apex Wall Lock: Clamping against Box 1 & Box 2...")
    for step in range(50):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.090
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)
        if step % 15 == 0:
            print(f"  Suspension: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_apex = float(env.data.qpos[2])
    print(f"✅ Apex Locked in Mid-Air: z = {z_apex:.3f}m (Climbed +{z_apex - 0.22:.2f}m up into chimney!)")

    # 6. Controlled Glide Descent: slide down between the boxes
    print("🪂 Controlled Glide Descent...")
    for step in range(50):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.038
        if env.data.qpos[2] < 0.26:
            t[u_z < -0.20] = 0.055
        env.step(t)
        if step % 15 == 0:
            print(f"  Descent: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    z_land = float(env.data.qpos[2])
    print(f"🛬 Touchdown: z = {z_land:.3f}m (Descended -{z_apex - z_land:.2f}m down!)")
    env.close()

test_climb_flow()
