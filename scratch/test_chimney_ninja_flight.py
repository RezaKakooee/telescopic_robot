"""Test tuck-during-takeoff vertical jump and apex lock in chimney."""
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
    scenario = make_chimney_scenario(cfg, shaft_width=0.40, shaft_height=4.0)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    obs, info = env.reset(seed=42)

    print("Initial resting z:", env.data.qpos[2])

    # 1. Stand at rest (20 steps)
    for _ in range(20):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        t = np.zeros(env.n_bars, dtype=np.float32)
        t[dirs[:, 2] < -0.3] = 0.045
        env.step(t)

    print("Stand z:", env.data.qpos[2])

    # 2. Crouch (12 steps)
    for _ in range(12):
        env.step(np.zeros(env.n_bars, dtype=np.float32))

    print("Crouch z:", env.data.qpos[2])

    # 3. Takeoff (12 steps): Blast downward ground rods only (u_z < 0.0 and |u_lat| < 0.45)
    for _ in range(12):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]

        t = np.zeros(env.n_bars, dtype=np.float32)
        # Ground cluster: points down, avoids smashing into walls
        ground = (u_z < 0.0) & (np.abs(u_lat) < 0.45)
        t[ground] = env.max_extend
        env.step(t)

    print(f"🚀 Takeoff: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # 4. Flight into Chimney (18 steps)
    max_flight_z = 0.0
    for step in range(18):
        env.step(np.full(env.n_bars, 0.010, dtype=np.float32))
        max_flight_z = max(max_flight_z, float(env.data.qpos[2]))
        if step % 3 == 0:
            print(f"  ✈️ Flight step {step:2d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    print(f"Peak Flight Apex: z = {max_flight_z:.3f}m (Net Lift +{max_flight_z - 0.17:.2f}m!)")

    # 5. Wall Catch & Lock: Clamp lateral rods into Box 1 & Box 2
    print("🔒 Wall Catch: Clamping against Box 1 & Box 2...")
    for step in range(60):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.098
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)
        if step % 15 == 0:
            print(f"  Locked Suspension: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_locked = float(env.data.qpos[2])
    print(f"✅ Suspended in Mid-Air: z = {z_locked:.3f}m")

    # 6. Glide Descent
    print("🪂 Glide Descent...")
    for step in range(60):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.038
        if env.data.qpos[2] < 0.25:
            t[u_z < -0.20] = 0.055
        env.step(t)
        if step % 15 == 0:
            print(f"  Descent: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    print(f"🛬 Landed on ground: z = {env.data.qpos[2]:.3f}m")
    env.close()

test()
