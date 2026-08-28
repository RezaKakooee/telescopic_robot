"""Test complete dynamic vertical jump, apex lock, and glide descent between two vertical boxes."""
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

    return Scenario(
        kind="chimney", name="chimney_climb",
        spawn_xy=np.array([0.0, 0.0]), goal=np.array([0.0, 0.0]),
        path_pts=np.array([[0.0, 0.0], [0.0, 0.0]]), markers=np.array([[0.0, 0.0]]),
        path_length=shaft_height,
        walls=walls,
        steps=steps,
    )


def run_demo():
    cfg = load_config("configs/rl/standing_jump_showcase.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.0)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    obs, info = env.reset(seed=42)

    # Spawn resting on floor at x=0, y=0, z=0.20m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.20
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Initial state: z =", env.data.qpos[2])

    # 1. Stand at rest (steps 1-25)
    for _ in range(25):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        t = np.zeros(env.n_bars, dtype=np.float32)
        t[dirs[:, 2] < -0.3] = 0.045
        env.step(t)

    print("Pre-jump stand: z =", env.data.qpos[2])

    # 2. Crouch (steps 25-35): retract all rods to 0
    for _ in range(10):
        env.step(np.zeros(env.n_bars, dtype=np.float32))

    print("Crouch: z =", env.data.qpos[2])

    # 3. Takeoff Launch (steps 35-47): all ground rods explode to 100% stroke
    for _ in range(12):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_z = dirs[:, 2]
        t = np.zeros(env.n_bars, dtype=np.float32)
        t[u_z < 0.10] = env.max_extend
        t[u_z > 0.15] = 0.0
        env.step(t)

    print(f"🚀 Takeoff Launch: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # 4. Flight into Chimney (tuck rods) until near apex (15 steps)
    for _ in range(15):
        env.step(np.full(env.n_bars, 0.015, dtype=np.float32))

    print(f"✈️ Apex Flight: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    # 5. Wall Clamp / Mid-Air Lock (steps 62-120): clamp lateral rods into Box 1 & Box 2
    print("🔒 Clamping against Box 1 & Box 2...")
    for step in range(60):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.088
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)
        if step % 20 == 0:
            print(f"  Suspension step {step:2d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_apex = float(env.data.qpos[2])
    print(f"Apex Locked: z = {z_apex:.3f}m")

    # 6. Controlled Glide Descent (steps 122-170): relax clamp to slide down
    print("🪂 Controlled Descent...")
    for step in range(50):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        # Compliant descent friction braking
        t = flank * 0.038
        # Extend landing gear when nearing floor
        if env.data.qpos[2] < 0.30:
            t[u_z < -0.20] = 0.055
        env.step(t)
        if step % 15 == 0:
            print(f"  Descent step {step:2d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s")

    print(f"🛬 Landed: z = {env.data.qpos[2]:.3f}m")
    env.close()

run_demo()
