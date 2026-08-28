"""Test true chimney jumping: explosive launch + apex wall catch."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5, shaft_length=1.2):
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


def test():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn on the floor at z = 0.20m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.20
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Initial pos:", env.data.qpos[0:3])

    # 1. Crouch (5 steps): all rods 0
    for _ in range(5):
        env.step(np.zeros(len(env.dirs_body), dtype=np.float32))

    # 2. Takeoff (8 steps): all downward rods extend to max_extend (0.16m)
    for _ in range(8):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_z = dirs[:, 2]
        t = np.zeros(len(env.dirs_body), dtype=np.float32)
        # Blast bottom rods (u_z < 0.0)
        t[u_z < 0.0] = env.max_extend
        env.step(t)

    print(f"Takeoff speed: vz = {env.data.qvel[2]:+.2f} m/s, z = {env.data.qpos[2]:.3f} m")

    # 3. Airborne ascent (retract rods to fly cleanly through the chimney)
    for step in range(25):
        t = np.zeros(len(env.dirs_body), dtype=np.float32)
        env.step(t)
        if step % 5 == 0:
            print(f"  Ascent step {step:2d}: z = {env.data.qpos[2]:.3f} m | vz = {env.data.qvel[2]:+.2f} m/s")

    # 4. Apex Wall Catch: CLAMP against Box 1 and Box 2
    print("Apex Catch: Clamping against Box 1 & Box 2...")
    for step in range(50):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        # Extend lateral rods to clamp walls
        t = flank * 0.088
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)

    print(f"Final Locked Height: z = {env.data.qpos[2]:.3f} m | vz = {env.data.qvel[2]:+.2f} m/s | y = {env.data.qpos[1]*100:+.1f} cm")
    env.close()

test()
