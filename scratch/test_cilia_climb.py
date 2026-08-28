"""Test true vertical climbing: push rods extend, return rods FULLY RETRACT."""
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


def test_climb():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn resting on floor at z = 0.20m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.20
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Initial pos:", env.data.qpos[0:3])

    # Traveling peristaltic climbing wave with FULL RETRACTION of return rods
    # Wave frequency: 3.0 Hz
    freq = 3.0
    omega = 2.0 * np.pi * freq

    for step in range(1, 401):
        t_sec = step * 0.01
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_x = dirs[:, 0]
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]

        flank = np.clip((np.abs(u_lat) - 0.22) / 0.30, 0.0, 1.0)
        y_now = float(env.data.qpos[1])
        side_centering = np.where(u_lat * y_now > 0, 1.15, 0.85)

        targets = np.zeros(len(env.dirs_body), dtype=np.float32)

        # Traveling wave along z:
        # Phase = omega * t + 3.0 * u_z
        phase = omega * t_sec + 3.0 * u_z
        wave = np.sin(phase)

        # Only extend when wave > 0 (push phase)
        # Full zero retraction when wave <= 0 (return phase)
        push_stroke = np.clip(wave, 0.0, 1.0) ** 0.6

        # Downward emphasis: rods pushing downward (u_z < 0.2)
        down_weight = np.clip((-u_z + 0.3) / 1.1, 0.0, 1.0)

        targets = flank * side_centering * (env.max_extend * push_stroke * down_weight)
        targets = np.clip(targets, 0.0, env.max_extend)
        targets[np.abs(u_lat) < 0.22] = 0.0

        env.step(targets)

        if step % 40 == 0:
            print(f"Step {step:3d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_apex = float(env.data.qpos[2])
    print(f"\nFinal height: z = {z_apex:.3f}m (Δz = {z_apex - 0.20:+.3f}m)")
    env.close()

test_climb()
