"""Test continuous vertical ratcheting climb."""
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


def test_power_stroke():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.0)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.30
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    freq = 2.5
    omega = 2.0 * np.pi * freq

    for step in range(1, 501):
        t_sec = step * 0.01
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T
        u_x = dirs_world[:, 0]
        u_lat = dirs_world[:, 1]
        u_z = dirs_world[:, 2]

        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        y_now = float(env.data.qpos[1])
        side_centering = np.where(u_lat * y_now > 0, 1.25, 0.75)

        # Cyclic vertical phase:
        # Downward velocity on wall is proportional to cos(omega*t + k*u_z)
        # Power stroke: when rod phase is in downward push window
        phase = omega * t_sec + 2.5 * u_z
        # Asymmetrical wave: sharp downward push (power stroke), soft retracted return
        push_phase = np.clip(np.sin(phase), 0.0, 1.0) ** 0.6
        # Downward quadrant emphasis (push downward against wall)
        down_weight = np.clip((-u_z + 0.35) / 1.0, 0.0, 1.0)

        # Baseline clamp to never drop
        clamp_base = 0.05

        stroke = clamp_base + (env.max_extend - clamp_base) * (push_phase * down_weight)
        targets = flank * side_centering * stroke
        targets = np.clip(targets, 0.0, env.max_extend)

        # Retract front/back
        targets[np.abs(u_lat) < 0.22] = 0.0

        env.step(targets)

        if step % 50 == 0:
            print(f"Step {step:3d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_end = float(env.data.qpos[2])
    print(f"\nFinal height: z = {z_end:.3f}m (Δz = {z_end - 0.30:+.3f}m)")
    env.close()

test_power_stroke()
