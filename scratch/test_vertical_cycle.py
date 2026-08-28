"""Test cyclical vertical climbing between two vertical walls."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=3.5, shaft_length=0.8):
    half_gap = shaft_width / 2.0  # 0.18m
    box_w = 0.60
    box_y_left = half_gap + box_w / 2.0
    box_y_right = -half_gap - box_w / 2.0

    steps = [
        [0.0, box_y_left, shaft_length / 2.0, box_w / 2.0, shaft_height],
        [0.0, box_y_right, shaft_length / 2.0, box_w / 2.0, shaft_height],
    ]

    # Guide slot to keep x = 0
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


def test_climb_cycle():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=3.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn ball between walls at z = 0.50m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.50
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Initial height: z = 0.50m")

    # Wave-based climbing: harmonic traveling wave moving upwards along the sphere
    # Rod target = A0 * flank + A_wave * flank * sin(omega * t - k * u_z)
    omega = 2.0 * np.pi * 1.5  # 1.5 Hz climb frequency
    dt = 0.01

    for step in range(1, 301):
        t_sec = step * dt
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T
        u_x = dirs_world[:, 0]
        u_lat = dirs_world[:, 1]
        u_z = dirs_world[:, 2]

        targets = np.zeros(len(env.dirs_body), dtype=np.float32)

        # Flank mask (left & right wall contact)
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)

        # Upward traveling wave: phase = omega * t + 3.0 * u_z
        # When wave moves upward, the downward-pushing phase propels the body UP!
        phase = omega * t_sec + 2.5 * u_z
        wave_raw = 0.5 + 0.5 * np.sin(phase)

        # Baseline clamp to maintain friction hold throughout cycle
        clamp_base = 0.075

        # Combined stroke
        targets = flank * (clamp_base + (env.max_extend - clamp_base) * wave_raw)
        targets = np.clip(targets, 0.0, env.max_extend)

        # Keep x centered (tuck front/back rods)
        targets[np.abs(u_lat) < 0.25] = 0.0

        env.step(targets)

        if step % 50 == 0:
            print(f"Step {step:3d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_final = float(env.data.qpos[2])
    print(f"\nFinal height: z = {z_final:.3f}m (Δz = {z_final - 0.50:+.3f}m)")
    env.close()

test_climb_cycle()
