"""Test rotating continuous vertical climb between two vertical box walls."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5, shaft_length=1.2):
    half_gap = shaft_width / 2.0
    box_w = 0.60
    box_y_left = half_gap + box_w / 2.0
    box_y_right = -half_gap - box_w / 2.0

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


def test_rotating_climb():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.50
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    # Let's test a traveling wave that rotates with frequency 2.0 Hz
    # and has active pitch torque bias
    freq = 2.0
    omega = 2.0 * np.pi * freq

    for step in range(1, 401):
        t = step * 0.01
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_x = dirs[:, 0]
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]

        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        y_now = float(env.data.qpos[1])
        side_centering = np.where(u_lat * y_now > 0, 1.20, 0.80)

        # Cylindrical angle around X axis: theta = arctan2(u_z, u_lat)
        # Or spherical wave moving downward along z:
        # We want rods to extend when moving downward and retract when moving upward!
        theta_vert = np.arctan2(u_z, np.abs(u_lat))  # angle from horizontal wall normal
        # Climbing wave phase:
        wave_phase = theta_vert + omega * t

        # Power stroke: rod extends when theta_vert is transitioning downward
        # Return stroke: rod retracts when theta_vert is moving upward
        stroke_factor = np.clip(0.5 + 0.5 * np.cos(wave_phase), 0.0, 1.0) ** 0.75

        # Base clamp to maintain continuous grip
        clamp_base = 0.065

        targets = flank * side_centering * (clamp_base + (env.max_extend - clamp_base) * stroke_factor)
        targets = np.clip(targets, 0.0, env.max_extend)
        targets[np.abs(u_lat) < 0.22] = 0.0

        env.step(targets)

        if step % 40 == 0:
            print(f"Step {step:3d}: z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm | wy = {env.data.qvel[4]:+.2f}rad/s")

    z_end = float(env.data.qpos[2])
    print(f"\nFinal height: z = {z_end:.3f}m (Δz = {z_end - 0.50:+.3f}m)")
    env.close()

test_rotating_climb()
