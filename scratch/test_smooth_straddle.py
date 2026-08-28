"""Test continuous smooth straddle drive on dual boxes from the very start (x=0)."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario


def test_smooth_straddle():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = generate_scenario("gap_bridge", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    box_height = 0.25
    # Start at x = 0.0 (beginning of the platforms)
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = box_height + 0.185
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    d_fwd = np.array([1.0, 0.0])
    ky = 2.2
    u_contact = 0.45
    gain = 2.4

    for step in range(1, 401):
        y_now = float(env.data.qpos[1])
        turn_angle = float(np.clip(-ky * y_now, -0.30, 0.30))
        c_t, s_t = np.cos(turn_angle), np.sin(turn_angle)
        d_cmd = np.array([c_t * 1.0 - s_t * 0.0, s_t * 1.0 + c_t * 0.0])

        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T
        u_long = dirs_world[:, 0] * d_cmd[0] + dirs_world[:, 1] * d_cmd[1]
        u_lat = dirs_world[:, 0] * (-d_cmd[1]) + dirs_world[:, 1] * d_cmd[0]
        u_z = dirs_world[:, 2]

        targets = np.zeros(len(env.dirs_body), dtype=np.float32)

        # Dual-peak contact profile along the two ledges
        flank_left = np.clip(1.0 - 5.0 * ((u_lat - u_contact) ** 2), 0.0, 1.0)
        flank_right = np.clip(1.0 - 5.0 * ((u_lat + u_contact) ** 2), 0.0, 1.0)
        flank = flank_left + flank_right

        # Rear rolling torque wave
        rear = np.clip((-u_long - 0.05) / 0.85, 0.0, 1.0)
        down = np.clip(1.0 - np.abs(u_z + 0.40) / 0.70, 0.0, 1.0)
        wave = np.clip(rear * down * flank * gain, 0.0, 1.0)

        # Base support on the contact flanks to avoid dead spots
        support = flank * np.clip(1.0 - np.abs(u_z + 0.50) / 0.60, 0.0, 1.0) * 0.035
        support[u_long > 0.05] = 0.0

        targets = 0.025 + (env.max_extend - 0.025) * wave + support
        targets = np.clip(targets, 0.0, env.max_extend)

        # Strict central void tuck (hole underneath)
        central_void = (np.abs(u_lat) < 0.18) & (u_z < 0.15)
        targets[central_void] = 0.0

        # Never brake on leading surface
        targets[u_long > 0.02] = 0.0

        env.step(targets)

        if step % 50 == 0:
            print(f"Step {step:3d}: x={env.data.qpos[0]:.2f}m y={env.data.qpos[1]:+.3f}m z={env.data.qpos[2]:.3f}m vx={env.data.qvel[0]:.2f}m/s")

    end_x, end_y, end_z = float(env.data.qpos[0]), float(env.data.qpos[1]), float(env.data.qpos[2])
    print(f"\nFinal: x={end_x:.2f}m y={end_y:+.3f}m z={end_z:.3f}m (deck={box_height:.2f}m)")
    env.close()

test_smooth_straddle()
