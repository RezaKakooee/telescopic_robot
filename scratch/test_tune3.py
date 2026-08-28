"""Test wide-angle flank drive for super-smooth non-stop rolling."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario


def run_continuous_straddle3(gain=3.5, ky=2.0, max_steps=400):
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = generate_scenario("gap_bridge", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    box_height = 0.25
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = box_height + 0.19
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    min_lat = 0.10

    for step in range(1, max_steps + 1):
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

        # Flank push wave: wide rear-flank envelope
        rear_w = np.clip((-u_long + 0.02) / 0.88, 0.0, 1.0) ** 0.90
        flank_w = np.clip((np.abs(u_lat) - min_lat) / 0.28, 0.0, 1.0)
        down_w = np.clip(1.0 - np.abs(u_z + 0.30) / 0.85, 0.0, 1.0)

        # Centering bias
        side_bias = np.where(u_lat * y_now > 0, 0.85, 1.15)

        wave = np.clip(rear_w * flank_w * down_w * gain * side_bias, 0.0, 1.0)

        # Central void cutoff
        central_void = (np.abs(u_lat) < min_lat + 0.02) & (u_z < 0.10)
        wave[central_void] = 0.0
        wave[u_long > 0.04] = 0.0
        wave[u_z > 0.30] = 0.0

        targets = 0.015 + (env.max_extend - 0.015) * wave
        targets[central_void] = 0.0
        targets[u_long > 0.02] = 0.0

        env.step(targets)

        if step % 50 == 0:
            print(f"Step {step:3d}: x={env.data.qpos[0]:.2f}m y={env.data.qpos[1]:+.3f}m z={env.data.qpos[2]:.3f}m vx={env.data.qvel[0]:.2f}m/s")

    end_x, end_y, end_z = float(env.data.qpos[0]), float(env.data.qpos[1]), float(env.data.qpos[2])
    print(f"\nFinal: x={end_x:.2f}m y={end_y:+.3f}m z={end_z:.3f}m")
    env.close()

run_continuous_straddle3()
