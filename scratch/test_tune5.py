"""Test immediate launch with clean trailing wave."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario


def run_launch(gain=4.2, ky=1.8, max_steps=350):
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = generate_scenario("gap_bridge", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    box_height = 0.25
    # Spawn at x = 0.0m (start line)
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = box_height + 0.185
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    min_lat = 0.10

    for step in range(1, max_steps + 1):
        y_now = float(env.data.qpos[1])
        turn_angle = float(np.clip(-ky * y_now, -0.22, 0.22))
        c_t, s_t = np.cos(turn_angle), np.sin(turn_angle)
        d_cmd = np.array([c_t * 1.0 - s_t * 0.0, s_t * 1.0 + c_t * 0.0])

        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T
        u_long = dirs_world[:, 0] * d_cmd[0] + dirs_world[:, 1] * d_cmd[1]
        u_lat = dirs_world[:, 0] * (-d_cmd[1]) + dirs_world[:, 1] * d_cmd[0]
        u_z = dirs_world[:, 2]

        targets = np.zeros(len(env.dirs_body), dtype=np.float32)

        # Pure trailing flank push wave
        rear_flank = (u_long < -0.01) & (np.abs(u_lat) > min_lat) & (u_z < 0.25)
        rear_w = np.clip((-u_long) / 0.85, 0.0, 1.0) ** 0.80
        flank_w = np.clip((np.abs(u_lat) - min_lat) / 0.22, 0.0, 1.0)
        down_w = np.clip(1.0 - np.abs(u_z + 0.35) / 0.80, 0.0, 1.0)

        wave = np.clip(rear_w * flank_w * down_w * gain, 0.0, 1.0)
        targets[rear_flank] = env.max_extend * wave[rear_flank]

        # Central void tuck
        central_void = (np.abs(u_lat) < min_lat + 0.02) & (u_z < 0.12)
        targets[central_void] = 0.0

        # Strict leading-rod cutoff
        targets[u_long > -0.01] = 0.0

        env.step(targets)

        if step % 50 == 0:
            print(f"Step {step:3d}: x={env.data.qpos[0]:.2f}m y={env.data.qpos[1]:+.3f}m z={env.data.qpos[2]:.3f}m vx={env.data.qvel[0]:.2f}m/s")

        if env.data.qpos[0] >= 5.0:
            print(f"🏁 Reached goal at step {step} ({step*0.01:.2f}s)!")
            break

    end_x, end_y, end_z = float(env.data.qpos[0]), float(env.data.qpos[1]), float(env.data.qpos[2])
    print(f"\nFinal: x={end_x:.2f}m y={end_y:+.3f}m z={end_z:.3f}m")
    env.close()

run_launch()
