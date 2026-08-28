"""Test continuous two-group inchworm chimney climbing between two vertical boxes."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.38, shaft_height=4.5, shaft_length=1.2):
    half_gap = shaft_width / 2.0  # 0.19m
    box_w = 0.50
    box_y_left = half_gap + box_w / 2.0    # +0.44m
    box_y_right = -half_gap - box_w / 2.0  # -0.44m

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
    scenario = make_chimney_scenario(cfg, shaft_width=0.38, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn suspended between the two boxes at z = 1.00m (mid-air)
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 1.00
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print(f"Spawned at z = {env.data.qpos[2]:.3f} m (mid-air between Box 1 & Box 2)")

    # 1. Initial clamp to lock in mid-air (20 steps)
    for _ in range(25):
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        t = flank * 0.088
        t[np.abs(u_lat) < 0.22] = 0.0
        env.step(t)

    print(f"Initial Mid-Air Lock: z = {env.data.qpos[2]:.3f} m")

    # 2. Continuous Inchworm Climbing UP (15 cycles)
    cycle_len = 20  # 20 env steps per cycle (0.10s)
    n_cycles = 12

    for cycle in range(n_cycles):
        for s in range(cycle_len):
            quat = env.data.qpos[3:7].copy()
            R = quat_to_rotmat(quat)
            dirs = env.dirs_body @ R.T
            u_lat = dirs[:, 1]
            u_z = dirs[:, 2]

            flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
            t = np.zeros(env.n_bars, dtype=np.float32)

            # Phase A: Bottom Push (steps 0 -> 10)
            # Lower-lateral rods extend hard downward into walls; top rods relax to glide
            if s < 10:
                # Lower rods (u_z < 0) push
                lower_flank = flank * np.clip((-u_z + 0.10) / 0.70, 0.0, 1.0)
                upper_flank = flank * np.clip((u_z + 0.10) / 0.70, 0.0, 1.0)
                t = lower_flank * env.max_extend + upper_flank * 0.035
            # Phase B: Top Clamp & Bottom Reset (steps 10 -> 20)
            # Upper-lateral rods clamp hard to lock height; lower rods retract to reset
            else:
                upper_flank = flank * np.clip((u_z + 0.10) / 0.70, 0.0, 1.0)
                t = upper_flank * 0.092

            t[np.abs(u_lat) < 0.22] = 0.0
            env.step(t)

        print(f"Cycle {cycle+1:2d} UP: z = {env.data.qpos[2]:.3f} m | vz = {env.data.qvel[2]:+.2f} m/s | y = {env.data.qpos[1]*100:+.1f} cm")

    z_apex = float(env.data.qpos[2])
    print(f"\n✅ Peak Climbed Height: z = {z_apex:.3f} m (Climbed +{z_apex - 1.00:+.3f} m up!)")

    # 3. Continuous Inchworm Climbing DOWN (10 cycles)
    for cycle in range(10):
        for s in range(cycle_len):
            quat = env.data.qpos[3:7].copy()
            R = quat_to_rotmat(quat)
            dirs = env.dirs_body @ R.T
            u_lat = dirs[:, 1]
            u_z = dirs[:, 2]

            flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
            t = np.zeros(env.n_bars, dtype=np.float32)

            # Phase A: Top Push Downward (steps 0 -> 10)
            if s < 10:
                upper_flank = flank * np.clip((u_z + 0.10) / 0.70, 0.0, 1.0)
                lower_flank = flank * np.clip((-u_z + 0.10) / 0.70, 0.0, 1.0)
                t = upper_flank * env.max_extend + lower_flank * 0.025
            # Phase B: Bottom Clamp & Top Reset (steps 10 -> 20)
            else:
                lower_flank = flank * np.clip((-u_z + 0.10) / 0.70, 0.0, 1.0)
                t = lower_flank * 0.092

            t[np.abs(u_lat) < 0.22] = 0.0
            env.step(t)

        print(f"Cycle {cycle+1:2d} DOWN: z = {env.data.qpos[2]:.3f} m | vz = {env.data.qvel[2]:+.2f} m/s | y = {env.data.qpos[1]*100:+.1f} cm")

    z_end = float(env.data.qpos[2])
    print(f"\n🛬 Final Lowered Height: z = {z_end:.3f} m (Descended -{z_apex - z_end:.3f} m down!)")
    env.close()

test()
