"""Test dynamic ninja chimney climbing: multi-meter vertical climb between two boxes."""
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
    box_w = 0.60
    box_y_left = half_gap + box_w / 2.0    # +0.48m
    box_y_right = -half_gap - box_w / 2.0  # -0.48m

    steps = [
        # Box 1 (Left vertical box): pos_x, pos_y, hx, hy, height
        [0.0, box_y_left, shaft_length / 2.0, box_w / 2.0, shaft_height],
        # Box 2 (Right vertical box)
        [0.0, box_y_right, shaft_length / 2.0, box_w / 2.0, shaft_height],
    ]

    walls = np.array([
        [-shaft_length / 2.0 - 0.2, -1.2, shaft_length / 2.0 + 0.2, -1.2],
        [shaft_length / 2.0 + 0.2, -1.2, shaft_length / 2.0 + 0.2, 1.2],
        [shaft_length / 2.0 + 0.2, 1.2, -shaft_length / 2.0 - 0.2, 1.2],
        [-shaft_length / 2.0 - 0.2, 1.2, -shaft_length / 2.0 - 0.2, -1.2],
    ])

    yardlines = []
    for h in np.arange(0.5, shaft_height, 0.5):
        color = "0.96 0.78 0.08 1.0" if int(round(h * 10)) % 10 == 0 else "0.95 0.95 0.95 0.8"
        yardlines.append([0.0, half_gap + 0.01, shaft_length / 2.0 * 0.9, 0.02, color])
        yardlines.append([0.0, -half_gap - 0.01, shaft_length / 2.0 * 0.9, 0.02, color])

    return Scenario(
        kind="chimney", name="chimney_climb",
        spawn_xy=np.array([0.0, 0.0]), goal=np.array([0.0, 0.0]),
        path_pts=np.array([[0.0, 0.0], [0.0, 0.0]]), markers=np.array([[0.0, 0.0]]),
        path_length=shaft_height,
        walls=walls,
        steps=steps,
        yardlines=yardlines,
    )


def test_ninja_climb():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Start on the ground between the two boxes at z = 0.25m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.25
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("=== CLIMBING UP MULTIPLE METERS (Ninja Stemming Leaps) ===")
    hop_period = 35  # 35 steps (0.35s) per leap

    for step in range(1, 281):
        cycle_step = step % hop_period
        t_phase = cycle_step / hop_period

        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T
        u_lat = dirs_world[:, 1]
        u_z = dirs_world[:, 2]

        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        y_now = float(env.data.qpos[1])
        side_centering = np.where(u_lat * y_now > 0, 1.15, 0.85)

        targets = np.zeros(len(env.dirs_body), dtype=np.float32)

        # 4-Phase Stemming Leap:
        # 1. Crouch / Pre-thrust (0.0 -> 0.15): clamp slightly, retract bottom
        # 2. Explosive Thrust (0.15 -> 0.40): blast bottom flank rods into walls & ground -> launch up!
        # 3. Flight Phase (0.40 -> 0.65): retract rods during upward ballistic coast
        # 4. Catch & Lock (0.65 -> 1.00): clamp lateral rods into Box 1 & Box 2 to lock at higher apex!
        if t_phase < 0.15:
            # Quick crouch
            targets = flank * 0.04
        elif t_phase < 0.40:
            # Blast downward-flank rods
            blast = (u_z < 0.15) & (flank > 0.3)
            targets[blast] = env.max_extend
            targets[~blast] = 0.02
        elif t_phase < 0.65:
            # Retract during coast
            targets = flank * 0.02
        else:
            # Catch and clamp firmly against both walls
            targets = flank * 0.092

        targets *= side_centering
        targets = np.clip(targets, 0.0, env.max_extend)
        targets[np.abs(u_lat) < 0.22] = 0.0

        env.step(targets)

        if step % 35 == 0:
            print(f"Leap {step // 35:2d} (step {step:3d}): z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_apex = float(env.data.qpos[2])
    print(f"\nReached Apex Height: z = {z_apex:.3f}m (Climbed +{z_apex - 0.25:.2f}m up!)")
    env.close()

test_ninja_climb()
