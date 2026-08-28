"""Test vertical chimney stemming hop between two vertical walls."""
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


def test_stem_hops():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.36, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Spawn ball between walls at z = 0.30m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.30
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print("Initial height: z = 0.30m")

    hop_period = 40  # 40 steps = 0.40s per hop cycle

    for step in range(1, 401):
        cycle_t = (step % hop_period) / hop_period

        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T
        u_lat = dirs_world[:, 1]
        u_z = dirs_world[:, 2]

        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        y_now = float(env.data.qpos[1])
        side_centering = np.where(u_lat * y_now > 0, 1.15, 0.85)

        targets = np.zeros(len(env.dirs_body), dtype=np.float32)

        # Chimney Stemming Hop Cycle:
        # 1. Crouch / Preload (cycle_t in [0.0, 0.25]):
        #    Lateral rods retract slightly to 0.04m, lower rods prepare.
        # 2. Explosive Thrust (cycle_t in [0.25, 0.50]):
        #    Downward-lateral rods (u_z < 0.1, |u_lat| > 0.35) punch out at max extension (0.16m)
        #    to launch the core UPWARD!
        # 3. Flight / Retract (cycle_t in [0.50, 0.70]):
        #    Lower rods retract to 0.02m while ball coasts upward.
        # 4. Catch & Clamp (cycle_t in [0.70, 1.00]):
        #    All lateral rods clamp outward (0.10m) to brake and lock at the higher apex!
        if cycle_t < 0.25:
            # Crouch
            targets = flank * 0.04
        elif cycle_t < 0.50:
            # Thrust: explode downward-lateral rods
            thrust_mask = (u_z < 0.15) & (flank > 0.3)
            targets[thrust_mask] = env.max_extend
            targets[~thrust_mask] = 0.02
        elif cycle_t < 0.70:
            # Flight: tuck rods
            targets = flank * 0.02
        else:
            # Catch: clamp walls hard to hold new height
            targets = flank * 0.095

        targets *= side_centering
        targets = np.clip(targets, 0.0, env.max_extend)
        targets[np.abs(u_lat) < 0.22] = 0.0

        env.step(targets)

        if step % 40 == 0:
            print(f"Hop {step // 40:2d} (step {step:3d}): z = {env.data.qpos[2]:.3f}m | vz = {env.data.qvel[2]:+.2f}m/s | y = {env.data.qpos[1]*100:+.1f}cm")

    z_end = float(env.data.qpos[2])
    print(f"\nFinal height after 10 hops: z = {z_end:.3f}m (Climbed Δz = {z_end - 0.30:+.3f}m!)")
    env.close()

test_stem_hops()
