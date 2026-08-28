"""Test robust vertical climbing and descending between two vertical boxes in MuJoCo."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


def make_chimney_scenario(cfg, shaft_width=0.38, shaft_height=4.5, shaft_length=1.4):
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

    yardlines = []
    for h in np.arange(0.25, shaft_height, 0.25):
        color = "0.96 0.78 0.08 1.0" if int(round(h * 100)) % 100 == 0 else "0.95 0.95 0.95 0.8"
        yardlines.append([0.0, half_gap + 0.005, shaft_length / 2.0 * 0.9, 0.015, color])
        yardlines.append([0.0, -half_gap - 0.005, shaft_length / 2.0 * 0.9, 0.015, color])

    return Scenario(
        kind="chimney", name="chimney_climb",
        spawn_xy=np.array([0.0, 0.0]), goal=np.array([0.0, 10.0]),
        path_pts=np.array([[0.0, 0.0], [0.0, 0.0]]), markers=np.array([[0.0, 0.0]]),
        path_length=shaft_height,
        walls=walls,
        steps=steps,
        yardlines=yardlines,
    )


def test():
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = make_chimney_scenario(cfg, shaft_width=0.38, shaft_height=4.5)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    obs, info = env.reset(seed=42)

    # Spawn resting on floor at z = 0.20m
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.20
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    print(f"Spawned at z = {env.data.qpos[2]:.3f} m")

    # Let's test a continuous 3-stage vertical locomoting gait:
    # 1. Stance Rest on floor (steps 1-20)
    # 2. Explosive Vertical Jump & Apex Lock at z = 0.75m (steps 20-100)
    # 3. Controlled Descent down to z = 0.20m (steps 100-180)
    # 4. Second Jump & Apex Lock at z = 0.80m (steps 180-260)
    # 5. Second Descent (steps 260-340)

    for step in range(1, 350):
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        R = quat_to_rotmat(quat)
        dirs = env.dirs_body @ R.T
        u_lat = dirs[:, 1]
        u_z = dirs[:, 2]

        flank = np.clip((np.abs(u_lat) - 0.20) / 0.35, 0.0, 1.0)
        side_centering = np.where(u_lat * pos[1] > 0, 1.15, 0.85)

        t = np.zeros(env.n_bars, dtype=np.float32)

        # Stage 1: Stance Rest (1-20)
        if step <= 20:
            phase = "STAND"
            ground = (u_z < -0.30) & (np.abs(u_lat) < 0.35)
            t[ground] = 0.045
        # Stage 2: Crouch 1 (21-30)
        elif step <= 30:
            phase = "CROUCH 1"
            t[:] = 0.00
        # Stage 3: Launch UP 1 (31-42)
        elif step <= 42:
            phase = "🚀 LAUNCH 1 (UP)"
            ground = (u_z < 0.0) & (np.abs(u_lat) < 0.45)
            t[ground] = env.max_extend
        # Stage 4: Flight 1 (43-55)
        elif step <= 55:
            phase = "✈️ ASCENT 1"
            t[:] = 0.010
        # Stage 5: Mid-Air Apex Lock 1 (56-110)
        elif step <= 110:
            phase = "🔒 MID-AIR LOCK 1"
            t = flank * side_centering * 0.095
            t[np.abs(u_lat) < 0.22] = 0.0
        # Stage 6: Glide Descent 1 (111-180)
        elif step <= 180:
            phase = "🪂 DESCENT 1 (DOWN)"
            t = flank * side_centering * 0.030
            if pos[2] < 0.24:
                t[u_z < -0.25] = 0.055
        # Stage 7: Crouch 2 (181-190)
        elif step <= 190:
            phase = "CROUCH 2"
            t[:] = 0.00
        # Stage 8: Launch UP 2 (191-202)
        elif step <= 202:
            phase = "🚀 LAUNCH 2 (UP)"
            ground = (u_z < 0.0) & (np.abs(u_lat) < 0.45)
            t[ground] = env.max_extend
        # Stage 9: Flight 2 (203-215)
        elif step <= 215:
            phase = "✈️ ASCENT 2"
            t[:] = 0.010
        # Stage 10: Mid-Air Apex Lock 2 (216-270)
        elif step <= 270:
            phase = "🔒 MID-AIR LOCK 2"
            t = flank * side_centering * 0.095
            t[np.abs(u_lat) < 0.22] = 0.0
        # Stage 11: Final Glide Descent (271-349)
        else:
            phase = "🪂 FINAL DESCENT (DOWN)"
            t = flank * side_centering * 0.030
            if pos[2] < 0.24:
                t[u_z < -0.25] = 0.055

        env.step(t)

        if step % 25 == 0 or "LAUNCH" in phase:
            print(f"Step {step:3d} [{phase:24s}]: z = {pos[2]:.3f} m | vz = {vel[2]:+.2f} m/s | y = {pos[1]*100:+.1f} cm")

    env.close()

test()
