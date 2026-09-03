import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
from scripts.skills.run_training_cones import _cone_contact

def test_slalom_cone_tuck(amp, sp, lookahead):
    cfg = load_config("configs/rl/training_cones.yaml")
    cfg.scenario.spacing = 2.40  # 2.4m spacing
    scenario = generate_scenario("training_cones", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=3000)
    obs, info = env.reset(seed=42)

    cones = np.asarray(scenario.cones, dtype=float)
    n_cones = len(cones)
    goal = np.asarray(scenario.goal, dtype=float)

    first_cone_x = float(cones[0, 0])
    last_cone_x = float(cones[-1, 0])
    spacing = float(cones[1, 0] - cones[0, 0])
    foot_base = 0.173
    cone_r = 0.12

    cone_min_dist = [float("inf")] * n_cones
    total_cone_contacts = 0
    max_steps = 2500

    y_at_cone_x = [0.0] * n_cones

    for step in range(max_steps):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()
        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        # Check cone clearances
        for ci, c in enumerate(cones):
            dist = float(np.linalg.norm(pos[:2] - c[:2]))
            cone_min_dist[ci] = min(cone_min_dist[ci], dist)
            if abs(pos[0] - c[0]) < 0.10:
                y_at_cone_x[ci] = float(pos[1])

        total_cone_contacts += _cone_contact(env)

        # Sinusoidal reference with lookahead
        x_target = pos[0] + lookahead
        if pos[0] < first_cone_x - 0.4:
            y_target = amp * 0.4
        elif pos[0] > last_cone_x + 0.3:
            y_target = 0.0
        else:
            phase = np.pi * (x_target - (first_cone_x - spacing / 2.0)) / spacing
            y_target = amp * np.sin(phase)

        heading_vec = np.array([lookahead, (y_target - pos[1]) * 1.6], dtype=np.float64)
        d_hat = heading_vec / np.linalg.norm(heading_vec)

        # Base gait targets
        targets = move(quat, env.dirs_body, env.max_extend, d_hat=d_hat, speed=sp)

        # Dynamic Obstacle Cone Tucking:
        # For the nearest 2 cones, tuck rods that point towards the cone to prevent touching
        for c in cones:
            c_xy = c[:2]
            d_vec = c_xy - pos[:2]
            d_cone = float(np.linalg.norm(d_vec))
            if d_cone < 1.0:  # Only for cones within 1.0 m
                d_hat_cone = np.array([d_vec[0], d_vec[1], 0.0]) / max(d_cone, 1e-6)
                u_cone = dirs_world @ d_hat_cone
                # Rods pointing towards the cone
                facing_cone = u_cone > 0.05
                # Maximum allowed extension so foot stays 4cm away from cone
                max_reach = max(d_cone - cone_r - foot_base - 0.04, 0.025)
                # Cap facing rods
                targets[facing_cone] = np.minimum(targets[facing_cone], max_reach)

        obs, reward, terminated, truncated, info = env.step(targets)

        if pos[0] >= goal[0] + 0.3:
            break

    env.close()
    min_clr = min(cone_min_dist)
    print(f"spacing=2.40m, amp={amp:.2f}, sp={sp:.1f}, look={lookahead:.2f} -> contacts={total_cone_contacts}, min_clr={min_clr:.3f}m, y_cones={[round(y, 2) for y in y_at_cone_x]}")
    return total_cone_contacts, min_clr

print("Testing dynamic cone tucking:")
for amp in [0.75, 0.85, 0.95]:
    for sp in [1.2, 1.4]:
        for look in [0.65, 0.80]:
            test_slalom_cone_tuck(amp, sp, look)
