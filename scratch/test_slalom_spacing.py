import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
from scripts.skills.run_training_cones import _cone_contact

def test_slalom_spacing(spacing, amp, speed, lookahead):
    cfg = load_config("configs/rl/training_cones.yaml")
    # Override spacing in scenario
    cfg.scenario.spacing = float(spacing)
    scenario = generate_scenario("training_cones", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=3500)
    obs, info = env.reset(seed=42)

    cones = np.asarray(scenario.cones, dtype=float)
    n_cones = len(cones)
    goal = np.asarray(scenario.goal, dtype=float)

    first_cone_x = float(cones[0, 0])
    last_cone_x = float(cones[-1, 0])

    cone_min_dist = [float("inf")] * n_cones
    total_cone_contacts = 0
    max_steps = 3000

    y_at_cone_x = [0.0] * n_cones

    for step in range(max_steps):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()

        # Check cone clearances
        for ci, c in enumerate(cones):
            dist = float(np.linalg.norm(pos[:2] - c[:2]))
            cone_min_dist[ci] = min(cone_min_dist[ci], dist)
            if abs(pos[0] - c[0]) < 0.10:
                y_at_cone_x[ci] = float(pos[1])

        total_cone_contacts += _cone_contact(env)

        # Sinusoidal reference with lookahead
        x_target = pos[0] + lookahead
        if x_target < first_cone_x - 0.4:
            y_target = amp * 0.5
        elif x_target > last_cone_x + 0.6:
            y_target = 0.0
        else:
            phase = np.pi * (x_target - (first_cone_x - spacing / 2.0)) / spacing
            y_target = amp * np.sin(phase)

        heading_vec = np.array([lookahead, (y_target - pos[1]) * 1.5], dtype=np.float64)
        d_hat = heading_vec / np.linalg.norm(heading_vec)

        targets = move(quat, env.dirs_body, env.max_extend, d_hat=d_hat, speed=speed)
        obs, reward, terminated, truncated, info = env.step(targets)

        if pos[0] >= goal[0] + 0.3:
            break

    env.close()
    min_clr = min(cone_min_dist)
    print(f"spacing={spacing:.2f}m, amp={amp:.2f}, sp={speed:.1f}, look={lookahead:.2f} -> contacts={total_cone_contacts}, min_clr={min_clr:.3f}m, y_cones={[round(y, 2) for y in y_at_cone_x]}")
    return total_cone_contacts, min_clr

print("Testing physical cone spacings (2.2m to 2.8m):")
for spc in [2.20, 2.50, 2.80]:
    for amp in [0.70, 0.80, 0.90]:
        for look in [0.60, 0.75, 0.90]:
            test_slalom_spacing(spc, amp, 1.4, look)
