import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
from scripts.skills.run_training_cones import _cone_contact

def test_agile_slalom(amp, base_speed, lead_phase_m):
    cfg = load_config("configs/rl/training_cones.yaml")
    scenario = generate_scenario("training_cones", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=3000)
    obs, info = env.reset(seed=42)

    cones = np.asarray(scenario.cones, dtype=float)
    n_cones = len(cones)
    goal = np.asarray(scenario.goal, dtype=float)

    first_cone_x = float(cones[0, 0])
    last_cone_x = float(cones[-1, 0])
    spacing = float(cones[1, 0] - cones[0, 0])

    cone_min_dist = [float("inf")] * n_cones
    total_cone_contacts = 0
    max_steps = 2500

    y_at_cone_x = [0.0] * n_cones

    for step in range(max_steps):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()

        # Check cone clearances
        for ci, c in enumerate(cones):
            dist = float(np.linalg.norm(pos[:2] - c[:2]))
            cone_min_dist[ci] = min(cone_min_dist[ci], dist)
            if abs(pos[0] - c[0]) < 0.08:
                y_at_cone_x[ci] = float(pos[1])

        total_cone_contacts += _cone_contact(env)

        # Target point with phase anticipation
        x_eval = pos[0] + lead_phase_m
        if pos[0] < first_cone_x - 0.4:
            target_y = amp * 0.5
        elif pos[0] > last_cone_x + 0.3:
            target_y = 0.0
        else:
            phase = np.pi * (x_eval - (first_cone_x - spacing / 2.0)) / spacing
            target_y = amp * np.sin(phase)

        # Lateral error
        dy = target_y - pos[1]
        
        # Heading vector with strong lateral authority
        cmd_y = float(np.clip(dy * 2.2, -1.8, 1.8))
        cmd_x = 0.85
        heading_vec = np.array([cmd_x, cmd_y], dtype=np.float64)
        d_hat = heading_vec / np.linalg.norm(heading_vec)

        # Speed scaling: cruise faster when aligned, modulate in sharp turns
        h_angle = abs(float(np.arctan2(cmd_y, cmd_x)))
        cur_speed = base_speed * max(float(np.cos(h_angle)), 0.60)

        targets = move(quat, env.dirs_body, env.max_extend, d_hat=d_hat, speed=cur_speed)
        obs, reward, terminated, truncated, info = env.step(targets)

        if pos[0] >= goal[0] + 0.3:
            break

    env.close()
    min_clr = min(cone_min_dist)
    print(f"amp={amp:.2f}, speed={base_speed:.1f}, lead={lead_phase_m:.2f} -> contacts={total_cone_contacts}, min_clr={min_clr:.3f}m, y_cones={[round(y, 2) for y in y_at_cone_x]}")
    return total_cone_contacts, min_clr

print("Testing agile slalom steering:")
for amp in [0.70, 0.80, 0.90]:
    for lead in [0.40, 0.55, 0.70, 0.85]:
        for sp in [1.0, 1.2, 1.4]:
            test_agile_slalom(amp, sp, lead)
