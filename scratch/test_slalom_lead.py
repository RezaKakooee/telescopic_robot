import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
from scripts.skills.run_training_cones import _cone_contact

def test_slalom_with_phase_lead(amp_target, phase_lead_m, speed):
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

    for step in range(max_steps):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()

        # Check cone clearances
        for ci, c in enumerate(cones):
            dist = float(np.linalg.norm(pos[:2] - c[:2]))
            cone_min_dist[ci] = min(cone_min_dist[ci], dist)

        total_cone_contacts += _cone_contact(env)

        # Slalom steering with phase lead
        # Lead the phase by phase_lead_m
        x_eval = pos[0] + phase_lead_m

        if x_eval < first_cone_x - 0.4:
            target_y = amp_target * 0.4
        elif x_eval > last_cone_x + 0.6:
            target_y = 0.0
        else:
            phase = np.pi * (x_eval - (first_cone_x - spacing / 2.0)) / spacing
            target_y = amp_target * np.sin(phase)

        # Pure pursuit heading vector
        lookahead = 0.50
        heading_vec = np.array([lookahead, (target_y - pos[1]) * 1.8], dtype=np.float64)
        d_hat = heading_vec / max(float(np.linalg.norm(heading_vec)), 1e-6)

        targets = move(quat, env.dirs_body, env.max_extend, d_hat=d_hat, speed=speed)
        obs, reward, terminated, truncated, info = env.step(targets)

        if pos[0] >= goal[0] + 0.3:
            break

    env.close()
    min_clr = min(cone_min_dist)
    print(f"amp={amp_target:.2f}, lead={phase_lead_m:.2f}, speed={speed:.1f} -> contacts={total_cone_contacts}, min_clr={min_clr:.3f}m, y_at_cones={[round(d, 2) for d in cone_min_dist]}")
    return total_cone_contacts, min_clr

print("Testing phase lead & amplitude boost:")
for amp in [1.00, 1.15, 1.30, 1.45]:
    for lead in [0.40, 0.55, 0.70, 0.85]:
        for sp in [1.2, 1.4]:
            test_slalom_with_phase_lead(amp, lead, sp)
