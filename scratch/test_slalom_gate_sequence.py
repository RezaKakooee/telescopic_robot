import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
from scripts.skills.run_training_cones import _cone_contact

def test_gate_sequence(lat_amp, speed, lead_dist):
    cfg = load_config("configs/rl/training_cones.yaml")
    scenario = generate_scenario("training_cones", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=3000)
    obs, info = env.reset(seed=42)

    cones = np.asarray(scenario.cones, dtype=float)
    n_cones = len(cones)
    goal = np.asarray(scenario.goal, dtype=float)

    # 10 Gates, alternating +lat_amp, -lat_amp
    gates = []
    for i, c in enumerate(cones):
        sign = +1.0 if (i % 2 == 0) else -1.0
        gates.append((float(c[0]), sign * lat_amp))
    gates.append((float(goal[0]), 0.0))
    gates = np.array(gates, dtype=np.float64)

    cone_min_dist = [float("inf")] * n_cones
    total_cone_contacts = 0
    max_steps = 2500

    y_at_cone_x = [0.0] * n_cones
    current_gate = 0

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

        # Switch to next gate when past current cone x
        if current_gate < len(gates) - 1:
            gx, gy = gates[current_gate]
            if pos[0] > gx - 0.15:
                current_gate += 1

        # Lookahead target point along vector to next gate
        gx, gy = gates[current_gate]
        target_vec = np.array([gx, gy]) - pos[:2]
        d_hat = target_vec / max(float(np.linalg.norm(target_vec)), 1e-6)

        targets = move(quat, env.dirs_body, env.max_extend, d_hat=d_hat, speed=speed)
        obs, reward, terminated, truncated, info = env.step(targets)

        if pos[0] >= goal[0] + 0.3:
            break

    env.close()
    min_clr = min(cone_min_dist)
    print(f"lat={lat_amp:.2f}, sp={speed:.1f} -> contacts={total_cone_contacts}, min_clr={min_clr:.3f}m, y_cones={[round(y, 2) for y in y_at_cone_x]}")
    return total_cone_contacts, min_clr

print("Testing gate sequence navigation:")
for lat in [0.70, 0.85, 1.00, 1.15, 1.30]:
    for sp in [1.0, 1.2, 1.4]:
        test_gate_sequence(lat, sp, 0.5)
