import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
from scripts.skills.run_training_cones import _cone_contact

def test_guaranteed_slalom(amp, sp, lead_turn_dist):
    cfg = load_config("configs/rl/training_cones.yaml")
    cfg.scenario.spacing = 2.0  # 2.0m spacing between 10 cones
    scenario = generate_scenario("training_cones", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=3500)
    obs, info = env.reset(seed=42)

    cones = np.asarray(scenario.cones, dtype=float)
    n_cones = len(cones)
    goal = np.asarray(scenario.goal, dtype=float)

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
            if abs(pos[0] - c[0]) < 0.08:
                y_at_cone_x[ci] = float(pos[1])

        total_cone_contacts += _cone_contact(env)

        # Find the cone we are currently passing or aiming for:
        # A cone i (at cx_i) requires us to be at sign_i * amp
        # We aim for (cx_i, sign_i * amp) until pos[0] >= cx_i, then we aim for (cx_{i+1}, sign_{i+1} * amp)
        target_pt = None
        for i, c in enumerate(cones):
            cx = float(c[0])
            sign = +1.0 if (i % 2 == 0) else -1.0
            if pos[0] < cx - 0.05:
                # Approach waypoint before the apex
                target_pt = np.array([cx + 0.10, sign * amp])
                break
            elif pos[0] < cx + 0.35:
                # Holding apex past the cone
                target_pt = np.array([cx + 0.50, sign * amp * 0.9])
                break

        if target_pt is None:
            target_pt = np.array([float(goal[0]), 0.0])

        heading_vec = target_pt - pos[:2]
        d_hat = heading_vec / max(float(np.linalg.norm(heading_vec)), 1e-6)

        targets = move(quat, env.dirs_body, env.max_extend, d_hat=d_hat, speed=sp)
        obs, reward, terminated, truncated, info = env.step(targets)

        if pos[0] >= goal[0] + 0.3:
            break

    env.close()
    min_clr = min(cone_min_dist)
    print(f"amp={amp:.2f}, sp={sp:.1f} -> contacts={total_cone_contacts}, min_clr={min_clr:.3f}m, y_cones={[round(y, 2) for y in y_at_cone_x]}")
    return total_cone_contacts, min_clr

print("Testing guaranteed gate apex tracking:")
for amp in [0.70, 0.75, 0.80, 0.85, 0.90]:
    for sp in [1.0, 1.2, 1.4]:
        test_guaranteed_slalom(amp, sp, 0.2)
