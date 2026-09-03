import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
from scripts.skills.run_training_cones import _cone_contact

def test_balanced_slalom(amp, sp, lead_x, kd):
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

        # Lookahead along the sine wave with velocity compensation
        x_eval = pos[0] + lead_x

        if x_eval < first_cone_x - 0.4:
            y_target = amp * 0.5
        elif x_eval > last_cone_x + 0.6:
            y_target = 0.0
        else:
            phase = np.pi * (x_eval - (first_cone_x - spacing / 2.0)) / spacing
            y_target = amp * np.sin(phase)

        # PD steering law on lateral tracking
        y_err = y_target - pos[1]
        vy = vel[1]
        
        # Desired lateral heading component
        cmd_y = 1.6 * y_err - kd * vy
        cmd_x = 1.0
        
        heading_vec = np.array([cmd_x, cmd_y], dtype=np.float64)
        d_hat = heading_vec / np.linalg.norm(heading_vec)

        # Tuck inner rods facing nearest cone if close
        targets = move(quat, env.dirs_body, env.max_extend, d_hat=d_hat, speed=sp)
        
        obs, reward, terminated, truncated, info = env.step(targets)

        if pos[0] >= goal[0] + 0.3:
            break

    env.close()
    min_clr = min(cone_min_dist)
    print(f"amp={amp:.2f}, sp={sp:.1f}, lead={lead_x:.2f}, kd={kd:.2f} -> contacts={total_cone_contacts}, min_clr={min_clr:.3f}m, y_cones={[round(y, 2) for y in y_at_cone_x]}")
    return total_cone_contacts, min_clr

print("Testing balanced slalom:")
for amp in [0.75, 0.85, 0.95]:
    for lead in [0.45, 0.60, 0.75]:
        for kd in [0.20, 0.40, 0.60]:
            test_balanced_slalom(amp, 1.2, lead, kd)
