import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
from scripts.skills.run_training_cones import _cone_contact

def test_lateral_authority_slalom(amp, sp, lat_gain, lead_m):
    cfg = load_config("configs/rl/training_cones.yaml")
    cfg.scenario.spacing = 2.40  # 2.4m spacing
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

    # Gates at each cone
    gates = []
    for i, c in enumerate(cones):
        sign = +1.0 if (i % 2 == 0) else -1.0
        gates.append((float(c[0]), sign * amp))
    gates.append((float(goal[0]), 0.0))

    current_gate_idx = 0

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

        # Advance gate early (lead_m before the cone)
        if current_gate_idx < len(gates) - 1:
            gx, gy = gates[current_gate_idx]
            if pos[0] > gx - lead_m:
                current_gate_idx += 1

        target_gx, target_gy = gates[current_gate_idx]
        
        # High lateral authority heading vector
        dx = target_gx - pos[0]
        dy = target_gy - pos[1]
        
        # Scale lateral error by lat_gain to command sharp turns
        cmd_x = max(dx, 0.3)
        cmd_y = dy * lat_gain
        
        heading_vec = np.array([cmd_x, cmd_y], dtype=np.float64)
        d_hat = heading_vec / np.linalg.norm(heading_vec)

        targets = move(quat, env.dirs_body, env.max_extend, d_hat=d_hat, speed=sp)
        obs, reward, terminated, truncated, info = env.step(targets)

        if pos[0] >= goal[0] + 0.3:
            break

    env.close()
    min_clr = min(cone_min_dist)
    print(f"lat_gain={lat_gain:.1f}, sp={sp:.1f}, lead={lead_m:.2f} -> contacts={total_cone_contacts}, min_clr={min_clr:.3f}m, y_cones={[round(y, 2) for y in y_at_cone_x]}")
    return total_cone_contacts, min_clr

print("Testing high lateral authority steering:")
for gain in [2.5, 3.5, 4.5, 6.0]:
    for sp in [1.0, 1.2, 1.4]:
        for lead in [0.40, 0.60, 0.80]:
            test_lateral_authority_slalom(0.75, sp, gain, lead)
