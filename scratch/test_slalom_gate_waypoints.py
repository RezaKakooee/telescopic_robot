import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
from scripts.skills.run_training_cones import _cone_contact

def test_gate_waypoints(lat_gate, speed, lookahead, k_p):
    cfg = load_config("configs/rl/training_cones.yaml")
    scenario = generate_scenario("training_cones", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=3000)
    obs, info = env.reset(seed=42)

    cones = np.asarray(scenario.cones, dtype=float)
    n_cones = len(cones)
    goal = np.asarray(scenario.goal, dtype=float)

    # Build sequence of gate waypoints
    # For each cone i:
    # 1. Entry point ahead of cone: (x_i - 0.45, sign * lat_gate * 0.7)
    # 2. Apex point abreast cone:   (x_i,        sign * lat_gate)
    # 3. Exit point past cone:     (x_i + 0.45, sign * lat_gate * 0.7)
    # 4. Center crossing:          (x_i + 0.90, 0.0)
    waypoints = [(0.0, 0.0), (1.2, lat_gate * 0.5)]
    spacing = float(cones[1, 0] - cones[0, 0])
    for i, c in enumerate(cones):
        sign = +1.0 if (i % 2 == 0) else -1.0
        cx = c[0]
        waypoints.append((cx - 0.35, sign * lat_gate * 0.75))
        waypoints.append((cx,        sign * lat_gate))
        waypoints.append((cx + 0.35, sign * lat_gate * 0.75))
        waypoints.append((cx + spacing * 0.5, 0.0))
    waypoints.append((float(goal[0]), 0.0))
    wp = np.array(waypoints, dtype=np.float64)

    cone_min_dist = [float("inf")] * n_cones
    total_cone_contacts = 0
    max_steps = 2500

    current_wp_idx = 0

    for step in range(max_steps):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()

        # Check cone clearances
        for ci, c in enumerate(cones):
            dist = float(np.linalg.norm(pos[:2] - c[:2]))
            cone_min_dist[ci] = min(cone_min_dist[ci], dist)

        total_cone_contacts += _cone_contact(env)

        # Advance current waypoint if passed
        while current_wp_idx < len(wp) - 1:
            target = wp[current_wp_idx]
            dist_to_wp = np.linalg.norm(pos[:2] - target)
            if pos[0] > target[0] or dist_to_wp < 0.35:
                current_wp_idx += 1
            else:
                break

        # Lookahead along waypoint polyline
        target = wp[current_wp_idx]
        heading_vec = target - pos[:2]
        # Normalize and steer
        d_cmd = heading_vec
        d_hat = d_cmd / max(float(np.linalg.norm(d_cmd)), 1e-6)

        targets = move(quat, env.dirs_body, env.max_extend, d_hat=d_hat, speed=speed)
        obs, reward, terminated, truncated, info = env.step(targets)

        if pos[0] >= goal[0] + 0.3:
            break

    env.close()
    min_clr = min(cone_min_dist)
    print(f"lat={lat_gate:.2f}, sp={speed:.1f} -> contacts={total_cone_contacts}, min_clr={min_clr:.3f}m, y_cones={[round(d, 2) for d in cone_min_dist]}")
    return total_cone_contacts, min_clr

print("Testing gate waypoints:")
for lat in [0.75, 0.85, 0.95, 1.05, 1.15]:
    for sp in [1.0, 1.2, 1.4]:
        test_gate_waypoints(lat, sp, 0.6, 1.5)
