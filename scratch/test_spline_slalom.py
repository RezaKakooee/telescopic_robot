import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from scipy.interpolate import CubicSpline

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
from scripts.skills.run_training_cones import _cone_contact

def test_spline_slalom(amp, sp, lookahead):
    cfg = load_config("configs/rl/training_cones.yaml")
    scenario = generate_scenario("training_cones", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=3000)
    obs, info = env.reset(seed=42)

    cones = np.asarray(scenario.cones, dtype=float)
    n_cones = len(cones)
    goal = np.asarray(scenario.goal, dtype=float)

    # Key spline knots:
    # (0, 0) -> before cone 0 -> (x_0, +amp) -> (mid_01, 0) -> (x_1, -amp) -> (mid_12, 0) -> ...
    knot_x = [0.0, 1.0]
    knot_y = [0.0, amp * 0.4]
    spacing = float(cones[1, 0] - cones[0, 0])

    for i, c in enumerate(cones):
        sign = +1.0 if (i % 2 == 0) else -1.0
        cx = float(c[0])
        # Cone apex
        knot_x.append(cx)
        knot_y.append(sign * amp)
        # Mid crossing
        if i < n_cones - 1:
            knot_x.append(cx + spacing * 0.5)
            knot_y.append(0.0)

    knot_x.extend([float(goal[0]) - 0.5, float(goal[0])])
    knot_y.extend([0.0, 0.0])

    cs = CubicSpline(knot_x, knot_y, bc_type='natural')

    cone_min_dist = [float("inf")] * n_cones
    total_cone_contacts = 0
    max_steps = 2500

    # Also compute lateral clearance at cone x coordinates
    y_at_cone_x = [0.0] * n_cones

    for step in range(max_steps):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()

        # Check cone clearances
        for ci, c in enumerate(cones):
            dist = float(np.linalg.norm(pos[:2] - c[:2]))
            cone_min_dist[ci] = min(cone_min_dist[ci], dist)
            if abs(pos[0] - c[0]) < 0.05:
                y_at_cone_x[ci] = abs(float(pos[1]))

        total_cone_contacts += _cone_contact(env)

        # Pure pursuit with lookahead along the cubic spline
        x_target = pos[0] + lookahead
        x_target = np.clip(x_target, 0.0, knot_x[-1])
        y_target = float(cs(x_target))
        # Tangent of spline at target
        dy_dx = float(cs(x_target, 1))
        
        # Heading error + lateral cross-track error
        heading_vec = np.array([x_target - pos[0], y_target - pos[1]], dtype=np.float64)
        # Feedforward tangent + feedback vector
        tangent_vec = np.array([1.0, dy_dx])
        tangent_vec /= np.linalg.norm(tangent_vec)
        
        cmd_vec = 0.6 * tangent_vec + 0.4 * (heading_vec / np.linalg.norm(heading_vec))
        d_hat = cmd_vec / np.linalg.norm(cmd_vec)

        targets = move(quat, env.dirs_body, env.max_extend, d_hat=d_hat, speed=sp)
        obs, reward, terminated, truncated, info = env.step(targets)

        if pos[0] >= goal[0] + 0.3:
            break

    env.close()
    min_clr = min(cone_min_dist)
    print(f"amp={amp:.2f}, sp={sp:.1f}, look={lookahead:.2f} -> contacts={total_cone_contacts}, min_clr={min_clr:.3f}m, y_at_x={[round(y, 2) for y in y_at_cone_x]}")
    return total_cone_contacts, min_clr

print("Testing cubic spline slalom tracking:")
for amp in [0.70, 0.85, 1.00]:
    for sp in [1.0, 1.2, 1.5]:
        for look in [0.35, 0.50, 0.65]:
            test_spline_slalom(amp, sp, look)
