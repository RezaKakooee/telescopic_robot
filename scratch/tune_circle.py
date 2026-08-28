import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill


def test_steering(lookahead=0.35, p_gain=1.5, radius=1.8, steps=800):
    cfg = load_config("configs/rl/standing_jump_showcase.yaml")
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    env.data.qpos[0] = radius
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.20
    env.data.qvel[:] = 0

    radii = []
    prev_th = 0.0
    accum_th = 0.0

    for step in range(steps):
        p = env.data.qpos[0:2].copy()
        quat = env.data.qpos[3:7].copy()
        r = float(np.linalg.norm(p))
        radii.append(r)

        th_now = float(np.arctan2(p[1], p[0]))
        d_th_val = th_now - prev_th
        if d_th_val > np.pi: d_th_val -= 2 * np.pi
        elif d_th_val < -np.pi: d_th_val += 2 * np.pi
        accum_th += abs(d_th_val)
        prev_th = th_now

        # Compute target point with understeer compensation
        d_th = lookahead / max(radius, 0.2)
        th_target = th_now + d_th
        # Dynamic radius target pulls inward when drifting out
        r_target_dynamic = max(0.2, radius - p_gain * (r - radius))
        p_target = r_target_dynamic * np.array([np.cos(th_target), np.sin(th_target)])

        heading_vec = p_target - p
        d_cmd = heading_vec / max(np.linalg.norm(heading_vec), 1e-6)

        targets = execute_skill("move", quat, env.dirs_body, env.max_extend,
                                d_hat=d_cmd, speed=1.0)
        env.step(targets)

    env.close()
    r_arr = np.array(radii[100:])
    mean_r = float(np.mean(r_arr))
    std_r = float(np.std(r_arr))
    laps = accum_th / (2 * np.pi)
    print(f"L={lookahead:.2f} P={p_gain:.2f} -> achieved R={mean_r:.3f}±{std_r:.3f}m, err={abs(mean_r-radius)*100:.1f}cm, laps={laps:.2f}")
    return mean_r, std_r, laps

print("Tuning circular controller...")
for L in [0.25, 0.35, 0.45]:
    for P in [1.0, 1.5, 2.0, 2.5]:
        test_steering(lookahead=L, p_gain=P)
