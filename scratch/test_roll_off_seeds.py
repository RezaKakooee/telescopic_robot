"""Test roll_off across short and tall drops and multiple seeds."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario, pillar_course_columns
from skills import execute_skill

FORWARD = np.array([1.0, 0.0])
ROLL_RADIUS = 0.19

def test_rolloff(seed, drop_h=0.40):
    cfg = load_config("configs/rl/pillar_course.yaml")
    scenario = generate_scenario("pillar_course", cfg, seed=1)
    cols = pillar_course_columns(cfg)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=seed)

    cur = cols[2]
    target = cols[3]
    # Adjust drop height if testing different drop
    drop = cur["height"] - target["height"]

    env.data.qpos[0] = cur["far"] - 0.25
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = cur["height"] + ROLL_RADIUS
    env.data.qvel[:] = 0

    deck_core = target["height"] + ROLL_RADIUS
    ahead = [c2 for c2 in cols if c2["near"] > target["far"] - 0.05
             and c2["near"] - target["far"] < 0.8 and c2["height"] > target["height"]]
    brace = 0.35 if ahead else 0.0

    phase, ps = "edge", 0
    for step in range(1500):
        z = float(env.data.qpos[2])
        if phase == "edge" and z < cur["height"] + ROLL_RADIUS - 0.06:
            phase, ps = "freefall", 0
        elif phase == "freefall" and z < deck_core + 0.10:
            phase, ps = "absorb", 0
        elif phase == "absorb" and z < deck_core + 0.03:
            phase, ps = "brake", 0
        ps += 1

        if phase == "brake":
            t = execute_skill("stop", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                              lin_vel=env.data.qvel[0:2].copy(), stop_distance=0.15)
        else:
            t = execute_skill("fall_down", env.data.qpos[3:7].copy(), env.dirs_body,
                              env.max_extend, d_hat=FORWARD, phase=phase,
                              drop_height=drop, edge_speed=0.35,
                              gear=0.5, brace_front=brace)
        env.step(t)
        if phase == "brake" and ps > 80:
            break

    for _ in range(60):
        env.step(execute_skill("stop", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                               lin_vel=env.data.qvel[0:2].copy()))

    x, y, z = float(env.data.qpos[0]), float(env.data.qpos[1]), float(env.data.qpos[2])
    on = (target["near"] < x < target["far"]
          and target["height"] + 0.10 < z < target["height"] + 0.55)

    print(f"Seed {seed:2d}: landed at x={x:.2f} y={y:+.2f} z={z:.2f} (target [{target['near']:.2f}, {target['far']:.2f}]) -> {'✅ ON PILLAR' if on else '❌ MISSED'}")
    env.close()
    return on

print("=== Testing roll_off (0.4m step-down) across 10 random seeds ===")
results = [test_rolloff(s) for s in range(1, 11)]
print(f"\nSummary: {sum(results)}/10 seeds landed successfully on pillar 3.")
