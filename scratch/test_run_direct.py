"""Test test_straddle_gap_traverse from test_skills.py directly."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill

def run_test(steps=500, box_height=0.25):
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = generate_scenario("gap_bridge", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = box_height + 0.19
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    # Let the ball settle onto Box 1 and Box 2 at the start line
    for _ in range(25):
        env.step(execute_skill("straddle_gap", env.data.qpos[3:7].copy(), env.dirs_body,
                               env.max_extend, speed=0.0))

    d_fwd = np.array([1.0, 0.0])
    for step in range(1, steps + 1):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        targets = execute_skill("straddle_gap", quat, env.dirs_body, env.max_extend,
                                d_hat=d_fwd, speed=1.3, lateral_offset=float(pos[1]))
        env.step(targets)

        if step % 50 == 0:
            print(f"Step {step:3d}: x={pos[0]:.2f}m y={pos[1]:+.3f}m z={pos[2]:.3f}m vx={env.data.qvel[0]:.2f}m/s")

    end_x, end_y, end_z = float(env.data.qpos[0]), float(env.data.qpos[1]), float(env.data.qpos[2])
    print(f"\nFinal: x={end_x:.2f}m y={end_y:+.3f}m z={end_z:.3f}m")
    env.close()

run_test()
