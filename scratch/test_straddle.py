"""Scratch test for straddle_gap skill over a 0.22m wide, 0.25m deep central hole."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill


def test_straddle():
    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = generate_scenario("gap_bridge", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    # Place ball spanning the two boxes (Box height = 0.25m, roll radius = 0.19m -> z = 0.44m)
    box_height = 0.25
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = box_height + 0.21
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    # Let the ball settle onto Box 1 and Box 2
    for _ in range(40):
        t = execute_skill("straddle_gap", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                          speed=0.0)
        env.step(t)

    z_start = float(env.data.qpos[2])
    print(f"Settled on dual boxes: x={float(env.data.qpos[0]):.2f} y={float(env.data.qpos[1]):+.2f} z={z_start:.3f}m (deck={box_height:.2f}m)")

    # Run straddle_gap for 500 steps
    d_fwd = np.array([1.0, 0.0])
    for step in range(1, 501):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        y_err = float(pos[1])

        t = execute_skill("straddle_gap", quat, env.dirs_body, env.max_extend,
                          d_hat=d_fwd, speed=1.2, lateral_offset=y_err)
        env.step(t)


        if step % 100 == 0:
            print(f"Step {step:3d}: x={pos[0]:.2f}m  y={pos[1]:+.3f}m  z={pos[2]:.3f}m  vx={float(env.data.qvel[0]):.2f}m/s")

    end_pos = env.data.qpos[0:3].copy()
    env.close()

    print("\n" + "=" * 60)
    print(f"Start:  z={z_start:.3f}m")
    print(f"End:    x={end_pos[0]:.2f}m  y={end_pos[1]:+.3f}m  z={end_pos[2]:.3f}m")
    print(f"Result: {'✅ SUCCESS — traversed gap on top of boxes' if end_pos[0] > 3.0 and end_pos[2] > 0.38 else '❌ FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    test_straddle()
