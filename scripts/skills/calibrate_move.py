"""Print the forward-gait calibration for the current configured robot.

PYTHONPATH=. MUJOCO_GL=egl python scripts/skills/calibrate_move.py
Six seconds per gain, mean world +x speed over the last two seconds.
The explicit back_gain bypasses speed and steering feedback.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.scenario import generate_scenario
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from skills.locomotion import move_forward, SPEED_CURVE


def main():
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range = [0.0, 0.0]
    cfg.scenario.goal.y_range = [-400.0, -400.0]
    env = MujocoRadialSphereEnv(
        cfg, scenario=generate_scenario("goal", cfg, seed=42),
        randomize=False, max_steps=10000,
    )
    dt = env.model.opt.timestep * env.action_repeat
    try:
        print(f"mechanism={cfg.robot.rod_mechanism}, stroke={env.max_extend} m")
        for gain, _ in SPEED_CURVE:
            env.reset(seed=42)
            speeds = []
            for step in range(round(6 / dt)):
                targets = move_forward(env.data.qpos[3:7], env.dirs_body,
                                       env.max_extend, [1, 0], back_gain=gain)
                _, _, terminated, truncated, _ = env.step(targets)
                if terminated or truncated:
                    raise RuntimeError("Calibration run terminated early")
                if step >= round(4 / dt):
                    speeds.append(env.data.qvel[0])
            print(f"({gain:.2f}, {np.mean(speeds):.5f}),", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
