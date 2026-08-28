"""Test two consecutive high launches and apex locks in chimney."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill


def test():
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = generate_scenario("chimney", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    obs, info = env.reset(seed=42)

    # Schedule:
    # 1. Stand 1 (1-20)
    # 2. Crouch 1 (20-32)
    # 3. Launch 1 (32-44) -> rises to z = 0.70m
    # 4. Flight 1 (44-58)
    # 5. Hold 1 (58-120) -> locked at z = 0.70m
    # 6. Descent 1 (120-230) -> slides smoothly to floor z = 0.20m
    # 7. Crouch 2 (230-245) -> settled on floor
    # 8. Launch 2 (245-257) -> rises to z = 0.70m
    # 9. Flight 2 (257-270)
    # 10. Hold 2 (270-330) -> locked at z = 0.70m
    # 11. Final Descent (330-420) -> lands on floor

    total_steps = 420

    for step in range(1, total_steps + 1):
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()
        quat = env.data.qpos[3:7].copy()

        if step <= 20:
            phase = "stand"
            mode = "STAND REST (ON FLOOR)"
        elif step <= 32:
            phase = "crouch"
            mode = "CROUCH PRELOAD 1"
        elif step <= 44:
            phase = "launch"
            mode = "🚀 LAUNCH UP 1"
        elif step <= 58:
            phase = "flight"
            mode = "✈️ ASCENT 1 (CLIMBING UP)"
        elif step <= 120:
            phase = "hold"
            mode = "🔒 APEX WALL CLAMP 1 (MID-AIR)"
        elif step <= 230:
            phase = "descent"
            mode = "🪂 DESCENT 1 (DOWN TO FLOOR)"
        elif step <= 245:
            phase = "crouch"
            mode = "CROUCH PRELOAD 2 (ON FLOOR)"
        elif step <= 257:
            phase = "launch"
            mode = "🚀 LAUNCH UP 2"
        elif step <= 270:
            phase = "flight"
            mode = "✈️ ASCENT 2 (CLIMBING UP)"
        elif step <= 330:
            phase = "hold"
            mode = "🔒 APEX WALL CLAMP 2 (MID-AIR)"
        else:
            phase = "descent"
            mode = "🪂 FINAL DESCENT & TOUCHDOWN"

        targets = execute_skill("chimney_climb", quat, env.dirs_body, env.max_extend,
                                phase=phase, lateral_offset=float(pos[1]))
        env.step(targets)

        if step % 25 == 0 or "LAUNCH" in mode:
            print(f"Step {step:3d} [{mode:34s}]: z = {pos[2]:.3f} m | vz = {vel[2]:+.2f} m/s | y = {pos[1]*100:+.1f} cm")

    env.close()

test()
