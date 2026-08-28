"""Test clean dual-climb (Up -> Hold -> Down to floor -> Up -> Hold -> Down to floor)."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill


def test():
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = generate_scenario("chimney", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    obs, info = env.reset(seed=42)

    # 1. Stand at rest (steps 1-20)
    # 2. Crouch 1 (steps 20-32)
    # 3. Launch UP 1 (steps 32-44) -> rises to z = 0.75m
    # 4. Flight 1 (steps 44-58)
    # 5. Apex Lock 1 (steps 58-120) -> LOCKED IN MID-AIR AT z = 0.75m
    # 6. Descent 1 to Floor (steps 120-210) -> lands at z = 0.20m
    # 7. Crouch 2 on Floor (steps 210-222)
    # 8. Launch UP 2 (steps 222-234) -> rises to z = 0.75m
    # 9. Flight 2 (steps 234-248)
    # 10. Apex Lock 2 (steps 248-310) -> LOCKED IN MID-AIR AT z = 0.75m
    # 11. Final Descent to Floor (steps 310-390) -> lands at z = 0.20m

    total_steps = 390
    max_z = 0.0

    for step in range(1, total_steps + 1):
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        max_z = max(max_z, float(pos[2]))

        if step <= 20:
            phase = "stand"
            mode = "STAND REST (ON FLOOR)"
        elif step <= 32:
            phase = "crouch"
            mode = "CROUCH PRELOAD 1"
        elif step <= 44:
            phase = "launch"
            mode = "🚀 EXPLOSIVE LAUNCH UP 1"
        elif step <= 58:
            phase = "flight"
            mode = "✈️ ASCENT 1 (CLIMBING UP)"
        elif step <= 120:
            phase = "hold"
            mode = "🔒 APEX WALL CLAMP 1 (MID-AIR)"
        elif step <= 210:
            phase = "descent"
            mode = "🪂 DESCENT 1 (DOWN TO FLOOR)"
        elif step <= 222:
            phase = "crouch"
            mode = "CROUCH PRELOAD 2 (ON FLOOR)"
        elif step <= 234:
            phase = "launch"
            mode = "🚀 EXPLOSIVE LAUNCH UP 2"
        elif step <= 248:
            phase = "flight"
            mode = "✈️ ASCENT 2 (CLIMBING UP)"
        elif step <= 310:
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

    print("\n" + "=" * 70)
    print(f"  Max Height Reached: z = {max_z:.3f} m (+{(max_z - 0.20)*100:.1f} cm above floor)")
    print(f"  Final Landed Height: z = {pos[2]:.3f} m")
    print("=" * 70)
    env.close()

test()
