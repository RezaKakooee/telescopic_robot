"""Measure what the jump take-off actually produces, so it can be planned.

The jump has three free numbers: where to start the crouch, how long to hold
it, and how hard to run up. None of them can be derived from first
principles, because the rods reload at a finite rate and the ball is rolling
while they do it. So we measure the map once:

    (run-up gain, crouch steps)  ->  take-off state

and write it to a table. `skills/jump_planner.py` inverts that table at run
time: it works out the take-off velocity an obstacle demands, then reads back
the crouch length and trigger distance that deliver it.

    python scripts/skills/calibrate_jump.py            # writes the table
    python scripts/skills/calibrate_jump.py --quick    # coarse, for a smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill

OUT = Path(__file__).resolve().parents[2] / "skills" / "jump_calibration.json"
LAUNCH_STEPS = 16          # held fixed; crouch length is the knob we vary
RUN_UP_M = 3.5             # distance to reach cruise before the crouch starts


def one_trial(env, d_hat, gain, crouch_steps, run_up_m=RUN_UP_M, phase_offset=0):
    """Run up, crouch, launch. Return the state at the moment of take-off."""
    env.reset(seed=42)
    q = lambda: env.data.qpos[3:7].copy()
    for _ in range(40):
        env.step(execute_skill("stop", q(), env.dirs_body, env.max_extend))

    x_start = float(env.data.qpos[0])
    # Run up until we have covered run_up_m, so the gait is at cruise.
    for _ in range(2000):
        if float(env.data.qpos[0]) - x_start >= run_up_m:
            break
        env.step(execute_skill("move_forward", q(), env.dirs_body,
                               env.max_extend, d_hat=d_hat, back_gain=gain))
    # Delay the crouch by a few steps to land on a different point of the
    # gait cycle; this is what the sampling above varies.
    for _ in range(phase_offset):
        env.step(execute_skill("move_forward", q(), env.dirs_body,
                               env.max_extend, d_hat=d_hat, back_gain=gain))
    vx_cruise = float(env.data.qvel[0])
    x_dip = float(env.data.qpos[0])

    for _ in range(crouch_steps):
        env.step(execute_skill("jump_forward_while_moving", q(), env.dirs_body,
                               env.max_extend, d_hat=d_hat, phase="dip"))
    for _ in range(LAUNCH_STEPS):
        env.step(execute_skill("jump_forward_while_moving", q(), env.dirs_body,
                               env.max_extend, d_hat=d_hat, phase="launch"))

    # Take-off: the ball leaves the ground at the end of the launch.
    x0, z0 = float(env.data.qpos[0]), float(env.data.qpos[2])
    vx0 = float(env.data.qvel[0])

    # Fly it out and record the arc. The rods keep pushing for a few steps
    # after the phase switch, so we describe the flight by its measured apex
    # rather than by an instantaneous velocity reading.
    peak, dx_apex = z0, 0.0
    for _ in range(110):
        env.step(execute_skill("jump_forward_while_moving", q(), env.dirs_body,
                               env.max_extend, d_hat=d_hat, phase="airborne"))
        z = float(env.data.qpos[2])
        if z > peak:
            peak, dx_apex = z, float(env.data.qpos[0]) - x0

    return {
        "gain": gain,
        "crouch_steps": crouch_steps,
        "vx_cruise": vx_cruise,
        "travel_dip_to_takeoff": x0 - x_dip,
        "z_takeoff": z0,
        "vx_takeoff": vx0,
        "peak_z": peak,
        "dx_to_apex": dx_apex,
        "rise": peak - z0,
    }


def main():
    p = argparse.ArgumentParser(description="Calibrate the jump take-off")
    p.add_argument("--config", default="configs/rl/skill_course.yaml")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    # Flat, empty arena: no walls or boxes to interfere with the run-up.
    cfg.scenario.goal.x_range = [0.0, 0.0]
    cfg.scenario.goal.y_range = [-400.0, -400.0]
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False,
                                max_steps=1_000_000)
    d_hat = np.array([1.0, 0.0])

    gains = [1.6, 2.4] if args.quick else [1.6, 2.0, 2.4]
    crouches = [12, 24] if args.quick else [12, 18, 24, 30]
    # Identical commands give peaks between 0.35 m and 0.61 m depending on
    # where the rolling gait is when the crouch starts. The cycle is about
    # 8 env steps long, so we sample a whole period by delaying the crouch
    # 0..7 steps, and the planner is handed the WORST peak of the set. That
    # turns a lottery into a guarantee it can plan against.
    phase_offsets = [0, 4] if args.quick else list(range(8))

    rows = []
    print(f"{'gain':>5}{'crouch':>7}{'vx_cru':>8}{'travel':>8}"
          f"{'z_tko':>7}{'vx_tko':>8}{'pk_min':>8}{'pk_mean':>8}{'pk_max':>8}{'dx_ap':>7}")
    for gain in gains:
        for crouch in crouches:
            samples = [one_trial(env, d_hat, gain, crouch, RUN_UP_M, off)
                       for off in phase_offsets]
            worst = min(samples, key=lambda r: r["peak_z"])
            agg = {
                "gain": gain,
                "crouch_steps": crouch,
                "n_samples": len(samples),
                "vx_cruise": float(np.mean([s["vx_cruise"] for s in samples])),
                "travel_dip_to_takeoff": float(np.mean(
                    [s["travel_dip_to_takeoff"] for s in samples])),
                # Conservative arc: lowest rise seen, with its own geometry.
                "z_takeoff": worst["z_takeoff"],
                "vx_takeoff": float(np.mean([s["vx_takeoff"] for s in samples])),
                # The guarantee: lowest peak seen across a full gait cycle.
                "peak_guaranteed": worst["peak_z"],
                "dx_to_apex": worst["dx_to_apex"],
                "peak_mean": float(np.mean([s["peak_z"] for s in samples])),
                "peak_best": float(np.max([s["peak_z"] for s in samples])),
            }
            rows.append(agg)
            print(f"{gain:>5.1f}{crouch:>7}{agg['vx_cruise']:>8.2f}"
                  f"{agg['travel_dip_to_takeoff']:>8.2f}{agg['z_takeoff']:>7.3f}"
                  f"{agg['vx_takeoff']:>8.2f}{agg['peak_guaranteed']:>8.3f}"
                  f"{agg['peak_mean']:>8.3f}{agg['peak_best']:>8.3f}"
                  f"{agg['dx_to_apex']:>7.2f}")
    env.close()

    OUT.write_text(json.dumps({
        "launch_steps": LAUNCH_STEPS,
        "run_up_m": RUN_UP_M,
        "config": args.config,
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
