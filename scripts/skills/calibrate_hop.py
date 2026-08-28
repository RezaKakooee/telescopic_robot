"""Calibrate the velocity-servo standing hop (`jump_to`).

For a grid of commanded take-off velocities, run the hop on flat ground and
measure what the flight actually did:

    vz_eff  -- from the apex: sqrt(2 g rise)
    vx_eff  -- from where the ball came back down to launch height:
               dx_land * g / (2 * vz_eff), i.e. averaged over the whole arc

Each cell is repeated with a different settle jitter so the ball's orientation
at crouch differs, and the WORST vz and the vx spread are stored. The hop
planner works from those, so its promises hold on a bad day, not a lucky one.

    python scripts/skills/calibrate_hop.py
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

OUT = Path(__file__).resolve().parents[2] / "skills" / "hop_calibration.json"
CROUCH_STEPS = 22
MAX_BURN = 45          # give up on the burn after this many steps
G = 9.81


def one_hop(env, d_hat, vx_cmd, vz_cmd, rng, args=None):
    """One hop from a RANDOM orientation.

    Extra settle steps do not change the orientation of a ball that is
    standing still -- an earlier calibration jittered exactly that way,
    sampled near-identical states three times, and reported a spread five
    times tighter than reality. The orientation is randomized outright.
    """
    import mujoco
    env.reset(seed=42)
    quat = rng.normal(size=4)
    env.data.qpos[3:7] = quat / np.linalg.norm(quat)
    mujoco.mj_forward(env.model, env.data)
    q = lambda: env.data.qpos[3:7].copy()
    for _ in range(80):                        # settle onto the new stance
        env.step(execute_skill("stop", q(), env.dirs_body, env.max_extend))
    x0, z0 = float(env.data.qpos[0]), float(env.data.qpos[2])

    phase, ps = "crouch", 0
    peak, lift = z0, None
    vz_best = -9.0
    for _ in range(500):
        z = float(env.data.qpos[2])
        vz = float(env.data.qvel[2])
        peak = max(peak, z)
        if phase == "crouch" and ps >= CROUCH_STEPS:
            phase, ps = "takeoff", 0
        elif phase == "takeoff":
            # The burn is not free: the asymmetric push runs the ball ALONG
            # the launch surface before it leaves it. Lift-off is the moment
            # vz stops growing -- past it the ball is ballistic whatever the
            # phase says -- so the state at the vz peak IS the launch state.
            if vz > vz_best:
                vz_best = vz
                lift = dict(travel=float(env.data.qpos[0]) - x0, z_lift=z,
                            vx_lift=float(env.data.qvel[0]))
            if vz >= vz_cmd or vz < vz_best - 0.15 or ps >= MAX_BURN:
                phase, ps = "airborne", 0
        elif phase == "airborne" and ps > 60:
            break
        ps += 1
        env.step(execute_skill("jump_to", q(), env.dirs_body, env.max_extend,
                               d_hat=d_hat, phase=phase,
                               vel=env.data.qvel[0:3].copy(),
                               vx_target=vx_cmd, vz_target=vz_cmd,
                               wall_lock=args.wall_lock))
    if lift is None:
        return None
    rise = max(peak - lift["z_lift"], 0.0)
    return dict(**lift, vz_eff=float(np.sqrt(2 * G * rise)), z0=z0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rl/pillar_course.yaml")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--wall-lock", action="store_true", default=True,
                   help="burn with the leading sector shut (course mode)")
    p.add_argument("--vz-grid", type=float, nargs="+",
                   default=[3.0, 3.6, 4.2, 4.8])
    p.add_argument("--vx-grid", type=float, nargs="+",
                   default=[0.6, 0.9, 1.2])
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.scenario.goal.x_range = [0.0, 0.0]
    cfg.scenario.goal.y_range = [-400.0, -400.0]
    sc = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=sc, randomize=False,
                                max_steps=1_000_000)
    d_hat = np.array([1.0, 0.0])

    rows = []
    print(f"{'vz_cmd':>7}{'vx_cmd':>7}{'vz_lo':>8}{'vz_hi':>8}"
          f"{'vx_lo':>8}{'vx_hi':>8}{'trv_lo':>8}{'trv_hi':>8}{'z_lift':>8}")
    for vz_cmd in args.vz_grid:
        for vx_cmd in args.vx_grid:
            rng = np.random.default_rng(hash((vz_cmd, vx_cmd)) % 2**32)
            trials = [one_hop(env, d_hat, vx_cmd, vz_cmd, rng, args)
                      for _ in range(args.repeats)]
            ok = [t for t in trials if t is not None]
            if not ok:
                print(f"{vz_cmd:>7.1f}{vx_cmd:>7.1f}  (no clean burn)")
                continue
            row = dict(
                vz_cmd=vz_cmd, vx_cmd=vx_cmd,
                vz_eff_min=min(t["vz_eff"] for t in ok),
                vz_eff_max=max(t["vz_eff"] for t in ok),
                vx_lift_min=min(t["vx_lift"] for t in ok),
                vx_lift_max=max(t["vx_lift"] for t in ok),
                travel_min=min(t["travel"] for t in ok),
                travel_max=max(t["travel"] for t in ok),
                z_lift_mean=float(np.mean([t["z_lift"] for t in ok])),
                z0=float(np.mean([t["z0"] for t in ok])),
                n=len(ok),
            )
            rows.append(row)
            print(f"{vz_cmd:>7.1f}{vx_cmd:>7.1f}{row['vz_eff_min']:>8.2f}"
                  f"{row['vz_eff_max']:>8.2f}{row['vx_lift_min']:>8.2f}"
                  f"{row['vx_lift_max']:>8.2f}{row['travel_min']:>8.2f}"
                  f"{row['travel_max']:>8.2f}{row['z_lift_mean']:>8.2f}")
    env.close()

    OUT.write_text(json.dumps({
        "config": args.config, "crouch_steps": CROUCH_STEPS,
        "max_burn": MAX_BURN, "wall_lock": args.wall_lock,
        "rows": rows}, indent=2))
    print(f"\nwrote {len(rows)} cells -> {OUT}")


if __name__ == "__main__":
    main()
