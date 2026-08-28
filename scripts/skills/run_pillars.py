"""Hop between tall narrow pillars using velocity-servo standing jumps.

The pads are 0.90 m square and the ball is 0.30 m across the core: nowhere to
roll, so every hop starts from a standstill. Each hop's parameters -- where to
stand, and the take-off velocity to command -- come from `skills/hop_planner`,
which brackets the calibrated take-off envelope against the target pad's own
geometry. Nothing here is hand-tuned per hop; change the pillars and the plans
change with them, or the planner says NO PLAN and the run refuses.

The pillars are 2.2x to 3.5x the core diameter, which needs the long-stroke
build in `configs/rl/pillar_course.yaml`.

    python scripts/skills/run_pillars.py --video
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.render import VideoRecorder
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario, pillar_course_columns
from radial_sphere.snapshot import make_run_dir
from skills import execute_skill
from skills.hop_planner import (PROBE_VX_STEP, PROBE_VZ_STEP, ROLL_RADIUS,
                                STAND_EDGE, plan_standing_hop)
from skills.overlay import annotate

FORWARD = np.array([1.0, 0.0])
CROUCH_STEPS = 22
MAX_BURN = 45


def main():
    p = argparse.ArgumentParser(description="Hop the pillar course")
    p.add_argument("--config", default="configs/rl/pillar_course.yaml")
    p.add_argument("--video", action="store_true")
    p.add_argument("--camera", default="pillar_side")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--frame-every", type=int, default=4)
    p.add_argument("--seed", type=int, default=None,
                   help="randomize the ball's starting orientation")
    p.add_argument("--max-probes", type=int, default=5)
    p.add_argument("--max-restarts", type=int, default=2)
    p.add_argument("--gear", type=float, default=0.5,
                   help="landing-gear extension (fraction of stroke) while falling")
    p.add_argument("--demo-recovery", action="store_true",
                   help="after the second pillar, shove the ball off sideways on "
                        "purpose so the side-lane recovery can be seen")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.floor.square_m = 0.5
    scenario = generate_scenario("pillar_course", cfg, seed=1)
    cols = pillar_course_columns(cfg)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False,
                                max_steps=1_000_000)
    env.reset(seed=1)
    if args.seed is not None:
        import mujoco
        rng = np.random.default_rng(args.seed)
        quat = rng.normal(size=4)
        env.data.qpos[3:7] = quat / np.linalg.norm(quat)
        mujoco.mj_forward(env.model, env.data)
        for _ in range(80):
            env.step(execute_skill("stop", env.data.qpos[3:7].copy(),
                                   env.dirs_body, env.max_extend))

    recorder = None
    if args.video:
        run_dir = make_run_dir(build_run_id("run_pillars", f"pillar_course_seed{args.seed}"))
        out = Path(run_dir) / "renders" / "pillar_course.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        recorder = VideoRecorder(out, fps=args.fps)

    tick = {"n": 0}

    def frame(title, note):
        tick["n"] += 1
        if recorder is None or tick["n"] % args.frame_every:
            return
        recorder.add(annotate(env.render(camera_name=args.camera), title, [
            note,
            f"height {float(env.data.qpos[2]):5.2f} m   "
            f"vz {float(env.data.qvel[2]):+4.1f} m/s",
            f"x      {float(env.data.qpos[0]):5.2f} m",
            f"t {tick['n'] * 0.01:5.1f}s"]))

    def q():
        return env.data.qpos[3:7].copy()

    def hold(steps, title, note):
        for _ in range(steps):
            env.step(execute_skill("stop", q(), env.dirs_body, env.max_extend,
                                   lin_vel=env.data.qvel[0:2].copy()))
            frame(title, note)

    def position_at(x_target, title, note, tol=0.035, max_steps=2500):
        """Creep to x_target and hold still. The stand point is the one part
        of a hop the robot controls exactly, so take the time to nail it.
        Aborts if the ball leaves the surface it started on -- continuing to
        creep after dropping off an edge just grinds against the next wall."""
        z_start = float(env.data.qpos[2])
        for _ in range(max_steps):
            if abs(float(env.data.qpos[2]) - z_start) > 0.15:
                break
            err = x_target - float(env.data.qpos[0])
            if abs(err) < tol and float(np.linalg.norm(env.data.qvel[0:2])) < 0.07:
                break
            if err > tol:
                t = execute_skill("move", q(), env.dirs_body, env.max_extend,
                                  d_hat=FORWARD, speed=0.45)
            elif err < -tol:
                t = execute_skill("reverse", q(), env.dirs_body, env.max_extend,
                                  d_hat=FORWARD, speed=0.45)
            else:
                t = execute_skill("stop", q(), env.dirs_body, env.max_extend,
                                  lin_vel=env.data.qvel[0:2].copy())
            env.step(t)
            frame(title, note)
        hold(80, title, note)

    def slide_y(dy, title, note):
        """Roll sideways by dy. Changes which rods are under the ball -- the
        cheapest way to re-deal the orientation without leaving the spot."""
        y0 = float(env.data.qpos[1])
        for _ in range(400):
            if abs(float(env.data.qpos[1]) - y0) >= abs(dy):
                break
            env.step(execute_skill("move", q(), env.dirs_body, env.max_extend,
                                   d_hat=FORWARD, speed=0.45,
                                   turn=np.pi / 2 if dy > 0 else -np.pi / 2))
            frame(title, note)
        hold(50, title, note)

    def servo_hop(plan, target, title, max_steps=700):
        """Crouch, burn until the commanded velocity is reached, fly, land,
        brake dead. The first steps of the burn are a PROBE: if the launch is
        already below the plan's bracket, tuck and abort while the ball is
        still only centimetres up. Returns (peak, x, z, on, aborted)."""
        deck_core = target["height"] + ROLL_RADIUS
        phase, ps, peak = "crouch", 0, float(env.data.qpos[2])
        vz_best, aborted = -9.0, None
        for _ in range(max_steps):
            z, vz = float(env.data.qpos[2]), float(env.data.qvel[2])
            vx = float(env.data.qvel[0])
            peak = max(peak, z)
            if phase == "crouch" and ps >= CROUCH_STEPS:
                phase, ps = "takeoff", 0
            elif phase == "takeoff":
                vz_best = max(vz_best, vz)
                burn_over = (vz >= plan.vz_cmd or vz < vz_best - 0.15
                             or ps >= MAX_BURN)
                if ps == PROBE_VX_STEP and vx < plan.vx_gate:
                    aborted = f"vx {vx:.2f} < gate {plan.vx_gate:.2f} at step {ps}"
                elif ps == PROBE_VX_STEP and vx > plan.vx_gate_hi:
                    aborted = f"vx {vx:.2f} > gate {plan.vx_gate_hi:.2f} at step {ps} (would overshoot)"
                elif (ps == PROBE_VZ_STEP or burn_over) and vz_best < plan.vz_gate:
                    # Checked at step 8 AND at whatever step the burn ends: a
                    # push that dies at step 6 must not slip past the gate.
                    aborted = f"vz {vz_best:.2f} < gate {plan.vz_gate:.2f} at step {ps}"
                if aborted or burn_over:
                    phase, ps = "airborne", 0
            elif (phase == "airborne" and ps > 6 and vz < 0
                  and z < (deck_core if not aborted else peak) + 0.14):
                phase, ps = "landing", 0
            ps += 1
            drop = max(peak - (target["height"] + ROLL_RADIUS), 0.10)
            env.step(execute_skill("jump_to", q(), env.dirs_body, env.max_extend,
                                   d_hat=FORWARD, phase=phase,
                                   vel=env.data.qvel[0:3].copy(),
                                   vx_target=plan.vx_cmd, vz_target=plan.vz_cmd,
                                   wall_lock=True, drop_height=drop))
            frame(title, ("[PROBE ABORT] weak push, re-deal footing" if aborted
                          else f"[{phase}]  cmd vz {plan.vz_cmd:.1f} vx {plan.vx_cmd:.1f}"))
            if phase == "landing" and ps > 12 and abs(vz) < 0.35:
                break
        hold(60 if aborted else 90, title,
             "[brake] settle" if aborted else "[brake] kill the touchdown speed")
        x, z = float(env.data.qpos[0]), float(env.data.qpos[2])
        on = (target["near"] < x < target["far"]
              and target["height"] + 0.10 < z < target["height"] + 0.55
              and abs(float(env.data.qvel[2])) < 0.3)
        return peak, x, z, on, aborted

    def roll_off(cur, target, title, max_steps=1500):
        """Descend by rolling off the edge, not by jumping. The gap between
        pads (0.12 m) is far narrower than the ball (0.34 m), so a creep off
        the lip crosses it by construction, lands with only the drop's own
        energy, and has none of the jump's scatter. This is `fall_down` doing
        the job it was built for."""
        deck_core = target["height"] + ROLL_RADIUS
        drop = cur["height"] - target["height"]
        # A wall or taller pad within reach of the landing? Put the front
        # rods out part way as a bumper.
        ahead = [c2 for c2 in cols if c2["near"] > target["far"] - 0.05
                 and c2["near"] - target["far"] < 0.8 and c2["height"] > target["height"]]
        brace = 0.35 if ahead else 0.0
        phase, ps = "edge", 0
        for _ in range(max_steps):
            z = float(env.data.qpos[2])
            if phase == "edge" and z < cur["height"] + ROLL_RADIUS - 0.06:
                phase, ps = "freefall", 0
            elif phase == "freefall" and z < deck_core + 0.10:
                phase, ps = "absorb", 0
            elif phase == "absorb" and z < deck_core + 0.03:
                phase, ps = "brake", 0
            ps += 1
            if phase == "brake":
                # The creep speed plus the drop is enough to carry the ball
                # across a 0.9 m pad and off the far side; kill it at once.
                t = execute_skill("stop", q(), env.dirs_body, env.max_extend,
                                  lin_vel=env.data.qvel[0:2].copy(), stop_distance=0.15)
            else:
                t = execute_skill("fall_down", q(), env.dirs_body,
                                  env.max_extend, d_hat=FORWARD, phase=phase,
                                  drop_height=drop, edge_speed=0.35,
                                  gear=args.gear, brace_front=brace)
            env.step(t)
            frame(title, f"[{phase}]  roll off, drop {drop:.2f} m")
            if phase == "brake" and ps > 80:
                break
        hold(60, title, "[brake] settled on the lower pad")
        x, z = float(env.data.qpos[0]), float(env.data.qpos[2])
        on = (target["near"] < x < target["far"]
              and target["height"] + 0.10 < z < target["height"] + 0.55)
        return x, z, on

    # ---- plan the whole course first ---------------------------------
    print(f"pillar course, stroke {cfg.robot.max_extend} m")
    plans, prev_h, prev_range = [], 0.0, (-100.0, 100.0)
    prev_col = None
    for c in cols:
        dz = c["height"] - prev_h
        gap = 0.0 if prev_col is None else c["near"] - prev_col["far"]
        tag = (f"pillar{c['index']} h={c['height']:.2f} "
               f"({c['height'] / 0.30:.1f}x core, dz {dz:+.2f})")
        if prev_col is not None and dz < -0.25 and gap <= 0.15:
            plans.append("roll_off")
            print(f"  {tag}: roll off the edge (gap {gap:.2f} m is narrower "
                  f"than the ball -- no jump needed)")
        else:
            plan = plan_standing_hop(prev_h, prev_range, c)
            plans.append(plan)
            print(f"  {tag}: " + (plan.describe() if plan else "NO PLAN"))
            if plan is None:
                print("refusing to run: a hop is beyond the calibrated envelope")
                env.close()
                return
        prev_h = c["height"]
        prev_range = (c["near"] + STAND_EDGE, c["far"] - STAND_EDGE)
        prev_col = c

    # ---- execute ------------------------------------------------------
    stats = {"probe_aborts": 0, "misses": 0, "restarts": 0}

    def on_floor():
        return float(env.data.qpos[2]) < 0.35

    def on_pad(c):
        x, z = float(env.data.qpos[0]), float(env.data.qpos[2])
        return (c["near"] < x < c["far"]
                and c["height"] + 0.10 < z < c["height"] + 0.55)

    WALL_Y = float(getattr(getattr(cfg.scenario, "pillar_course", None),
                           "width", 2.2)) / 2.0          # corridor half-width
    PAD_Y = cols[0]["half_top"]
    LANE_Y = PAD_Y + (WALL_Y - PAD_Y) / 2.0               # centre of the side lane

    def slide_to_y(y_target, title, note, tol=0.05):
        for _ in range(900):
            y = float(env.data.qpos[1])
            err = y_target - y
            if abs(err) < tol and float(np.linalg.norm(env.data.qvel[0:2])) < 0.08:
                break
            if abs(err) < tol:
                t = execute_skill("stop", q(), env.dirs_body, env.max_extend,
                                  lin_vel=env.data.qvel[0:2].copy())
            elif abs(y) > WALL_Y - 0.48 and err * y < 0:
                # Too close to the corridor wall to roll away from it: the
                # sideways gait plants its pushing rods INTO the wall and
                # pins the ball. Shove off the wall first, then roll.
                t = execute_skill("push_against_wall", q(), env.dirs_body,
                                  env.max_extend,
                                  wall_normal=np.array([0.0, -np.sign(y)]))
            else:
                t = execute_skill("move", q(), env.dirs_body, env.max_extend,
                                  d_hat=FORWARD, speed=0.5,
                                  turn=np.pi / 2 if err > 0 else -np.pi / 2)
            env.step(t)
            frame(title, note)
        hold(40, title, note)

    def recover_to_start(title):
        """Fell off. Take the side lane back to the start and climb again.
        The corridor is 2.2 m wide and the pillars 0.9 m, so there is a
        0.65 m lane down each side that the 0.34 m ball fits through."""
        if not on_floor():
            hold(150, title, "hung up on an edge; wait for it to drop")
            if not on_floor():
                return False
        lane = LANE_Y if float(env.data.qpos[1]) >= 0 else -LANE_Y
        slide_to_y(lane, title, "into the side lane")
        print(f"    recover: in lane at x {float(env.data.qpos[0]):.2f} y {float(env.data.qpos[1]):+.2f}")
        position_at(cols[0]["near"] - 0.9, title, "back down the lane to the start")
        print(f"    recover: at start x {float(env.data.qpos[0]):.2f} y {float(env.data.qpos[1]):+.2f}")
        slide_to_y(0.0, title, "back onto the centre line")
        print(f"    recover: centred x {float(env.data.qpos[0]):.2f} y {float(env.data.qpos[1]):+.2f} z {float(env.data.qpos[2]):.2f}")
        return (on_floor() and float(env.data.qpos[0]) < cols[0]["near"] - 0.5
                and abs(float(env.data.qpos[1])) < 0.15)

    made, restarts = 0, 0
    i = 0
    while i < len(cols):
        c, plan = cols[i], plans[i]
        where = "the floor" if i == 0 else f"pillar{i - 1}"
        if plan == "roll_off":
            cur = cols[i - 1]
            position_at(cur["far"] - 0.26, f"{2 * i + 1}. line up",
                        f"line up short of pillar{i - 1}'s far edge")
            x, z, on = roll_off(cur, c, f"{2 * i + 2}. roll off onto pillar{i}")
            print(f"  hop{i} (roll off): landed x {x:5.2f} y {float(env.data.qpos[1]):+.2f} z {z:.2f}  "
                  f"-> {'ON THE PILLAR' if on else 'MISSED'}")
            if not on:
                break
            made += 1
            i += 1
            continue

        on, probes, attempt = False, 0, 0
        while not on:
            # Landings drift sideways a little every hop; on a 0.9 m pad that
            # compounds into an edge landing. Re-centre before every jump.
            if abs(float(env.data.qpos[1])) > 0.06:
                slide_to_y(0.0, f"{2 * i + 1}. line up", "re-centre on the pad")
            position_at(plan.x0, f"{2 * i + 1}. line up",
                        f"stand at x={plan.x0:.2f} on {where}"
                        + (f"  (probe re-deal {probes})" if probes else "")
                        + (f"  (retry {attempt})" if attempt else ""))
            peak, x, z, on, aborted = servo_hop(plan, c, f"{2 * i + 2}. hop onto pillar{i}")
            if aborted and on:
                print(f"  hop{i} probe abort ({aborted}) -- but it landed on the pad anyway")
                break
            if aborted:
                probes += 1
                stats["probe_aborts"] += 1
                print(f"  hop{i} probe abort #{probes}: {aborted}")
                # An aborted launch still hops a few cm; make sure it is
                # still standing where it can retry.
                still_ok = (on_floor() and x < c["near"] - 0.15) if i == 0 \
                    else on_pad(cols[i - 1])
                if not still_ok:
                    print("  the abort left the launch pad; recovering")
                    if restarts < args.max_restarts and recover_to_start(f"{2 * i + 2}. recover"):
                        restarts += 1
                        stats["restarts"] += 1
                        print(f"  recovered to the start (restart {restarts})")
                        made, i = 0, -1
                    break
                if probes >= args.max_probes:
                    print("  too many weak launches from this spot; jumping anyway")
                else:
                    # Re-deal the footing: a growing sideways shuffle,
                    # alternating sides so it never walks off the pad.
                    dy = (0.06 + 0.03 * probes) * (1 if probes % 2 else -1)
                    slide_y(dy, f"{2 * i + 1}. re-deal",
                            "shuffle sideways to change the footing")
                    continue
            print(f"  hop{i}{' (retry %d)' % attempt if attempt else ''}: peak {peak:.2f}"
                  f"  landed x {x:5.2f} y {float(env.data.qpos[1]):+.2f} z {z:.2f}"
                  f"  predicted {plan.land_lo:.2f}..{plan.land_hi:.2f}"
                  f"  -> {'ON THE PILLAR' if on else 'MISSED'}")
            if on:
                break
            # A ball on a lip or in a gap is still moving; let it settle,
            # then look again before deciding what kind of miss this was.
            hold(120, f"{2 * i + 2}. hop onto pillar{i}", "[settle] where did it end up?")
            x, z = float(env.data.qpos[0]), float(env.data.qpos[2])
            if on_pad(c):
                print(f"  hop{i}: settled onto pillar{i} after all (x {x:.2f} z {z:.2f})")
                on = True
                break
            # Overshot onto a LATER pad? That is progress, not a miss.
            later = [j for j in range(i + 1, len(cols)) if on_pad(cols[j])]
            if later:
                j = later[-1]
                print(f"  overshot straight onto pillar{j}; carrying on from there")
                made += (j - i)
                i = j
                on = True
                break
            stats["misses"] += 1
            attempt += 1
            # Retry in place only from somewhere it can genuinely launch:
            # the floor short of the first pillar, or the previous pad. On
            # the floor BESIDE a pillar it must go round via the lane.
            can_retry_here = ((i == 0 and on_floor() and x < c["near"] - 0.15)
                              or (i > 0 and on_pad(cols[i - 1])))
            if can_retry_here and attempt < 4:
                continue
            # Otherwise: down the side lane and restart the ladder.
            if restarts < args.max_restarts and recover_to_start(f"{2 * i + 2}. recover"):
                restarts += 1
                stats["restarts"] += 1
                print(f"  fell off -> recovered to the start (restart {restarts})")
                made, i = 0, -1
                break
            break
        if i < 0:
            i = 0
            continue
        if not on:
            break
        made += 1
        i += 1
        if args.demo_recovery and i == 2 and stats["restarts"] == 0:
            # Staged: knock it off the pad so the recovery path runs on camera.
            for _ in range(70):
                env.step(execute_skill("move", q(), env.dirs_body, env.max_extend,
                                       d_hat=FORWARD, speed=1.0, turn=np.pi / 2))
                frame("DEMO", "[staged] shoved off the pad on purpose")
            hold(60, "DEMO", "[staged] fell to the floor beside the pillars")
            print("  demo: shoved off the pad on purpose")
            if recover_to_start("recover"):
                restarts += 1
                stats["restarts"] += 1
                print(f"  recovered to the start (restart {restarts})")
                made, i = 0, 0

    if made == len(cols):
        hold(150, f"{2 * len(cols) + 1}. stop",
             f"balanced on pillar{len(cols) - 1}, course complete")
    print(f"\n{made}/{len(cols)} pillars made, "
          f"{stats['probe_aborts']} probe aborts, {stats['misses']} misses, "
          f"{stats['restarts']} restarts, "
          f"{tick['n']} steps = {tick['n'] * 0.01:.1f}s of real time")

    if recorder is not None:
        recorder.close()
        print(f"video: {recorder.path}  ({recorder.n_frames / args.fps:.1f}s)")
    env.close()


if __name__ == "__main__":
    main()
