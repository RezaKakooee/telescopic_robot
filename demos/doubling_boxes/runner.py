"""Hop up five decks and hop down four, each step twice the last, a pit
between every pair.

The decks are too short to roll on, so this is the pillar course's method:
stand at a planned spot, hop with a velocity-commanded take-off (`jump_to`),
land, brake dead, creep to the next stand point, hop again. Every hop's
stand point and command come from `skills/mid_level/hop_planner`, computed
from the next deck's own height and edges; if the planner cannot promise a
hop it says so and the run refuses. A pit narrower than `ROLL_OFF_MAX_PIT`
before a lower deck is not hopped but rolled off (`fall_down`): the rods
span it.

The decks are 0.45, 0.50, 0.60, 0.80, 1.20, 0.80, 0.60, 0.50, 0.45 m tall:
the RISES and DROPS double (0.05, 0.10, 0.20, 0.40 m). The heights
themselves cannot double from a low first deck: the rods reach 0.41 m, so
on a deck under about 0.3 m they stand on the floor beside it and the push
is weak and uneven (measured: a hop from a 0.05 or 0.10 m deck aborts or
barely lifts). And a rise past about 0.45 m is outside the hop calibration.

It needs the long-stroke build in `configs/rl/doubling_boxes.yaml`: a
standing jump goes high but not far, and the stroke sets the height.

Measured, and why the numbers in the config are what they are:

* the hops land up to 0.6 m off the planner's bracket, so 1.0 m decks are
  at the scatter's limit (1 of 5 starts made the way up); decks are 1.2 m;
* the creep cannot go below 1 m/s on this build, so the line-up is pulsed
  (a short push, a stop, again), else it overshoots the mark by 0.4 m and
  off the lip on the way down;
* the pits on the way up are 0.25 m; on the way down 0.35 m, wider than the
  0.30 m core so the ball must hop, not roll. The weakest hop cannot cross
  0.45 m, so the planner has to use the strong cells, and those scatter
  about 1 m on a descent (longer flight). Each descent hop lands on its
  deck about 85 % of the time; the whole course, nine hops, 1 of 8 start
  orientations. With `valley_down: 0.25` the descent is rolled off instead
  (4 of 6 starts made the whole course that way).

    python demos/doubling_boxes/runner.py --video
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from radial_sphere.config import load_config, demo_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.overlay import annotate
from radial_sphere.render import VideoRecorder
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir
from skills import execute_skill
from skills.mid_level.hop_planner import (PROBE_VX_STEP, PROBE_VZ_STEP, ROLL_RADIUS,
                                          STAND_EDGE, plan_standing_hop)

FORWARD = np.array([1.0, 0.0])
CROUCH_STEPS = 22
MAX_BURN = 45
ROLL_OFF_MAX_PIT = 0.30    # a pit the rods (0.41 m reach) span: roll off it; wider ones are hopped
PULSE_FROM = 0.60          # creep in pulses only this close to the mark; further out roll steadily


def decks_of(scenario) -> list[dict]:
    """The boxes in travel order, with their edges worked out."""
    out = []
    for i, (x, _y, hx, _hy, h) in enumerate(sorted(np.asarray(scenario.steps, dtype=float).tolist())):
        out.append({"index": i, "x": x, "height": h, "near": x - hx, "far": x + hx})
    return out


def main():
    args = demo_config("doubling_boxes").knobs

    cfg = load_config(args.config)
    cfg.floor.square_m = 0.5
    scenario = generate_scenario(cfg.scenario.kind, cfg, seed=1)
    decks = decks_of(scenario)
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
        run_dir = make_run_dir(build_run_id("run_doubling_boxes", f"seed{args.seed}"))
        out = Path(run_dir) / "renders" / "doubling_boxes.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        recorder = VideoRecorder(out, fps=args.fps)

    tick = {"n": 0}

    def chase_view():
        """The inspection videos' camera: tracking, steep, looking along the
        row. The heading is fixed (the expert videos follow the velocity, but
        here the ball creeps backwards and sideways between hops, and a
        following camera swings round every time)."""
        import mujoco
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = env.core_body_id
        cam.distance, cam.elevation, cam.azimuth = 3.2, -48.0, 0.0
        if env.renderer is None:
            env.render()                                 # creates the renderer
        env.renderer.update_scene(env.data, camera=cam)
        return env.renderer.render().copy()

    def frame(title, note):
        tick["n"] += 1
        if recorder is None or tick["n"] % args.frame_every:
            return
        img = chase_view() if args.camera == "chase" else env.render(camera_name=args.camera)
        recorder.add(annotate(img, title, [
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
        """Creep to x_target in pulses and hold still. The stand point is the
        one part of a hop the robot controls exactly, so take the time to
        nail it. The gait cannot cruise below 1 m/s on this build, so a
        steady creep overshoots the mark by up to 0.4 m: on the way up that
        ends against the next deck's face, on the way down it ends in the
        pit. Pulsing (a short push, then a stop) keeps the ball slow, and the
        push gets shorter as the mark gets nearer. Aborts if the ball leaves
        the surface it started on."""
        z_start = float(env.data.qpos[2])
        n = 0
        while n < max_steps:
            if abs(float(env.data.qpos[2]) - z_start) > 0.15:
                break
            err = x_target - float(env.data.qpos[0])
            if abs(err) < tol and float(np.linalg.norm(env.data.qvel[0:2])) < 0.07:
                break
            if abs(err) > PULSE_FROM:
                # Far from the mark: roll steadily, no pulsing (it looks like
                # a stutter and takes 13 s over the run-in from the start).
                t = execute_skill("move" if err > 0 else "reverse", q(), env.dirs_body,
                                  env.max_extend, d_hat=FORWARD, speed=0.45)
                env.step(t)
                frame(title, note)
                n += 1
                continue
            push = int(np.clip(6 + 40 * abs(err), 6, 25))
            for _ in range(push):
                t = execute_skill("move" if err > 0 else "reverse", q(), env.dirs_body,
                                  env.max_extend, d_hat=FORWARD, speed=0.45)
                env.step(t)
                frame(title, note)
            for _ in range(25):
                env.step(execute_skill("stop", q(), env.dirs_body, env.max_extend,
                                       lin_vel=env.data.qvel[0:2].copy()))
                frame(title, note)
            n += push + 25
        hold(80, title, note)

    def slide_y(dy, title, note):
        """Roll sideways by dy: re-deals which rods are under the ball."""
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
        brake dead. The first steps of the burn are a probe: a launch below
        the plan's bracket is aborted while the ball is still only
        centimetres up. Returns (peak, x, z, on, aborted)."""
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
        """Descend by rolling off the edge, not by jumping. The pit is
        narrower than the ball's rod span, so a slow creep off the lip carries
        the ball across it while it drops, with only the drop's own energy and
        none of a jump's scatter. `fall_down` does the job it was built for."""
        deck_core = target["height"] + ROLL_RADIUS
        drop = cur["height"] - target["height"]
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
                # The creep plus the drop would carry the ball across a 1 m
                # deck and off its far side; kill it at once.
                t = execute_skill("stop", q(), env.dirs_body, env.max_extend,
                                  lin_vel=env.data.qvel[0:2].copy(), stop_distance=0.15)
            else:
                t = execute_skill("fall_down", q(), env.dirs_body,
                                  env.max_extend, d_hat=FORWARD, phase=phase,
                                  drop_height=drop, edge_speed=args.edge_speed, gear=args.gear)
            env.step(t)
            frame(title, f"[{phase}]  roll off, drop {drop:.2f} m")
            if phase == "brake" and ps > 80:
                break
        hold(60, title, "[brake] settled on the lower deck")
        x, z = float(env.data.qpos[0]), float(env.data.qpos[2])
        on = (target["near"] < x < target["far"]
              and target["height"] + 0.10 < z < target["height"] + 0.55)
        return x, z, on

    def on_deck(d):
        x, z = float(env.data.qpos[0]), float(env.data.qpos[2])
        return d["near"] < x < d["far"] and d["height"] + 0.10 < z < d["height"] + 0.55

    # ---- plan the whole row first --------------------------------------
    # The floor past the last deck is the last "deck" to drop onto.
    valley = decks[-1]["near"] - decks[-2]["far"]
    decks.append({"index": len(decks), "height": 0.0,
                  "near": decks[-1]["far"] + valley, "far": float(scenario.goal[0])})
    print(f"doubling boxes, stroke {cfg.robot.max_extend} m, {len(decks) - 1} decks of "
          f"{decks[0]['far'] - decks[0]['near']:.2f} m, then the floor")
    plans, prev_h, prev_range = [], 0.0, (-100.0, 100.0)
    prev_d = None
    for d in decks:
        dz = d["height"] - prev_h
        pit = 0.0 if prev_d is None else d["near"] - prev_d["far"]
        tag = f"deck{d['index']} h={d['height']:.2f} (dz {dz:+.2f}, pit {pit:.2f})"
        if dz < 0 and pit <= ROLL_OFF_MAX_PIT:
            # The rods span a narrow pit: roll off the lip, no jump needed.
            plans.append("roll_off")
            print(f"  {tag}: roll off the edge, across the pit")
        else:
            plan = plan_standing_hop(prev_h, prev_range, d)
            plans.append(plan)
            print(f"  {tag}: " + (plan.describe() if plan else "NO PLAN"))
            if plan is None:
                print("refusing to run: a hop is beyond the calibrated envelope")
                env.close()
                return
        prev_h = d["height"]
        prev_range = (d["near"] + STAND_EDGE, d["far"] - STAND_EDGE)
        prev_d = d

    # ---- execute ---------------------------------------------------------
    stats = {"probe_aborts": 0, "misses": 0}
    made = 0
    skip_to = None
    if args.start_on is not None:
        # Testing knob: begin standing on deck `start_on`, skipping the hops up.
        import mujoco
        d0 = decks[int(args.start_on)]
        env.data.qpos[0:3] = [d0["near"] + 0.5, 0.0, d0["height"] + ROLL_RADIUS + 0.05]
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        hold(100, "start", f"placed on deck{d0['index']}")
        skip_to = int(args.start_on)
    for i, (d, plan) in enumerate(zip(decks, plans)):
        if skip_to is not None and i <= skip_to:
            made += 1                                    # overshot onto this one earlier
            continue
        where = "the floor" if i == 0 else f"deck{i - 1}"
        if plan == "roll_off":
            cur = decks[i - 1]
            # 0.50 m back, not the pillar course's 0.26: the creep cannot go
            # slower than 1 m/s on this build and overshoots the mark by up
            # to 0.2 m, which from 0.26 m is off the edge and into the pit.
            position_at(cur["far"] - 0.50, f"{2 * i + 1}. line up",
                        f"line up short of deck{i - 1}'s far edge")
            x, z, on = roll_off(cur, d, f"{2 * i + 2}. drop onto "
                                + ("the floor" if d["height"] == 0.0 else f"deck{i}"))
            later = [j for j in range(i + 1, len(decks)) if on_deck(decks[j])]
            print(f"  drop{i}: landed x {x:5.2f} y {float(env.data.qpos[1]):+.2f} z {z:.2f}  "
                  f"-> {'ON THE DECK' if on else ('OVERSHOT onTO deck%d' % later[-1] if later else 'MISSED')}")
            if later and not on:
                skip_to = later[-1]                   # overshot a deck: carry on from there
                made += 1
                continue
            if not on:
                break
            made += 1
            continue
        on, probes, attempt = False, 0, 0
        while not on:
            if abs(float(env.data.qpos[1])) > 0.10:
                slide_y(-float(env.data.qpos[1]), f"{2 * i + 1}. line up", "re-centre on the deck")
            position_at(plan.x0, f"{2 * i + 1}. line up",
                        f"stand at x={plan.x0:.2f} on {where}"
                        + (f"  (probe re-deal {probes})" if probes else "")
                        + (f"  (retry {attempt})" if attempt else ""))
            peak, x, z, on, aborted = servo_hop(plan, d, f"{2 * i + 2}. hop onto deck{i}")
            if aborted and on:
                print(f"  hop{i} probe abort ({aborted}) -- but it landed on the deck anyway")
                break
            if aborted:
                probes += 1
                stats["probe_aborts"] += 1
                print(f"  hop{i} probe abort #{probes}: {aborted}")
                still_ok = (x < d["near"] - 0.15) if i == 0 else on_deck(decks[i - 1])
                if not still_ok:
                    print("  the abort left the launch deck; giving up")
                    break
                if probes < args.max_probes:
                    dy = (0.06 + 0.03 * probes) * (1 if probes % 2 else -1)
                    slide_y(dy, f"{2 * i + 1}. re-deal", "shuffle sideways to change the footing")
                    continue
                print("  too many weak launches from this spot; jumping anyway")
            print(f"  hop{i}{' (retry %d)' % attempt if attempt else ''}: peak {peak:.2f}"
                  f"  landed x {x:5.2f} y {float(env.data.qpos[1]):+.2f} z {z:.2f}"
                  f"  predicted {plan.land_lo:.2f}..{plan.land_hi:.2f}"
                  f"  -> {'ON THE DECK' if on else 'MISSED'}")
            if on:
                break
            hold(120, f"{2 * i + 2}. hop onto deck{i}", "[settle] where did it end up?")
            if on_deck(d):
                print(f"  hop{i}: settled onto deck{i} after all")
                on = True
                break
            # Overshot onto a LATER deck? That is progress, not a miss.
            later = [j for j in range(i + 1, len(decks)) if on_deck(decks[j])]
            if later:
                skip_to = later[-1]
                print(f"  overshot straight onto deck{skip_to}; carrying on from there")
                on = True
                break
            stats["misses"] += 1
            attempt += 1
            can_retry = (i == 0 and float(env.data.qpos[0]) < d["near"] - 0.15) or (i > 0 and on_deck(decks[i - 1]))
            if not (can_retry and attempt <= args.max_retries):
                break
        if not on:
            break
        made += 1

    # ---- on to the goal ---------------------------------------------------
    if made == len(decks):
        goal_x = float(scenario.goal[0])
        for _ in range(3000):
            if float(env.data.qpos[0]) > goal_x - 0.5:
                break
            env.step(execute_skill("move", q(), env.dirs_body, env.max_extend,
                                   d_hat=FORWARD, speed=0.8))
            frame(f"{2 * len(decks) + 2}. move", "on to the goal")
        hold(80, f"{2 * len(decks) + 3}. stop", "stopped at the goal")

    print(f"\n{made}/{len(decks)} decks made, {stats['probe_aborts']} probe aborts, "
          f"{stats['misses']} misses, {tick['n']} steps = {tick['n'] * 0.01:.1f}s of real time, "
          f"final x {float(env.data.qpos[0]):.2f} z {float(env.data.qpos[2]):.2f}")
    if recorder is not None:
        recorder.close()
        print(f"video: {recorder.path}  ({recorder.n_frames / args.fps:.1f}s)")
    env.close()


if __name__ == "__main__":
    main()
