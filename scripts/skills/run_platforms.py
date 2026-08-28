"""Run the platform course: clear a box, climb a stack, hop between decks, drop off.

A straight corridor studded with boxes. The robot clears the first from the
floor, jumps onto the next, steps up to a taller one, hops a gap to a third at
the same height, drops across to a lower fourth, then falls back to the floor
and runs to the goal.

Every jump's parameters come from `skills/jump_planner.py`, computed from that
box's own height, depth and the gap in front of it. Nothing here is tuned by
hand; if the planner cannot guarantee a jump it says so and the run stops
rather than driving into the box.

    python scripts/skills/run_platforms.py --video
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
from radial_sphere.scenario import generate_scenario, platform_course_boxes
from radial_sphere.snapshot import make_run_dir
from skills import execute_skill
from skills.jump_planner import plan_jump
from skills.overlay import annotate

FORWARD = np.array([1.0, 0.0])
SETTLE_STEPS = 70      # brake and steady up on a deck after landing
RUNUP_M = 1.30         # deck length wanted ahead of a hop's trigger point
JUMP_LAND_MARGIN = 0.10
BALL_UNDERSIDE = 0.165


def drive_to(env, x_target, gain, on_frame, label, note, max_steps=3000):
    """Roll forward until the ball reaches x_target."""
    n = 0
    while n < max_steps and float(env.data.qpos[0]) < x_target:
        env.step(execute_skill("move_forward", env.data.qpos[3:7].copy(),
                               env.dirs_body, env.max_extend,
                               d_hat=FORWARD, back_gain=gain))
        on_frame(label, note)
        n += 1
    return n


def hold_still(env, steps, on_frame, label, note):
    for _ in range(steps):
        env.step(execute_skill("stop", env.data.qpos[3:7].copy(),
                               env.dirs_body, env.max_extend,
                               lin_vel=env.data.qvel[0:2].copy()))
        on_frame(label, note)
    return steps


def do_jump(env, box, plan, on_frame, label, note, deck_z, max_steps=2500):
    """Run up, crouch, launch, fly, land. Phases fire on measured distance."""
    phase, phase_steps, n = "sprint", 0, 0
    peak = float(env.data.qpos[2])
    box_contacts = 0
    import mujoco
    # Only THIS box's geoms count. The ball is standing on the previous deck
    # for the whole run-up, and those contacts are not mistakes.
    suffix = f"_{box['index']}"
    box_ids = set()
    for i in range(env.model.ngeom):
        nm = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith("wood_") and nm.endswith(suffix):
            box_ids.add(i)

    def touching_box():
        return any(env.data.contact[c].geom1 in box_ids or env.data.contact[c].geom2 in box_ids
                   for c in range(env.data.ncon))

    while n < max_steps:
        x, z = float(env.data.qpos[0]), float(env.data.qpos[2])
        peak = max(peak, z)
        gap_to_face = box["near"] - x

        if phase == "sprint" and gap_to_face < plan.trigger_distance:
            phase, phase_steps = "dip", 0
        elif phase == "dip" and phase_steps >= plan.crouch_steps:
            phase, phase_steps = "launch", 0
        elif phase == "launch" and phase_steps >= plan.launch_steps:
            phase, phase_steps = "airborne", 0
        elif (phase == "airborne" and phase_steps > 6
              and float(env.data.qvel[2]) < -0.20):
            # Descending. Height thresholds cannot be used here: the deck the
            # ball is aiming at is a different height on every hop.
            phase, phase_steps = "landing", 0
        phase_steps += 1

        if phase == "sprint":
            targets = execute_skill("move_forward", env.data.qpos[3:7].copy(),
                                    env.dirs_body, env.max_extend,
                                    d_hat=FORWARD, back_gain=plan.approach_gain)
        else:
            targets = execute_skill("jump_forward_while_moving",
                                    env.data.qpos[3:7].copy(), env.dirs_body,
                                    env.max_extend, d_hat=FORWARD, phase=phase)
        env.step(targets)
        on_frame(label, f"{note} [{phase}]")
        n += 1

        # A hit is contact with the target box while NOT safely up on its deck,
        # i.e. clipping its front face or its edge on the way in.
        on_deck = (box["near"] < float(env.data.qpos[0]) < box["far"]
                   and float(env.data.qpos[2]) > box["height"] + JUMP_LAND_MARGIN)
        if touching_box() and not on_deck:
            box_contacts += 1

        # End shortly after touchdown. The landing phase keeps a rollout push
        # on, which would otherwise carry the ball to the far edge of the deck
        # and leave nothing to run up on for the next hop.
        if phase == "landing" and phase_steps > 22:
            break
        if box["role"] == "over" and float(env.data.qpos[0]) > box["far"] + 0.5:
            break

    return n, peak, box_contacts


def back_up_to(env, x_target, on_frame, label, note, max_steps=1500):
    """Roll backwards to x_target, to buy run-up room on a deck.

    After a landing the ball sits well into the deck. The next hop needs the
    trigger distance PLUS space to build speed, so it reverses to the deck's
    near end first -- the same thing a person does before a running jump.
    """
    n = 0
    while n < max_steps and float(env.data.qpos[0]) > x_target:
        env.step(execute_skill("reverse", env.data.qpos[3:7].copy(),
                               env.dirs_body, env.max_extend, d_hat=FORWARD))
        on_frame(label, note)
        n += 1
    n += hold_still(env, 40, on_frame, label, note)
    return n


def fall_off(env, from_height, on_frame, label, note, max_steps=2500):
    """Step off a deck and land on the floor."""
    deck_z = from_height + 0.19
    phase, n = "edge", 0
    while n < max_steps:
        z = float(env.data.qpos[2])
        if phase == "edge" and z < deck_z - 0.05:
            phase = "freefall"
        elif phase == "freefall" and z < 0.26:
            phase = "absorb"
        elif phase == "absorb" and z < 0.215:
            phase = "settle"
        env.step(execute_skill("fall_down", env.data.qpos[3:7].copy(),
                               env.dirs_body, env.max_extend,
                               d_hat=FORWARD, phase=phase))
        on_frame(label, f"{note} [{phase}]")
        n += 1
        if phase == "settle" and n > 40:
            break
    return n


def main():
    p = argparse.ArgumentParser(description="Run the platform course")
    p.add_argument("--config", default="configs/rl/skill_course.yaml")
    p.add_argument("--video", action="store_true")
    p.add_argument("--camera", default="fixed_angle_side_close")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--frame-every", type=int, default=4)
    p.add_argument("--margin", type=float, default=0.03)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.floor.square_m = 0.5
    scenario = generate_scenario("platform_course", cfg, seed=1)
    boxes = platform_course_boxes(cfg)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False,
                                max_steps=1_000_000)
    env.reset(seed=1)

    recorder = None
    if args.video:
        run_dir = make_run_dir(build_run_id("run_platforms", "platform_course"))
        out = Path(run_dir) / "renders" / "platform_course.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        recorder = VideoRecorder(out, fps=args.fps)

    total = {"n": 0}

    def on_frame(label, note):
        total["n"] += 1
        if recorder is None or total["n"] % args.frame_every:
            return
        v = env.data.qvel[0:2]
        recorder.add(annotate(
            env.render(camera_name=args.camera), label,
            [note,
             f"speed {float(np.linalg.norm(v)):5.2f} m/s",
             f"pos   {float(env.data.qpos[0]):6.2f} m   height {float(env.data.qpos[2]):5.2f} m",
             f"t {total['n'] * 0.01:5.1f}s"],
        ))

    # --- plan every jump up front, from the boxes' own geometry ---------
    print(f"platform course: {len(boxes)} boxes, goal at x={scenario.goal[0]:.1f}")
    plans, prev_deck = [], None
    for b in boxes:
        from_h = prev_deck["height"] if prev_deck else 0.0
        gap = 0.0 if prev_deck is None else max(0.0, b["near"] - prev_deck["far"])
        plan = plan_jump(b["height"], b["half_depth"], mode=b["role"],
                         from_height=from_h, gap=gap, margin=args.margin)
        plans.append((b, plan, from_h, gap))
        rise = b["height"] - from_h
        if plan is None:
            print(f"  box{b['index']} {b['role']:<5} rise {rise:+.2f} gap {gap:.2f}"
                  f"  -> NO PLAN, cannot make this jump")
        else:
            print(f"  box{b['index']} {b['role']:<5} rise {rise:+.2f} gap {gap:.2f}"
                  f"  -> {plan.describe()}")
        if b["role"] == "onto":
            prev_deck = b
    if any(pl is None for _b, pl, _f, _g in plans):
        print("\nrefusing to run: at least one jump is beyond the robot")
        env.close()
        return

    # --- execute -------------------------------------------------------
    results = []
    deck = 0.0
    for i, (b, plan, from_h, gap) in enumerate(plans, 1):
        rise = b["height"] - from_h
        kind = ("clear it" if b["role"] == "over" else
                "from the floor" if from_h == 0.0 else
                f"deck to deck, {rise:+.2f} m over a {gap:.2f} m gap")
        label = f"{i}. jump {b['role']} box{b['index']}"
        n, peak, contacts = do_jump(env, b, plan, on_frame, label, kind, deck)

        x, z = float(env.data.qpos[0]), float(env.data.qpos[2])
        if b["role"] == "over":
            ok = x > b["far"]
            deck = 0.0
        else:
            ok = b["near"] < x < b["far"] and z > b["height"] + JUMP_LAND_MARGIN
            if ok:
                deck = b["height"]
        results.append((b, ok, contacts, peak))
        print(f"  {label:<26} {n:>4} steps  peak {peak:.3f}  x {x:6.2f}  z {z:.3f}"
              f"  box-hits {contacts}  -> {'OK' if ok else 'FAILED'}")

        if b["role"] == "onto" and ok:
            hold_still(env, SETTLE_STEPS, on_frame, f"{i}b. stop",
                       "brake and balance on the deck")
            # Back up to the near end of this deck so the next hop has room to
            # build speed. Skip it for the last box, which is jumped off, not from.
            if i < len(plans):
                nxt = plans[i][0]
                nxt_plan = plans[i][1]
                want = nxt["near"] - nxt_plan.trigger_distance - RUNUP_M
                want = max(want, b["near"] + 0.30)
                if float(env.data.qpos[0]) > want + 0.10:
                    m = back_up_to(env, want, on_frame, f"{i}c. reverse",
                                   "back up for a run-up")
                    print(f"  {i}c. reverse{'':<15} {m:>4} steps  "
                          f"backed up to x {float(env.data.qpos[0]):.2f}")

    # --- come back down and finish -------------------------------------
    if deck > 0.0:
        n = fall_off(env, deck, on_frame, f"{len(plans)+1}. fall_down",
                     f"drop {deck:.2f} m back to the floor")
        print(f"  fall_down                  {n:>4} steps  z {float(env.data.qpos[2]):.3f}")
    n = drive_to(env, float(scenario.goal[0]) - 0.3, 2.0, on_frame,
                 f"{len(plans)+2}. move_forward", "run to the goal")
    hold_still(env, 60, on_frame, f"{len(plans)+3}. stop", "stopped at the goal")

    made = sum(1 for _b, ok, _c, _p in results if ok)
    hits = sum(c for _b, _ok, c, _p in results)
    print(f"\n{made}/{len(results)} jumps made, {hits} box contacts, "
          f"{total['n']} steps = {total['n'] * 0.01:.1f}s of real time, "
          f"final x {float(env.data.qpos[0]):.2f} (goal {scenario.goal[0]:.1f})")

    if recorder is not None:
        recorder.close()
        print(f"video: {recorder.path}  ({recorder.n_frames / args.fps:.1f}s)")
    env.close()


if __name__ == "__main__":
    main()
