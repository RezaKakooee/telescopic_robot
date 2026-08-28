"""Command a single skill, or a demo sequence, and optionally record video.

Examples
--------
    # list every skill
    python scripts/skills/run_skill.py --list

    # drive forward for 200 steps and record a video
    python scripts/skills/run_skill.py --skill go_fast --steps 200 --video

    # run the full demo sequence through all 11 skills
    python scripts/skills/run_skill.py --demo --video

    # push against the nearest maze wall (wall found by lidar)
    python scripts/skills/run_skill.py --skill push_against_wall --kind maze --video
"""
from __future__ import annotations

import argparse
import math
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
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir
from skills import SKILL_NAMES
from skills.overlay import annotate
from skills.runner import (NEEDS_WALL_NORMAL, PHASE_SCHEDULES, default_steps,
                           run_skill)

# Sentinel used inside the wall-push cycle: roll back toward the wall so the
# next push has something to push on.
APPROACH_WALL = "__approach_wall__"

# Ordered walkthrough of every skill, as (name, steps, kwargs).
FORWARD = np.array([1.0, 0.0])
# Ground locomotion, run as ONE continuous take with no reset between steps.
# A skill only reads clearly next to its neighbours: "go slow" needs "go fast"
# beside it, and from a standstill "reverse" is indistinguishable from
# "move forward". Each entry is (skill, steps, label note).
COMBO_PROGRAM = [
    ("move_forward", 120, "baseline roll"),
    ("go_slow", 120, "same heading, less power"),
    ("go_fast", 140, "same heading, full power"),
    ("move_right", 120, "strafe while rolling"),
    ("move_left", 120, "strafe back the other way"),
    ("reverse", 150, "decelerate, then roll backward"),
    ("stop", 140, "brake to a standstill"),
]

DEMO_PROGRAM = [
    ("move_forward", 60, {"d_hat": FORWARD}),
    ("go_fast", 60, {"d_hat": FORWARD}),
    ("stop", 80, {}),
    ("go_slow", 60, {"d_hat": FORWARD}),
    ("move_right", 50, {"d_hat": FORWARD}),
    ("move_left", 50, {"d_hat": FORWARD}),
    ("reverse", 60, {"d_hat": FORWARD}),
    ("stop", 60, {}),
    ("jump_up", None, {}),
    ("jump_forward_while_stopped", None, {"d_hat": FORWARD}),
    ("jump_forward_while_moving", None, {"d_hat": FORWARD}),
]


def cycle_for(name, seconds, fps):
    """Repeat a skill until it fills *seconds* of video.

    A single skill run is only a few seconds long, so for a longer clip we
    repeat one meaningful cycle. What counts as a cycle depends on the skill:

    - ``stop`` only shows a brake if the ball is moving, so its cycle is
      accelerate-then-brake.
    - ``push_against_wall`` shoves the ball off the wall, so its cycle is
      push-then-roll-back-to-the-wall.
    - Jump skills repeat their own phase schedule.
    - Everything else just keeps driving.
    """
    if name == "stop":
        unit = [("go_fast", 55, {}), ("stop", 105, {})]
    elif name == "push_against_wall":
        unit = [("push_against_wall", 95, {}), (APPROACH_WALL, 65, {})]
    elif name in PHASE_SCHEDULES:
        # Brake to rest after each jump. Without this the next cycle relaunches
        # a ball that is still flying, and speed compounds every repeat.
        settle = {"jump_forward_while_moving": 60}.get(name, 45)
        unit = [(name, default_steps(name), {}), ("stop", settle, {})]
    else:
        unit = [(name, 120, {})]

    unit_steps = sum(steps for _, steps, _ in unit)
    # Round up: the clip must be at least `seconds` long, never short.
    reps = max(1, math.ceil(seconds * fps / unit_steps))
    return unit * reps, reps


def find_wall_normal(env):
    """Locate the nearest wall with the 16-ray lidar.

    Returns a unit vector pointing FROM the wall TOWARD the robot, which is
    what ``push_against_wall`` expects.
    """
    rays = env.raycast_lidar(n_rays=16, max_range=3.0, g=np.array([1.0, 0.0]))
    k = int(np.argmin(rays))
    angle = k / 16 * 2 * np.pi
    to_wall = np.array([np.cos(angle), np.sin(angle)])
    return -to_wall, float(rays[k]) * 3.0


def run_combo(env, args, renders_dir):
    """One continuous take of the ground locomotion skills, with on-screen labels.

    No reset between skills. Each one inherits the momentum the previous one
    left behind, which is the only way the differences read on screen.
    """
    recorder = None
    if renders_dir is not None:
        recorder = VideoRecorder(renders_dir / "combo_locomotion.mp4", fps=args.fps)

    # Every skill gets the same airtime, so they are directly comparable.
    per_skill = int(round(args.seconds * args.fps))
    total = per_skill * len(COMBO_PROGRAM)
    print(f"combo: {len(COMBO_PROGRAM)} skills x {args.seconds:.0f}s "
          f"= {total} steps = {total / args.fps:.1f}s")

    elapsed = 0
    start_all = env.data.qpos[0:3].copy()
    for idx, (name, _default_steps, note) in enumerate(COMBO_PROGRAM, 1):
        steps = per_skill
        seg_start = env.data.qpos[0:3].copy()

        def on_frame(e, step, _n=name, _i=idx, _s=steps, _note=note, _e0=elapsed):
            if recorder is None:
                return
            v = e.data.qvel[0:2]
            speed = float(np.linalg.norm(v))
            moved = float(e.data.qpos[0] - start_all[0])
            recorder.add(annotate(
                e.render(camera_name=args.camera),
                f"{_i}. {_n}",
                [
                    _note,
                    f"speed   {speed:5.2f} m/s",
                    f"vx {v[0]:+5.2f}   vy {v[1]:+5.2f} m/s",
                    f"x moved {moved:+6.2f} m",
                    f"t {(_e0 + step) / args.fps:5.1f}s / {total / args.fps:.0f}s",
                ],
            ))

        stats = run_skill(env, name, steps, d_hat=FORWARD, on_frame=on_frame)
        elapsed += steps
        seg = env.data.qpos[0:3] - seg_start
        print(f"  {idx}. {name:<14} {steps:>4} steps  "
              f"Δx={seg[0]:+.2f}m Δy={seg[1]:+.2f}m  "
              f"v_max={stats['max_speed']:.2f} v_end={stats['final_speed']:.2f} m/s")

    if recorder is not None:
        recorder.close()
        print(f"\nvideo: {recorder.path}  ({recorder.n_frames / args.fps:.1f}s)")


def report(stats):
    d = stats["displacement"]
    print(f"  {stats['skill']:<28} steps={stats['steps']:<4} "
          f"Δx={d[0]:+.3f}m Δy={d[1]:+.3f}m  "
          f"peak_z={stats['peak_z']:.3f}m (+{stats['net_lift'] * 100:.1f}cm)  "
          f"v_max={stats['max_speed']:.2f} v_end={stats['final_speed']:.2f} m/s")


def main():
    p = argparse.ArgumentParser(description="Run a radial-sphere skill")
    p.add_argument("--skill", choices=SKILL_NAMES, help="skill to run")
    p.add_argument("--demo", action="store_true", help="run all skills in sequence")
    p.add_argument("--combo", action="store_true",
                   help="one continuous labelled clip of the ground locomotion skills")
    p.add_argument("--list", action="store_true", help="list skills and exit")
    p.add_argument("--steps", type=int, default=None, help="override step budget")
    p.add_argument("--kind", default=None, help="scenario kind (default: from config)")
    p.add_argument("--config", default="configs/rl/standing_jump_showcase.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video", action="store_true", help="record an mp4")
    p.add_argument("--per-skill", action="store_true",
                   help="with --demo: write one video per skill instead of one long clip")
    p.add_argument("--seconds", type=float, default=None,
                   help="make each clip this long by repeating the skill's cycle")
    p.add_argument("--open-arena", action="store_true",
                   help="force the open goal arena (no rails, no hurdles)")
    p.add_argument("--no-reset-each", action="store_true",
                   help="with --demo: keep state across skills instead of resetting")
    p.add_argument("--camera", default="dual", help="camera name for the video")
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()

    if args.list:
        print("Available skills:")
        for i, name in enumerate(SKILL_NAMES, 1):
            print(f"  {i:2d}. {name}  (default steps: {default_steps(name)})")
        return

    if not args.skill and not args.demo and not args.combo:
        p.error("pass --skill NAME, or --demo, or --combo, or --list")

    cfg = load_config(args.config)
    kind = args.kind or getattr(cfg.scenario, "kind", "goal")

    # A long clip needs room. The default jump track has guide rails at
    # y = +-1.2 m and hurdles at x = 1.45 / 3.25 m, so a 30 s roll stalls
    # against them. Fall back to an open arena with the goal pushed out of
    # the way, unless the caller asked for a specific scenario.
    if args.combo and args.seconds is None:
        args.seconds = 30.0

    if args.open_arena or args.combo or (args.seconds and args.kind is None):
        kind = "goal"
        # Park the goal collider far off the ball's path. A 30 s go_fast run
        # covers ~40 m along +x, so a goal sitting in that lane would be hit.
        cfg.scenario.goal.x_range = [0.0, 0.0]
        cfg.scenario.goal.y_range = [-400.0, -400.0]
        print("open arena: kind=goal, goal parked 400 m off-path")

    scenario = generate_scenario(kind, cfg, seed=args.seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=args.seed)

    renders_dir = None
    if args.video:
        tag = "combo" if args.combo else ("demo" if args.demo else args.skill)
        run_dir = make_run_dir(build_run_id("run_skill", tag))
        renders_dir = Path(run_dir) / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)

    if args.combo:
        run_combo(env, args, renders_dir)
        env.close()
        return

    # One recorder for the whole run, unless --per-skill splits it up.
    whole = None
    if renders_dir is not None and not args.per_skill:
        whole = VideoRecorder(renders_dir / f"{tag}.mp4", fps=args.fps)

    program = DEMO_PROGRAM if args.demo else [(args.skill, args.steps, {})]

    print(f"scenario={kind}  camera={args.camera}  video={'yes' if args.video else 'no'}"
          + (f"  seconds={args.seconds}" if args.seconds else ""))
    written = []
    for i, (name, steps, kw) in enumerate(program, 1):
        base_kw = dict(kw)
        if not args.demo and args.steps is not None:
            steps = args.steps

        # Each skill starts from a clean state, so one skill's leftover
        # momentum cannot distort the next one's clip.
        if i > 1 and not args.no_reset_each:
            env.reset(seed=args.seed)

        clip = None
        if renders_dir is not None and args.per_skill:
            clip = VideoRecorder(renders_dir / f"{i:02d}_{name}.mp4", fps=args.fps)

        target = clip or whole
        on_frame = None
        if target is not None:
            def on_frame(e, step, _rec=target):
                _rec.add(e.render(camera_name=args.camera))

        # A long clip repeats one meaningful cycle; a short one runs once.
        if args.seconds:
            sub_program, reps = cycle_for(name, args.seconds, args.fps)
            print(f"  {name}: {reps} cycle(s) to fill {args.seconds}s")
        else:
            sub_program = [(name, steps, {})]

        start = env.data.qpos[0:3].copy()
        peak_z, max_speed, total_steps = float(start[2]), 0.0, 0
        for sub_name, sub_steps, sub_kw in sub_program:
            call = dict(base_kw)
            call.update(sub_kw)
            run_name = sub_name

            # Roll back toward the wall so the next push has a surface.
            if sub_name == APPROACH_WALL:
                run_name = "move_forward"
                wall_normal, _ = find_wall_normal(env)
                call["d_hat"] = -wall_normal
                call.pop("wall_normal", None)
            elif run_name in NEEDS_WALL_NORMAL:
                # Re-locate the wall each cycle; the ball has moved since the last one.
                wall_normal, dist = find_wall_normal(env)
                call["wall_normal"] = wall_normal
                print(f"    wall at {dist:.3f} m")

            call.setdefault("d_hat", FORWARD)
            if run_name not in NEEDS_WALL_NORMAL:
                call.pop("wall_normal", None)

            s = run_skill(env, run_name, sub_steps, on_frame=on_frame, **call)
            peak_z = max(peak_z, s["peak_z"])
            max_speed = max(max_speed, s["max_speed"])
            total_steps += s["steps"]

        end = env.data.qpos[0:3].copy()
        report({
            "skill": name, "steps": total_steps, "displacement": end - start,
            "peak_z": peak_z, "net_lift": peak_z - float(start[2]),
            "max_speed": max_speed,
            "final_speed": float(np.linalg.norm(env.data.qvel[0:2])),
        })

        if clip is not None:
            clip.close()
            written.append(f"{clip.path}  ({clip.n_frames / args.fps:.1f}s)")

    if whole is not None:
        whole.close()
        written.append(f"{whole.path}  ({whole.n_frames / args.fps:.1f}s)")
    if written:
        print("\nvideos:")
        for path in written:
            print(f"  {path}")
    env.close()


if __name__ == "__main__":
    main()
