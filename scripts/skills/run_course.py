"""Drive the hand-drawn skill course using the skill library.

The course is the sketched circuit: start corridor, climb, long top straight,
right-hand descent, a dead-end spur entered forwards and left in reverse,
the bottom corridor with a pause, then down to the goal.

Each leg is driven by a named skill, so the video shows the skills doing real
navigation rather than rolling on an empty floor.

    python scripts/skills/run_course.py --video
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
from radial_sphere.scenario import (generate_scenario, skill_course_hurdle,
                                    skill_course_platform, skill_course_route)
from radial_sphere.snapshot import make_run_dir
from skills.jump_planner import plan_jump
from skills.overlay import annotate
from skills.runner import skill_targets

# Sub-legs of the route, as (route index from, route index to, split fraction,
# skill, note). `split` cuts a leg in two so one straight can change skill
# partway, matching the sketch's "move normal then fast".
#
# Route indices (from skill_course_route):
#   0 start   1 corner   2 top-left   3 top-right   4 spur junction
#   5 spur end   6 junction again   7 bottom-right   8 bottom-left
#   9 goal corridor   10 goal
PLAN = [
    # (from, to, fraction, skill, note, obstacle, hold_steps)
    # `obstacle` names which box the phase triggers fire against.
    (0, 1, 0.18, "move_forward", "start corridor, platform ahead", None, 0),
    # Runs past the platform's far edge (fraction 0.60) so the leg ends when
    # the ball is actually down on the deck, not when it first reaches the
    # centre waypoint still in mid-air.
    (0, 1, 0.60, "jump_forward_while_moving", "leap up ONTO the platform", "platform", 0),
    (0, 1, 0.60, "stop", "steady itself on top of the platform", None, 170),
    (0, 1, 0.76, "fall_down", "step off the edge and drop back down", "platform", 0),
    (0, 1, 1.0, "move_forward", "carry on along the start corridor", None, 0),
    (1, 2, 1.0, "move_forward", "climb to the top", None, 0),
    (2, 3, 0.45, "move_forward", "top straight, normal", None, 0),
    (2, 3, 1.0, "go_fast", "top straight, full power", None, 0),
    (3, 4, 1.0, "move_forward", "turn down the right side", None, 0),
    (4, 5, 1.0, "move_forward", "drive into the dead-end spur", None, 0),
    (5, 6, 1.0, "reverse", "back out of the spur in reverse", None, 0),
    (6, 7, 1.0, "move_forward", "right-hand descent", None, 0),
    (7, 8, 0.34, "move_forward", "bottom corridor, box ahead", None, 0),
    (7, 8, 0.34, "stop", "pause for a moment before the box", None, 90),
    (7, 8, 0.62, "jump_forward_while_moving", "sprint and leap OVER the box", "hurdle", 0),
    (7, 8, 1.0, "move_forward", "bottom corridor again", None, 0),
    (8, 9, 1.0, "move_forward", "final descent", None, 0),
    (9, 10, 1.0, "go_slow", "ease up to the goal", None, 0),
    (10, 10, 1.0, "stop", "stopped at the goal", None, 90),
]

CORNER_GAIN = 1.15   # push wave while taking a 90 degree corner
CRUISE_GAIN = 2.0    # move_forward default
HOLD_STEPS = 90      # steps spent on a "stop" leg

# Running jump over the box. The phases are triggered by distance to the box,
# not by a step count: the launch has to happen at the right place, and the
# run-up speed varies, so a fixed schedule would take off early or late.
# Both the crouch and the launch are triggered by DISTANCE to the box, not by
# a step count. A fixed-length dip fires the launch at whatever point the
# rolling gait happens to be in, so takeoff height swings between 0.17 m and
# 0.30 m and the leap either clears the box or climbs it. Locking the launch
# to a distance makes takeoff happen at the same place every run.
# The box is only 0.30 m deep, so the leap needs height far more than reach.
# Charging it at full sprint spends the impulse on forward speed and the ball
# skims the lid; approaching at a moderate speed leaves the rods time to load
# and turns the launch into a proper vertical jump.
# Measured limits of this robot's running leap, on flat ground with the box
# removed: it peaks at 0.577 m and carries 0.40 m while airborne. A full
# sprint approach is worse, not better -- the crouch cannot seat the rods at
# 3 m/s, so the take-off is weak and the ball drops onto the obstacle.
# Per-obstacle tuning. Clearing a low box and landing on a tall platform are
# different problems: the platform needs a slower run-up so the leap goes up
# rather than along, and so the ball can brake on the short deck.
JUMP_TUNING = {
    "hurdle":   {"gain": 2.5, "dip_at": 0.90},
    # 1.9 / 0.60 is the one combination that lands on the platform with ZERO
    # contact against its front face. It is a narrow window: nearby values
    # either clip the face on the way up or fall short and shove into it.
    "platform": {"gain": 1.9, "dip_at": 0.60},
}
JUMP_APPROACH_GAIN = 2.0   # moderate run-up, about 1.2 m/s
JUMP_DIP_AT = 0.55      # start the crouch this far from the near face (m)
JUMP_DIP_STEPS = 24     # hold the crouch; shorter and the rods never seat
JUMP_LAUNCH_STEPS = 16  # hold the extension; the ball leaves ground at its end
JUMP_LAND_Z = 0.28      # below this core height the ball is back down


def leg_waypoints(route, i_from, i_to, frac_start, frac_end, spacing=0.25):
    """Dense points along one straight sub-leg, between two fractions of it."""
    a, b = np.asarray(route[i_from], float), np.asarray(route[i_to], float)
    p0, p1 = a + frac_start * (b - a), a + frac_end * (b - a)
    n = max(2, int(np.linalg.norm(p1 - p0) / spacing))
    return [p0 + t * (p1 - p0) for t in np.linspace(0.0, 1.0, n)]


def build_plan(route):
    """Expand PLAN into (skill, note, waypoints, obstacle, hold) sub-legs."""
    legs, seen = [], {}
    for i_from, i_to, frac, skill, note, obstacle, hold in PLAN:
        if i_from == i_to:                       # a pure hold, no travel
            legs.append((skill, note, [np.asarray(route[i_to], float)], obstacle, hold))
            continue
        start = seen.get((i_from, i_to), 0.0)
        pts = leg_waypoints(route, i_from, i_to, start, frac)
        legs.append((skill, note, pts, obstacle, hold))
        seen[(i_from, i_to)] = frac
    return legs


def main():
    p = argparse.ArgumentParser(description="Navigate the skill course")
    p.add_argument("--config", default="configs/rl/skill_course.yaml")
    p.add_argument("--video", action="store_true")
    p.add_argument("--camera", default="course_dual")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--frame-every", type=int, default=4,
                   help="record one frame every N env steps. One env step is\n0.01 s of simulated time, so --frame-every 4 at --fps 25 plays back in real time")
    p.add_argument("--lookahead", type=float, default=0.85)
    p.add_argument("--reach", type=float, default=0.45, help="waypoint capture radius (m)")
    p.add_argument("--max-steps-per-leg", type=int, default=4000)
    p.add_argument("--debug-jump", action="store_true")
    p.add_argument("--max-legs", type=int, default=None)
    p.add_argument("--dip-at", type=float, default=None)
    p.add_argument("--box-height", type=float, default=None)
    p.add_argument("--box-depth", type=float, default=None)
    p.add_argument("--plat-height", type=float, default=None)
    p.add_argument("--plat-depth", type=float, default=None)
    p.add_argument("--plat-gain", type=float, default=None)
    p.add_argument("--plat-dip-at", type=float, default=None)
    p.add_argument("--box-gain", type=float, default=None)
    p.add_argument("--box-dip-at", type=float, default=None)
    p.add_argument("--launch-steps", type=int, default=None)
    p.add_argument("--dip-steps", type=int, default=None)
    p.add_argument("--approach-gain", type=float, default=None)
    p.add_argument("--wall-height", type=float, default=0.55,
                   help="0.22 m walls vanish on a 34 m course seen from above")
    p.add_argument("--floor-square", type=float, default=0.9,
                   help="finer squares turn the map view into visual noise")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.scenario.maze.wall_height = args.wall_height
    cfg.floor.square_m = args.floor_square
    if args.box_height is not None:
        cfg.scenario.skill_course.hurdle_height = args.box_height
    if args.box_depth is not None:
        cfg.scenario.skill_course.hurdle_depth = args.box_depth
    if args.plat_height is not None:
        cfg.scenario.skill_course.platform_height = args.plat_height
    if args.plat_depth is not None:
        cfg.scenario.skill_course.platform_depth = args.plat_depth
    if args.plat_gain is not None:
        JUMP_TUNING['platform']['gain'] = args.plat_gain
    if args.plat_dip_at is not None:
        JUMP_TUNING['platform']['dip_at'] = args.plat_dip_at
    if args.box_gain is not None:
        JUMP_TUNING['hurdle']['gain'] = args.box_gain
    if args.box_dip_at is not None:
        JUMP_TUNING['hurdle']['dip_at'] = args.box_dip_at
    scenario = generate_scenario("skill_course", cfg, seed=1)
    route = skill_course_route(cfg)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=1_000_000)
    env.reset(seed=1)

    recorder = None
    if args.video:
        run_dir = make_run_dir(build_run_id("run_course", "skill_course"))
        out = Path(run_dir) / "renders" / "skill_course.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        recorder = VideoRecorder(out, fps=args.fps)

    global JUMP_DIP_AT
    if args.dip_at is not None: JUMP_DIP_AT = args.dip_at
    global JUMP_LAUNCH_STEPS
    if args.launch_steps is not None: JUMP_LAUNCH_STEPS = args.launch_steps
    global JUMP_APPROACH_GAIN
    if args.approach_gain is not None: JUMP_APPROACH_GAIN = args.approach_gain
    global JUMP_DIP_STEPS
    if args.dip_steps is not None: JUMP_DIP_STEPS = args.dip_steps
    legs = build_plan(route)
    if args.max_legs:
        legs = legs[:args.max_legs]
    print(f"course: {len(legs)} legs, route {scenario.path_length:.1f} m, "
          f"goal at {scenario.goal}")

    total_steps = 0
    wall_hits = 0
    reached_goal = False

    # Geoms making up the box, so we can prove the ball flew over instead of
    # shoving through it.
    import mujoco
    box_geom_ids = {
        i for i in range(env.model.ngeom)
        if (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("wood_")
    }

    def box_touched():
        for c in range(env.data.ncon):
            con = env.data.contact[c]
            if con.geom1 in box_geom_ids or con.geom2 in box_geom_ids:
                return True
        return False

    OBSTACLES = {
        "hurdle": skill_course_hurdle(cfg),
        "platform": skill_course_platform(cfg),
    }

    # Ask the planner what each obstacle needs, instead of carrying tuned
    # constants. It works from the measured take-off calibration, so a plan is
    # a guarantee against the worst point of the gait cycle -- and no plan
    # means the robot genuinely cannot make that jump.
    PLANS = {}
    for tag, obs in OBSTACLES.items():
        mode = "onto" if tag == "platform" else "over"
        plan = plan_jump(obs["height"], obs["half_depth"], mode=mode)
        PLANS[tag] = plan
        if plan is None:
            print(f"  PLANNER: cannot jump {mode} the {tag} "
                  f"(h={obs['height']:.2f} m, half-depth={obs['half_depth']:.2f} m)"
                  f" -- beyond this robot's guaranteed take-off")
        else:
            print(f"  PLANNER {tag} ({mode}): {plan.describe()}")

    for leg_no, (skill, note, waypoints, obstacle, hold) in enumerate(legs, 1):
        obs = OBSTACLES.get(obstacle)
        box_xy = None if obs is None else np.asarray(obs["xy"], dtype=float)
        box_face = 0.0 if obs is None else obs["half_depth"]
        box_h = 0.0 if obs is None else obs["height"]
        plan = PLANS.get(obstacle)
        tune = JUMP_TUNING.get(obstacle, {})
        if plan is not None:
            approach_gain = plan.approach_gain
            dip_at = plan.trigger_distance
            dip_steps = plan.crouch_steps
        else:
            approach_gain = tune.get("gain", JUMP_APPROACH_GAIN)
            dip_at = tune.get("dip_at", JUMP_DIP_AT)
            dip_steps = JUMP_DIP_STEPS
        hold_steps = hold or HOLD_STEPS
        target_i = 0
        steps = 0
        seg_start = env.data.qpos[0:2].copy()
        jump_phase, phase_steps, jump_peak, cleared = "sprint", 0, 0.0, False
        landed_on = False
        on_deck_steps = 0
        face_hits = 0
        _prev_jump_phase = "sprint"
        over_box_min_z = None      # lowest core height while above the box
        z_at_near_face = None      # core height crossing the near edge
        box_touch_steps = 0

        while steps < args.max_steps_per_leg:
            ball = env.data.qpos[0:2].copy()

            if skill == "stop":
                if steps >= hold_steps:
                    break
                heading = np.array([1.0, 0.0])
            else:
                # Pure pursuit: walk the index forward to the first waypoint
                # that is at least `lookahead` away, then steer at it.
                while (target_i < len(waypoints) - 1
                       and np.linalg.norm(waypoints[target_i] - ball) < args.lookahead):
                    target_i += 1
                aim = waypoints[target_i] - ball
                n = float(np.linalg.norm(aim))
                if target_i >= len(waypoints) - 1 and n < args.reach:
                    break
                heading = aim / n if n > 1e-6 else np.array([1.0, 0.0])

            # `reverse` drives opposite to d_hat, so flip the heading for it.
            d_hat = -heading if skill == "reverse" else heading

            kw = {}
            run_skill_name = skill
            if skill in ("move_forward", "go_fast"):
                # Ease off into a turn: if the direction to the waypoint after
                # next differs sharply, drop to the corner gain.
                nxt = min(target_i + 4, len(waypoints) - 1)
                ahead = waypoints[nxt] - waypoints[target_i]
                if np.linalg.norm(ahead) > 1e-6:
                    ahead = ahead / np.linalg.norm(ahead)
                    turn = float(np.dot(ahead, heading))
                    if turn < 0.80:
                        run_skill_name, kw = "move_forward", {"back_gain": CORNER_GAIN}

            if skill == "fall_down":
                # Height-driven phases: creep to the lip, tuck while falling,
                # spread the landing gear, then settle. Step counts cannot
                # work here because how long the drop takes depends on how
                # long the creep to the edge takes.
                core_z = float(env.data.qpos[2])
                deck = box_h + 0.19          # core height while stood on top
                if jump_phase == "sprint":
                    jump_phase = "edge"
                if jump_phase == "edge" and core_z < deck - 0.05:
                    jump_phase, phase_steps = "freefall", 0
                elif jump_phase == "freefall" and core_z < 0.26:
                    jump_phase, phase_steps = "absorb", 0
                elif jump_phase == "absorb" and core_z < 0.215:
                    jump_phase, phase_steps = "settle", 0
                phase_steps += 1
                if jump_phase != _prev_jump_phase:
                    if args.debug_jump:
                        print(f"      {_prev_jump_phase:>8} -> {jump_phase:<8} "
                              f"x={ball[0]:6.2f} z={core_z:.3f} "
                              f"vz={env.data.qvel[2]:+5.2f}")
                    _prev_jump_phase = jump_phase
                kw["phase"] = jump_phase
                # "settle" holds a stance and does not drive, so pure pursuit
                # would never reach the leg's last waypoint. End on landing.
                if jump_phase == "settle" and phase_steps > 45:
                    break

            if skill == "jump_forward_while_moving":
                # Signed position along the leg, measured from the box centre.
                # Negative while approaching, positive once past it.
                leg_dir = np.asarray(waypoints[-1], float) - np.asarray(waypoints[0], float)
                leg_dir = leg_dir / max(float(np.linalg.norm(leg_dir)), 1e-6)
                along = float(np.dot(ball - box_xy, leg_dir))
                gap = -along - box_face          # distance to the near face
                core_z = float(env.data.qpos[2])
                jump_peak = max(jump_peak, core_z)

                if jump_phase == "sprint" and gap < dip_at:
                    jump_phase, phase_steps = "dip", 0
                elif jump_phase == "dip" and phase_steps >= dip_steps:
                    jump_phase, phase_steps = "launch", 0
                elif jump_phase == "launch" and phase_steps >= JUMP_LAUNCH_STEPS:
                    jump_phase, phase_steps = "airborne", 0
                elif jump_phase == "airborne" and core_z < JUMP_LAND_Z and phase_steps > 5:
                    jump_phase, phase_steps = "landing", 0
                if jump_phase != _prev_jump_phase:
                    if args.debug_jump:
                        print(f"      {_prev_jump_phase:>8} -> {jump_phase:<8} "
                              f"x={ball[0]:6.2f} gap={gap:+5.2f} z={core_z:.3f} "
                              f"vx={env.data.qvel[0]:+5.2f} vz={env.data.qvel[2]:+5.2f}")
                    _prev_jump_phase = jump_phase
                phase_steps += 1
                if jump_phase == "sprint":
                    run_skill_name = "move_forward"
                    kw = {"back_gain": approach_gain}
                else:
                    kw["phase"] = jump_phase
                if abs(along) <= box_face:
                    over_box_min_z = (core_z if over_box_min_z is None
                                      else min(over_box_min_z, core_z))
                    if z_at_near_face is None:
                        z_at_near_face = core_z
                if not cleared and along > box_face:
                    cleared = True
                # Contact with the box while NOT standing on its deck means the
                # ball hit the front face. That is the thing to drive to zero.
                on_deck_now = (abs(along) <= box_face and core_z > box_h + 0.10)
                if box_touched() and not on_deck_now:
                    face_hits += 1
                # Landing ON a platform ends the leg: the waypoints run to the
                # platform centre, but the ball stops wherever it touches down,
                # so pure pursuit would never reach the last one.
                if obstacle == "platform":
                    if (abs(along) <= box_face and core_z > box_h + 0.10
                            and float(np.linalg.norm(env.data.qvel[0:2])) < 0.40):
                        on_deck_steps += 1
                        if on_deck_steps > 40:
                            break
                    else:
                        on_deck_steps = 0
                if box_touched():
                    box_touch_steps += 1

            if skill == "stop":
                kw = {"lin_vel": env.data.qvel[0:2].copy()}

            targets = skill_targets(env, run_skill_name, steps, d_hat=d_hat, **kw)
            _, _, _, _, info = env.step(targets)
            if info.get("wall_contact"):
                wall_hits += 1
            if info.get("goal_contact") or info.get("success"):
                reached_goal = True

            if recorder is not None and total_steps % args.frame_every == 0:
                v = env.data.qvel[0:2]
                recorder.add(annotate(
                    env.render(camera_name=args.camera),
                    f"{leg_no}. {skill}",
                    [
                        note,
                        f"speed  {float(np.linalg.norm(v)):5.2f} m/s",
                        f"pos    {ball[0]:6.1f}, {ball[1]:6.1f} m",
                        f"t {total_steps * 0.01:5.1f}s   wall hits {wall_hits}",
                    ],
                ))
            steps += 1
            total_steps += 1

        if skill == "fall_down":
            z = float(env.data.qpos[2])
            print(f"      fall: ended at z {z:.3f} m, "
                  f"{'down off the platform' if z < box_h + 0.12 else 'STILL UP'}")

        if skill == "jump_forward_while_moving" and obstacle == "platform":
            # On the deck = inside the platform footprint and riding high.
            z = float(env.data.qpos[2])
            xy = env.data.qpos[0:2].copy()
            leg_dir = np.asarray(waypoints[-1], float) - np.asarray(waypoints[0], float)
            leg_dir = leg_dir / max(float(np.linalg.norm(leg_dir)), 1e-6)
            along = float(np.dot(xy - box_xy, leg_dir))
            landed_on = abs(along) <= box_face and z > box_h + 0.10
            print(f"      jump-on: peak z {jump_peak:.3f}  landed z {z:.3f}  "
                  f"face-hits {face_hits}  "
                  f"-> {'ON THE PLATFORM' if landed_on else 'MISSED'}"
                  f"{', CLEAN TAKEOFF' if landed_on and face_hits == 0 else ''}")
        elif skill == "jump_forward_while_moving":
            clean = cleared and box_touch_steps == 0
            # Underside of the tucked ball is roughly core radius + tucked rod.
            BALL_UNDER = 0.165
            gap_over = (None if over_box_min_z is None
                        else over_box_min_z - BALL_UNDER - box_h)
            print(f"      jump: peak z {jump_peak:.3f}  "
                  f"z@near-edge {z_at_near_face if z_at_near_face is None else round(z_at_near_face,3)}  "
                  f"min-z-over-box {over_box_min_z if over_box_min_z is None else round(over_box_min_z,3)}  "
                  f"clearance {'n/a' if gap_over is None else format(gap_over, '+.3f')} m  "
                  f"touch {box_touch_steps}  "
                  f"-> {'CLEAN' if clean else ('scraped' if cleared else 'FAILED')}")

        moved = env.data.qpos[0:2] - seg_start
        print(f"  {leg_no:2d}. {skill:<13} {note:<32} "
              f"{steps:>4} steps  Δ({moved[0]:+5.1f},{moved[1]:+5.1f}) m  "
              f"end ({env.data.qpos[0]:5.1f},{env.data.qpos[1]:6.1f})")

    goal_dist = float(np.linalg.norm(env.data.qpos[0:2] - scenario.goal))
    print(f"\ntotal {total_steps} steps = {total_steps * 0.01:.1f}s of real time   "
          f"wall-contact steps {wall_hits}   "
          f"final distance to goal {goal_dist:.2f} m   "
          f"goal touched: {reached_goal}")

    if recorder is not None:
        recorder.close()
        print(f"video: {recorder.path}  ({recorder.n_frames / args.fps:.1f}s of playback)")
    env.close()


if __name__ == "__main__":
    main()
