# Handoff — Skill library, planners, and the pillar course

Date: 2026-08-28. Supersedes the 2026-08-14 maze-RL handoff (that work is
recorded in `docs/project_journey/01_*.md`; the checkpoints it names still
exist under `storage_local/`).

Read `docs/project_journey/02_skill_library_and_the_skill_course.md` for the
full story with numbers. This file is the short version: what exists, how to
run it, what is open.

## What exists

A library of **parametric motion skills** for the 60-rod ball, a **planner**
that computes jump parameters from obstacle geometry, and three courses that
prove them.

| Piece | Where | One line |
|---|---|---|
| Skills | `skills/` | Pure functions: state in, 60 rod targets out. `skills/README.md` is the API. |
| The gait | `skills/locomotion.py: move(turn, speed)` | Radians and m/s. Every named locomotion skill is a preset of it. |
| Aimed hop | `skills/jumping.py: jump_to(vx_target, vz_target)` | Standing jump, velocity-servoed during the burn. |
| Hop planner | `skills/hop_planner.py` | Pad geometry → stand point + velocity command, or `None`. |
| Run-jump planner | `skills/jump_planner.py` | Box geometry → run-up gain, crouch, trigger, or `None`. |
| Calibrations | `skills/hop_calibration.json`, `skills/jump_calibration.json` | Measured over random orientations. Regenerate if the robot changes. |
| Courses | `radial_sphere/scenario.py` | `skill_course`, `platform_course`, `pillar_course`. |
| Drivers | `scripts/skills/run_course.py`, `run_platforms.py`, `run_pillars.py` | Each prints per-hop results and can record video. |
| Tests | `tests/test_skills.py` | 12 physics assertions. All pass as of this commit. |

## How to run

```bash
conda activate roboverse
export MUJOCO_GL=egl PYTHONPATH=.

python tests/test_skills.py                                  # ~10 min
python scripts/skills/run_pillars.py --seed 2 --video        # pillar ladder
python scripts/skills/run_platforms.py --video               # platform course
python scripts/skills/run_course.py --video                  # sketched circuit
python scripts/skills/run_parametric_demo.py --video         # move(turn, speed)
python scripts/skills/calibrate_hop.py                       # rebuild hop table (~15 min)
```

Videos land in `storage_local/<run id>/renders/`. One env step is 0.01 s of
simulated time; the drivers record every 4th step at 25 fps = real time.

EGL prints a harmless `EGLError ... EGL_NOT_INITIALIZED` traceback at exit.
Pipe through `2>/dev/null` or ignore it.

## Results as of this commit

| Course | Result |
|---|---|
| Skill course (sketched circuit) | goal reached, both jumps clean |
| Platform course (5 boxes) | 5/5 jumps |
| Pillar course, 14 random orientations, stand-off 0.24 m | 14/14 |
| Pillar course, 6 random orientations, stand-off 0.45 m | 6/6 |

## OPEN: the fall skill is not what was asked

The request: while falling, the ball should **open its rods underneath**
(land on rods, not on its shell) and put **some front rods out when a wall
is ahead**.

What is in the code: `fall_down(gear=0.5, brace_front=0.0)` does exactly
that in free fall — but only when `drop_height >= 0.5`. On shorter drops the
gear waits until touchdown.

Why: on the 0.40 m roll-off between pillars, opening the gear early was
harmful. The ball still has forward motion as it leaves the lip; rods already
extended reach the lower deck first and pole-vault it over the pad. Four of
six seeds overshot. The 0.5 m rule was a stop-gap, not the answer.

What the next person should try, in order:

1. Trigger the gear on **clearing the lip**, not on drop height. Open the
   rods once the ball's trailing edge is past the launch edge (x known from
   the pad geometry), so nothing can catch the deck while still leaving.
2. Failing that, open the gear **behind and below only** (rods with
   `u_long < 0` and `u_z < -0.35`) so the leading rods cannot vault.
3. Verify on `run_pillars.py --seed N` for N in 1..6: the roll-off must land
   on pillar 3 every time. Then add a `tests/test_skills.py` assertion.

`brace_front` is implemented and wired (`run_pillars.py` sets it when a
taller face is within 0.8 m of the landing) but no course exercises it yet.

## Other open items

- **Probe-abort rate.** With `PROBE_TRUST = 1.0` the planner assumes the
  mean take-off; about a third of launches are aborted and re-dealt. Lower
  the constant to stand closer with fewer aborts.
- **Unrecoverable misses.** A ball that lands hung on a lip, or on the floor
  between pillars where it cannot line up, ends the run. The side-lane
  recovery only handles the floor beside the pillars.
- **Run-jump calibration is stale-prone.** `jump_calibration.json` was
  measured on the standard 0.16 m stroke; `pillar_course.yaml` uses 0.26 m.
  The pillar course uses `hop_calibration.json` (measured at 0.26 m), so this
  is fine today, but do not mix them.
- **RL.** This was the goal of making the skills parametric. The action
  space is ready: `move(turn, speed)`, `stop(stop_distance)`,
  `jump_to(vx, vz)`, `fall_down(drop_height)`. Nothing has been trained on
  it yet.

## Things that will bite you

- **Calibration jitter must rotate the ball.** Extra settle steps do not.
  Randomize the quaternion (`calibrate_hop.py` does).
- **Lift-off is the vz peak.** Not the phase switch.
- **A leading rod on a rolling ball is a brake.** Used on purpose in
  `stop`; it wrecked the running jump until masked (§8 of the doc).
- **The ball pins itself on any wall within 0.41 m** (its rod reach). That
  is why the pillar corridor is 2.8 m wide.
- **Model extent drives camera clipping.** The MJCF pins
  `<statistic extent="4">`; without it a big floor deletes the robot from
  close-up cameras.
- **Cameras inside walls render nothing.** `pillar_side` sits at 1.3 m for
  that reason.
- **Two renders in one minute share a run dir** unless the tag differs.
  `run_pillars.py` puts the seed in the tag.
