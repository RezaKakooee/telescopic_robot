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

## RESOLVED: the fall skill pole-vault bug is fixed

**The fix:** During `freefall`, on short drops (`drop_height < 0.5 m`) gear deployment is restricted to the trailing-downward hemisphere (`(u_long < 0.0) & (u_z < -0.35)`). Because leading rods are held shut until `absorb`, they can never touch the upper deck early or pole-vault the ball forward. On tall drops (`drop_height >= 0.5 m`), the full downward hemisphere opens for maximum compliance.

**Verification:**
- Pillar 3 roll-off tested across 10 random orientation seeds: **10/10 landed squarely on pillar 3** at $x = 5.23\,\text{m}$ (target pad $[4.81, 5.71]\,\text{m}$).
- Added dedicated test `test_fall_down_pillar_rolloff()` to `tests/test_skills.py` (all 13 physics tests passing).


## Chimney: rebuilt on real physics (2026-08-28, late)

The inherited `run_chimney.py` pinned the ball's orientation every step and
never descended. Rebuilt: `chimney_climb` has `launch` / `push` / `fly` /
`hold` / `descend` phases; the driver's state machine reads position and
velocity. The ball wall-jumps up a 0.40 m shaft, bursts out over the LOWER
of the two walls (4.0 m / 3.3 m — a push cannot clear a lip level with the
wall it pushed off), lands on the box top and stops. 6/6 seeds; gentle mode
`--push-frac 0.7` 3/3; `--target 3.0` does hold-and-descend instead.

Measured and closed: there is **no smooth static climb** for radial
position-controlled rods (five inchworm variants, all creep down; see doc
§12.9). Do not spend time on it again without a different actuator.

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
