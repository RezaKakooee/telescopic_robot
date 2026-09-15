# Handoff — current state

Date: 2026-09-14. Supersedes earlier handoffs.
The VLA work (skills for a policy, the playground course, the ten inspection
courses, the generic expert, demo collection for SFT) is in `HANDOFF_VLA_RL.md`.

A 60-rod spherical robot in MuJoCo. It moves by extending and retracting
telescopic rods; there are no wheels and no legs. On top of that sits a
library of motion skills, a set of demos that prove them, and an RL stack.

## Layout

| Where | What |
|---|---|
| `radial_sphere/` | The robot and the runtime: MuJoCo env, MJCF builder, scenarios, the shared gait maths, the demo runner. Also `inspection_scenarios.py` (ten courses), `inspection_oracle.py` (generic expert), `playground_course.py` (course dimensions). |
| `skills/low_level/` | 10 modules. State in, 60 rod targets out. One behaviour each, no branching. `pipe_crawling.py` is the newest. |
| `skills_vla/` | The policy-facing skill layer: six classes with bounded params, each a thin wrapper over `skills`. |
| `skills/mid_level/` | 4 modules. Choose a low-level skill each step and delegate: `follow_path`, `stay_in_boundary`, `climb_stairs`, plus the jump planners. |
| `skills/high_level/` | 1 module. `go_to_goal` plans a route and answers with a skill name plus arguments, never rod targets. |
| `demos/<name>/` | 17 folders. `demo.yaml` plus `runner.py` when the control flow is the point. |
| `configs/rl/` | 45 scenario presets: arena, floor, robot, sim2real. |
| `configs/run/` | 16 knob files, one per entry script, grouped like `scripts/`. |
| `scripts/` | Entry points: RL training, imitation, calibration, `run_demo.py`, `run_tests.py`. |
| `storage_local/` | All run output. Gitignored. |

44 registry names reach 27 skill functions; the extras are aliases.
`skills/README.md` is the API reference, `demos/README.md` the demo spec.

## Running things

No command-line flags anywhere. Every entry script takes `key=value`
overrides, and `--help` prints its full knob list with current values.

```bash
PYTHONPATH=. python scripts/run_tests.py              # 190 tests, ~60 s
PYTHONPATH=. python scripts/run_tests.py all=true     # plus the ~10 min drivers

PYTHONPATH=. python scripts/run_demo.py list=true     # what demos exist
PYTHONPATH=. python scripts/run_demo.py demo=gap
PYTHONPATH=. python scripts/run_demo.py demo=all video=false

python scripts/rl/train_rl.py kind=maze rl.n_envs=8   # script knob + scenario override
```

Environment: the `roboverse` conda env, `MUJOCO_GL=egl`, run from the repo
root with `PYTHONPATH=.`.

Blog videos regenerate into a scratch directory rather than over the
published ones:

```bash
BLOG_ASSETS_DIR=$PWD/regen_check python docs/blog/render_wall_push.py
```

## State

`main` is at `ab6a22e` (the VLA day). The working tree holds the SmolVLA
evaluator and round-2 tooling (mine, see `HANDOFF_VLA_RL.md` section 10) plus
files from another session (`wall_jump`, `shaft_climbing`, chimney demo)
that were not reviewed here. 190 tests passed before those files appeared.

## Just done (2026-09-14, VLA day)

Short list; details in `HANDOFF_VLA_RL.md`.

- **The macro running jump was an accident.** Its sprint phase kicked a
  rolling ball into the air with the rods out. Fixed in
  `radial_sphere/handcrafted_skill_backend.py`; the option now also waits
  for the ball to settle before it ends.
- **Obstacle hits are measured.** `info["obstacle_hit"]` in the env, a
  reward penalty, and "clean success" in the evaluations.
- **`crawl_pipe`.** Rolling inside a round conduit down to 0.22 m radius.
- **Playground course** parametrised in `radial_sphere/playground_course.py`
  and pushed to the jump's limits. The conduit became a wall.
- **Ten inspection courses** with short and tour routes, a generic expert
  that reads the obstacles from the scenario, and a demo collector.
  216 tour demos (108k frames) are in
  `storage_local/20260914_1612__local__generate_inspection_demos__30ep/`.
- **Three builder bugs.** Stone fields were never built into the MJCF;
  staircases ignored `yaw`; the pipe builder ignored `yaw`.
- **Every experiment folder has a timestamp** now; old folders were renamed
  by their mtime.
- **SmolVLA fine-tuned on 216 tour demos.** Skill agreement with the expert
  96 %, but only 5/10 courses reached on the short routes: the policy misses
  the moment of the rare skills. Round 2 (433 demos, rare windows
  oversampled 4x) reaches 6/10 with 98 % agreement. Three jump-drill
  courses (hurdle lane, trench field, box steps) were added for jump data;
  the obstacle-hit detector was missing the inspection object names, so all
  hit counts before 2026-09-15 evening are too low. MuJoCo must stay at 3.8.1 in the
  LeRobot venv; 3.13 changes the contacts enough to break the expert.

## Just fixed

- **`boundary-stones` contains again.** Max radius 1.77 m in a 2.0 m circle,
  mean speed 0.47 m/s. It was not a tuning problem. The suspension was
  driving the robot: an unreachable `target_ride_height` pinned the height PD
  term at its clip, and that correction only touches the trailing support
  rods, which is how the rolling gait pushes. `regen_check/README.md` has all
  three faults and the measurements.
- **`demos/chimney` climbs out again.** Peak 3.59 m, lands on the 3.3 m box
  top at 13.7 s. See the rod-mechanism note below for the cause.
- **`docs/backflip_skill.md` is kept on purpose.** The backflip is going to be
  rebuilt. The banner now reads as a specification, not as an obituary. The
  `backflip` row in `skills/README.md` is marked planned, since the name is
  not in `SKILL_REGISTRY`.
- **`skills/high_level/` has its first skill.** `go_to_goal` in
  `goal_seeking.py` returns a skill name plus arguments, never rod targets.
  It routes around declared obstacles and decides over-or-around from the
  jump calibration. `skills/README.md` documents it;
  `tests/test_goal_seeking.py` has 13 tests, including a route property
  checked on 200 random maps. `demos/goal_seeking/` shows it: the video pairs
  a chase camera with a plan view drawing the inflated pillars, the chosen
- **Multi-stage RL maze locomotion successfully retrained & benchmarked.**
  As noted under Open below, switching to `multi_stage` rods absorbed the single-stage
  RL policy's thrust impulses (speed dropped from ~1.1 m/s to 0.061 m/s, causing
  0/7 maze completions). Retraining PPO natively on `multi_stage` for 400k steps
  (`storage_local/20260911_0011__local_455348__train_rl/`) restored full locomotion:
  mean speed jumped 11x to 1.03 m/s, solving the training maze in 422 steps and
  generalizing zero-shot to 4 of 6 unseen maze topologies (including the 45m Gauntlet
  with 0 wall collisions). Complete suite of videos and evaluation metrics are saved
  in the experiment run folder `storage_local/20260911_0011__local_455348__train_rl/renders/`.

## Open

- **The `multi_stage` rod may have broken more than the chimney.** Commit
  `ead03dd` switched `configs/rl/config.yaml` from `single_stage` to
  `multi_stage`. That rod puts a passive middle stage in series with the
  actuator through a soft equality constraint, so an impulse is partly
  absorbed instead of reaching the ball. The chimney was tuned before that
  commit and stalled at 0.82 m against a 3.3 m lip; raising
  `actuator_force_limit` from 100 N to 200 N in `configs/rl/chimney.yaml`
  brings it back to 5 of 6 orientation seeds. Any other demo tuned before
  `ead03dd` deserves the same check. The bisect: `ad86844` passes,
  `ead03dd` fails, nothing since made it worse.
- **The chimney is slower than it was.** 13.7 s to the box top against 4.8 s
  at `ad86844`, and 5 of 6 seeds against a recorded 6 of 6. The remaining gap
  is inside `ead03dd`'s asset changes, which also moved `joint_damping` from
  0.5 to 0.35, `joint_frictionloss` from 0.8 to 0.08 and the rubber solref
  time constant from 0.020 to 0.006. None of those was bisected further.
- **Two more demos fail, and neither is the force limit.** Found while
  checking the rest of `run_demo.py demo=all`; both predate this work and
  neither touches anything changed here.
  - `vertical_cylinder` does not climb at all: +0.02 m against a 3.5 m
    target, with tangential and vertical speed both at zero. It shares
    `configs/rl/chimney.yaml`, so it was re-run at 100, 200 and 400 N and the
    result is identical every time. Whatever stops it, the actuator limit is
    not it.
  - `platforms` makes 4 of 5 jumps. The last one, onto box4, is a 0.16 m drop
    across a 0.50 m gap; the ball stalls at x 10.63 and spends 2,500 steps
    there. The four before it all clear.
- **`scratch/`** is 331 tracked files that are gitignored. Untrack with
  `git rm -r --cached scratch/` when convenient.
- **The blog videos are stale for boundary-stones.** `regen_check/` holds the
  fixed render; `docs/blog/assets/` still holds the published one. Copying it
  over is a decision, not a chore, so it was left alone.

## Things worth knowing before changing code

- A tuning number is declared once. `SuspensionGains` holds the nine
  suspension gains; passing one as a keyword raises `TypeError` on purpose.
- A `SuspensionGains` target the build cannot reach is not a harmless
  request. The height term saturates at its clip and stays there, and since
  the correction only extends the trailing support rods, a constant clip is a
  constant forward push. Measure what the robot actually holds on that
  surface before setting `target_ride_height`.
- Any caller of `stay_in_boundary` or `traverse_rough_terrain` that drives the
  skill directly must pass its own `SuspensionState`. `skills/runner.py` does
  it automatically; the blog render scripts bypass the runner and have to do
  it by hand.
- `mid_level` may import `low_level`, never the reverse.
  `tests/test_skill_levels.py` enforces it.
- A module is never named after a skill. `skills.slalom` used to resolve to
  the function while `skills.stairs` resolved to the module.
- Demo `expect:` bounds are regression guards measured from a real run, not
  physical ideals. Each demo's yaml records its reference numbers.
