# skills — commandable motion primitives

You command a skill by name. The skill decides which of the 60 rods to
extend. That is the whole idea.

Every skill is a **pure function**. State goes in. Rod targets come out.

```python
targets = execute_skill("move_forward", quat, dirs_body, max_extend, d_hat=[1, 0])
env.step(targets)
```

Return value is always a `(n_bars,)` array of rod extensions in metres,
inside `[0, max_extend]`.

This file is the API reference. For **how the method works** — the task-frame
projection, the scoring windows, the leading-sector mask, and why the jump
skills are phase machines — see
[`docs/project_journey/02_skill_library_and_the_skill_course.md`](../docs/project_journey/02_skill_library_and_the_skill_course.md) §2.

## One movement skill controlled by speed

Use `move` for normal, faster, slower, and reverse travel. Keep `d_hat` fixed
and change only `speed` (in m/s):

```python
run_skill(env, "move", d_hat=[1, 0], speed=0.6)  # 60 cm/s
run_skill(env, "move", d_hat=[1, 0], speed=1.2)  # 120 cm/s
run_skill(env, "move", d_hat=[1, 0], speed=2.0)  # 200 cm/s
run_skill(env, "move", d_hat=[1, 0], speed=-1.2) # reverse at 120 cm/s
run_skill(env, "move", d_hat=[1, 0], speed=0.0)  # stop
```

```bash
PYTHONPATH=. python scripts/skills/run_skill.py skill=move speed=0.6 open_arena=--steps 500
```

Positive speed follows the reference, negative speed opposes it, and zero
uses the stop controller. The sign applies after any requested turn. The
reference stays fixed during a command; reversing does not mutate it.
For direct calls, measure `cross_track_error` left of the resulting travel
direction, including the speed sign. The runner handles this automatically.

The older names `move_forward`, `go_fast`, `go_slow`, and `reverse` remain available for
existing callers; new examples use `move` with an explicit speed.

### Forward movement with feedback

For the current antenna-style robot, use `skills.runner.run_skill` to supply
live velocity and the configured rod mechanism automatically. During `move`
and `move_forward`, it also measures sideways displacement from the line
through the starting position in the requested direction:

```python
from skills.runner import run_skill
run_skill(env, "move", steps=600, d_hat=[1, 0], speed=1.2)
```

Direct calls can pass `lin_vel`, `cross_track_error` (signed metres left of
the requested line), and `rod_mechanism`. Velocity enables speed correction
and sideways damping; the distance error adds line restoration. The single-step
`skill_targets` helper supplies velocity but needs a caller-provided distance
error for line restoration. Without these inputs, direct calls retain the
existing feedforward behavior. Explicit `back_gain` bypasses the new calibration
and motion corrections, preserving manually tuned composed skills.

The 16 cm `multi_stage` build has a separate calibration, reproducible with
`PYTHONPATH=. python scripts/skills/calibrate_move.py`. Other builds keep their
existing curves. In the six-second blog run, the corrected 120 cm/s command
averages 121.4 cm/s over seconds 4–6 and ends 6.1 cm off the starting line.
These are flat-ground measurements, not guarantees for different surfaces.

The older tables below describe the original feedforward presets, not fresh
antenna-model measurements.

### One signed-angle turn skill

`turn` accepts `angle_deg`: positive degrees turn right, negative degrees turn
left, and zero continues straight relative to `d_hat`, viewed from above.

```python
run_skill(env, "turn", steps=400, d_hat=[1, 0], angle_deg=+30)
run_skill(env, "turn", steps=400, d_hat=[1, 0], angle_deg=-30)
```

Both examples use the same world +x reference. Hold that reference fixed
throughout a command; the angle is a heading offset, not a per-step rotation.
The runner supplies velocity feedback and line tracking automatically.

```bash
PYTHONPATH=. python scripts/skills/run_skill.py skill=turn angle_deg=30 open_arena=--steps 400
```

The existing `move_left` and `move_right` presets remain compatible. The lower
level `move(turn=...)` API still uses counter-clockwise radians; the new `turn`
skill converts the user-facing clockwise-positive degrees internally.

### Parametric gait

There is a single locomotion skill. It takes two continuous numbers in
physical units, which together are a complete action for this robot:

```python
targets = execute_skill("move", quat, dirs_body, max_extend,
                        d_hat=[1, 0],     # reference direction
                        turn=-0.7,        # radians to rotate it by
                        speed=1.4)        # m/s
```

| Parameter | Range | Meaning |
|---|---|---|
| `turn` | −π … π rad | how far off the reference to drive |
| `speed` | 0.33 … 2.80 m/s | how fast to cruise |

`speed` is m/s, not an internal gain, so a planner or a policy asks for a
physical quantity. The conversion comes from `SPEED_CURVE`, measured on flat
ground. Verified against the simulator:

| asked | achieved | | asked | achieved |
|---|---|---|---|---|
| 0.40 m/s | 0.36 | | −90° | −84.5° |
| 1.20 m/s | 1.17 | | 0° | −0.5° |
| 2.40 m/s | 2.39 | | +90° | +87.1° |

Speed lands within 0.04 m/s, angle within 9°, and the speed holds whatever
angle is asked for.

`move_forward`, `move_right`, `move_left`, `reverse`, `go_fast` and `go_slow`
are all one-line presets of `move`. They exist because named commands read
better in a plan, not because they are different gaits.

| Preset | is | 
|---|---|
| `move_forward` | `move(turn=0, speed=1.2)` |
| `go_fast` | `move(turn=0, speed=2.25)` |
| `go_slow` | `move(turn=0, speed=0.45)` |
| `move_right` | `move(turn=-π/2)` |
| `move_left` | `move(turn=+π/2)` |
| `reverse` | `move(turn=π)` |

## Core skill registry

| # | Skill | What it does | Needs |
|---|---|---|---|
| 0 | `move` | **The gait.** Drive at any angle, any speed. | `d_hat`, `turn`, `speed` |
| 1 | `move_forward` | `move(turn=0)`. | `d_hat`, `speed` |
| 2 | `move_right` | `move(turn=-π/2)`. | `d_hat`, `speed` |
| 3 | `move_left` | `move(turn=+π/2)`. | `d_hat`, `speed` |
| 4 | `stop` | Brakes to rest within `stop_distance`. | `lin_vel`, `stop_distance` |
| 5 | `go_fast` | Forward at full power. | `d_hat` |
| 6 | `go_slow` | Forward at low power. | `d_hat` |
| 7 | `reverse` | `move(turn=π)`. | `d_hat`, `speed` |
| 8 | `push_against_wall` | Extends the rods on the wall side. | `wall_normal` |
| 9 | `jump_up` | Vertical jump from standstill. | `phase` |
| 10 | `jump_forward_while_stopped` | Forward jump from standstill. | `d_hat`, `phase` |
| 11 | `jump_forward_while_moving` | Hurdle leap while sprinting. | `d_hat`, `phase` |
| 12 | `fall_down` | Step off a ledge; cushion scales with `drop_height`; `gear` opens the rods underneath on long drops, `brace_front` bumpers a wall ahead. | `d_hat`, `phase`, `drop_height`, `gear`, `brace_front` |
| 13 | `jump_to` | Standing jump aimed by TAKE-OFF VELOCITY: the burn is servo-controlled against the live velocity each step, with lateral drift trimmed to zero. | `d_hat`, `phase`, `vx_target`, `vz_target`, `wall_lock` |
| 14 | `circle` | Continuous circular orbit with pure-pursuit lead & dynamic understeer compensation. Holds radius to within ±1.5 cm. | `ball_xy`, `center_xy`, `radius`, `speed`, `clockwise` |
| 15 | `straddle_gap` | Dual-flank outrigger locomotion across a central hole/trench between two platforms (Box 1 & Box 2). Tucks central underbelly while driving on lateral flanks with active heading centering. | `d_hat`, `speed`, `lateral_offset`, `min_lat` |
| 16 | `chimney_climb` | Between two walls, under free physics: `launch` off the floor, `push`/`fly` wall-jump zig-zag up, `hold` (clamp both walls, ~1 kN), `descend` (clamp extension servoed on vz). Exits over the LOWER wall onto its top. | `wall_axis`, `phase`, `side`, `clamp_ext`, `push_frac`, `x_off` |
| 17 | `backflip` | Rightward rebound with counter-clockwise airborne pitch; aliases: `somersault`, `flip`. | `phase`, `direction`, `launch_power`, `launch_torque` |
| 18 | `stairs` | Dispatches a composed stair traversal: planned `jump_to` hops, `stop`, `move`, and controlled `fall_down` drops; aliases: `climb_stairs`, `step_vault`. | `phase`, `d_hat`, live velocity, planned takeoff/drop values |




`d_hat` is a 2-vector heading in world xy. `wall_normal` is a 2-vector
pointing **from the wall toward the robot**.

### There is no "forward"

The shell is a Fibonacci sphere, so the robot has no front, back or side.
`d_hat` is not a facing the robot holds — it is simply the direction being
asked for, rebuilt from scratch every step. Nothing stores a heading.

Driving at eight compass headings covers 4.40 m to 4.60 m over the same run,
a spread of 4.5 %. The robot is genuinely direction-blind, so `move_right`,
`move_left` and `reverse` are one gait on a rotated heading rather than three
separate gaits.

What is *not* symmetric is the robot's momentum. Changing the commanded
direction while rolling costs time to re-establish speed:

| turn | time to regain 90 % of cruise |
|---|---|
| 0° | 0.01 s |
| 30° | 0.10 s |
| 60° | 0.33 s |
| 90° | 0.57 s |
| 180° | 0.82 s |

So the geometry is isotropic but the state is not. A planner should treat a
direction change as costed, not free.

## Measured behaviour

From `tests/test_skills.py`. Numbers are real MuJoCo runs, not estimates.

| Skill | Result |
|---|---|
| `move_forward` | +0.121 m in 40 steps, 0.48 m/s |
| `move_right` | −0.182 m sideways, 0.87 m/s |
| `move_left` | +0.198 m sideways, 0.89 m/s |
| `stop` | 1.12 m/s to 0.015 m/s. 99% cut. Coasts 0.35 m. |
| `go_fast` | +0.239 m in 40 steps, 1.01 m/s |
| `go_slow` | +0.036 m in 40 steps, 0.14 m/s |
| `reverse` | −0.208 m backward, 0.87 m/s |
| `push_against_wall` | Wall-side rods 0.074 m, far-side 0.031 m. Pushes 3.3 cm off a real maze wall. |
| `jump_up` | Peak 0.589 m. Net lift +41.5 cm. |
| `jump_forward_while_stopped` | Peak 0.577 m, +0.41 m forward. |
| `jump_forward_while_moving` | Peak 0.483 m, +3.60 m forward, 2.55 m/s. |
| `fall_down` | Drops 36 cm off a ledge, lands upright at 2.27 m/s. |
| `stairs` | Compact 1.8 m-wide course: three first-attempt 25 cm climbs and three verified drops; peak z 1.602 m, zero core impacts, settles at x=11.28 m. |

## Jump phases

The three jump skills are state machines. You pass a `phase` string.

- `jump_up`: `crouch` → `takeoff` → `airborne` → `landing`
- `jump_forward_while_stopped`: same phases, rear-biased takeoff
- `jump_forward_while_moving`: `sprint` → `dip` → `launch` → `airborne` → `landing`
- `fall_down`: `edge` → `freefall` → `absorb` → `settle`
- `stairs` ascent: `hop_crouch` → `hop_takeoff` → `hop_airborne` → `hop_landing`; descent delegates to the complete `fall_down` sequence
- full backflip runner: `approach` → `compact` → `counter-plant` → `preload-tuck` → `rebound-launch` → `tuck` → `flare` → `settle`

`fall_down` is driven by **height**, not by a step count. How long the creep
to the lip takes varies, so a fixed schedule tucks the rods at the wrong
moment.

You do not have to time these yourself. `skills/runner.py` holds the
verified timings and applies them for you.

The backflip is an exception because it also distinguishes a preload hop from
the main flight using live contacts. Run it through
`scripts/skills/run_somersault.py`. See the
[Backflip Skill guide](../docs/backflip_skill.md) for its signed direction
convention, state machine, telemetry, and acceptance criteria.

## Running a skill

```bash
# list all skills
python scripts/skills/run_skill.py list=true

# run one skill
python scripts/skills/run_skill.py skill=go_fast steps=200 video=true

# the 7 ground skills as ONE continuous labelled take (best for comparing them)
python scripts/skills/run_skill.py combo=--video camera=fixed_close_dual

# run all 11 in sequence, one 30 s video per skill
python scripts/skills/run_skill.py demo=--seconds 30 video=--per-skill \
    --camera fixed_close_dual

# push against the nearest maze wall (lidar finds it)
python scripts/skills/run_skill.py skill=push_against_wall kind=maze \
    --config configs/rl/config.yaml --seed 3 --seconds 30 --video --per-skill

# physics-verified rightward, counter-clockwise backflip
python scripts/skills/run_somersault.py

# geometry- and contact-verified stair course (add --no-video for tests)
python demos/stairs/runner.py
```

The stair controller is intentionally a composition of existing skills. The
runner plans each tread hop with `skills/mid_level/hop_planner.py`, retries after a small
sideways footing change when orientation produces a weak launch, and counts a
step only after contact with that tread plus a stable pose inside its bounds.
See the [Stairs Skill guide](../docs/stairs_skill.md).

Videos land under `storage_local/<run id>/renders/`.

### Options that matter for video

| Flag | Effect |
|---|---|
| `--combo` | One continuous take of the 7 ground skills, with labels. |
| `--seconds N` | Repeat the skill's cycle until the clip is at least N seconds. |
| `--per-skill` | One file per skill instead of one long clip. |
| `--open-arena` | Force the open goal arena. |
| `--camera` | `fixed_close_dual` follows the ball. `dual` does not. |

### Why `--combo` exists

A skill on its own is hard to judge. From a standstill, `move_forward` and
`reverse` look the same. `go_slow` means nothing without `go_fast` next to it.

`--combo` runs all 7 ground skills back to back with **no reset**. Each skill
inherits the momentum the last one left. Every frame is stamped with the
active skill, live speed, vx, vy, and distance travelled.

Each skill holds for 30 s by default, so there is time to see what it does.
Change it with `--seconds`. Total run is 7 × that.

| Order | Skill | Measured over 30 s |
|---|---|---|
| 1 | `move_forward` | +5.71 m, tops out at 0.91 m/s |
| 2 | `go_slow` | +0.86 m, decays to a standstill |
| 3 | `go_fast` | +9.46 m, tops out at 1.44 m/s |
| 4 | `move_right` | −3.69 m sideways |
| 5 | `move_left` | +2.94 m sideways |
| 6 | `reverse` | −2.84 m backward |
| 7 | `stop` | Brakes to 0.00 m/s |

A single skill run lasts only a few seconds, so `--seconds` repeats a cycle.
What counts as a cycle depends on the skill:

- `stop`: accelerate, then brake. A brake is invisible if the ball is still.
- `push_against_wall`: push off the wall, then roll back to it.
- Jump skills: jump, then settle to rest before the next jump.
- Everything else: keep driving.

### Arena warning

The jump track has guide rails at y = ±1.2 m and hurdles at x = 1.45 m and
3.25 m. A 30 s roll hits them and stalls. `--seconds` switches to the open
goal arena by default. Pass `--kind` to override that.

From Python:

```python
from skills.runner import run_skill, run_program

stats = run_skill(env, "go_fast", steps=200, d_hat=np.array([1.0, 0.0]))
stats = run_skill(env, "jump_up")          # phase timing is automatic

run_program(env, [
    ("go_fast", 60, {"d_hat": FORWARD}),
    ("stop", 80, {}),
    ("jump_up", None, {}),
])
```

## Running a skill as a demo

A demo is a yaml, not a script. `demos/<name>/demo.yaml` names the
scenario, the skill, how long to run, which cameras to record, and what
counts as success. One runner executes all of them:

    python scripts/run_demo.py demo=gap
    python scripts/run_demo.py demo=all video=false
    python scripts/run_demo.py list=true

Each demo's `expect` block turns it into a regression test, and
`tests/test_demos.py` runs every one. See `demos/README.md`.

Demos whose control flow is the point stay as scripts under
`scripts/skills/`: the course state machines, the phase machines and the
calibration sweeps.

## Writing a skill

The package holds to a few conventions. `tests/test_skill_interface.py`
enforces the ones a machine can check.

1. **Signature.** Start with `(quat, dirs_body, max_extend, ...)`. Every one of
   the 30 motion skills does. Everything after that is keyword-only.
2. **Return.** A `(n_bars,)` array of rod extensions in metres, inside
   `[min_offset, max_extend]`. Return `(targets, meta)` only if the skill has
   telemetry worth reporting; `execute_skill` supplies a generic summary for
   the rest, so callers never have to know which kind they are holding.
3. **Docstring.** One full sentence on the first line. A skill with eight or
   more options needs a `Parameters` section, or a `Phases` section if it is a
   phase machine and explains its options there.
4. **Shared names keep shared meanings.** `min_offset` is 0.025 everywhere,
   except in `stop`, which has no `min_offset` at all: it holds its stance on
   `stance_height` and retracts every other rod to zero. Passing `min_offset`
   to `stop` raises `TypeError`, which is how `follow_path` used to crash on a
   single-waypoint path.
   `rod_mechanism` defaults to `None`, meaning the generic speed calibration.
   `back_gain` overrides the speed lookup. Do not invent a different default
   for one skill: `curve` used to default `rod_mechanism` to `"multi_stage"`,
   which quietly drove it about 15 % softer than its neighbours.
5. **Register it.** Add the name to `SKILL_REGISTRY` in `skills/__init__.py`,
   otherwise `execute_skill` cannot reach it and the runner cannot drive it.

## Every name in the registry

`execute_skill` accepts all of these. Aliases exist because callers and
older scripts spell the same skill differently; they take identical
arguments and run identical code. `meta["skill"]` always reports the
function that actually ran, and `meta["requested_as"]` the name you used.

| Skill | Also answers to | Module |
| --- | --- | --- |
| `move` | — | `locomotion.py` |
| `turn` | — | `locomotion.py` |
| `move_forward` | — | `locomotion.py` |
| `move_right` | — | `locomotion.py` |
| `move_left` | — | `locomotion.py` |
| `stop` | — | `locomotion.py` |
| `go_fast` | — | `locomotion.py` |
| `go_slow` | — | `locomotion.py` |
| `reverse` | — | `locomotion.py` |
| `circle` | — | `locomotion.py` |
| `curve` | `curved_movement` | `locomotion.py` |
| `straddle_gap` | `straddle` | `locomotion.py` |
| `surface_drive` | `wall_ride` | `locomotion.py` |
| `wall_of_death` | `motordrome` | `bowl_riding.py` |
| `wall_run` | `horizontal_wall_run` | `wall_running.py` |
| `slalom` | `training_cones`, `curved_slalom`, `curved_training_cones` | `cone_courses.py` |
| `follow_path` | `track_path` | `navigation.py` |
| `traverse_rough_terrain` | `rough_terrain`, `active_suspension` | `terrain_following.py` |
| `stay_in_boundary` | `stay_within_boundary`, `boundary_containment` | `navigation.py` |
| `push_against_wall` | — | `climbing.py` |
| `chimney_climb` | `chimney`, `vertical_climb` | `climbing.py` |
| `jump_up` | — | `jumping.py` |
| `jump_forward_while_stopped` | — | `jumping.py` |
| `jump_forward_while_moving` | — | `jumping.py` |
| `jump_to` | — | `jumping.py` |
| `climb_stairs` | `stairs`, `step_vault` | `stair_climbing.py` |
| `fall_down` | — | `falling.py` |

Public skill functions that are **not** in the registry, so
`execute_skill` cannot reach them:

- `chimney_friction_servo` in `climbing.py`
- `chimney_step_down` in `climbing.py`
- `cylinder_spiral_climb` in `climbing.py`

## Verify

```bash
PYTHONPATH=. python scripts/run_tests.py          # fast suite, about 20 s
PYTHONPATH=. python scripts/run_tests.py all=true    # plus the long drivers
PYTHONPATH=. python scripts/run_tests.py list=true   # what would run
```

The fast suite is everything `unittest discover` collects under `tests/`.
Modules written as plain functions opt in through `tests/_function_suite.py`;
without that hook `discover` walks past them and reports a green run having
executed none of them.

The long drivers stay runnable on their own:

```bash
MUJOCO_GL=egl PYTHONPATH=. python tests/test_skills.py
```

Runs the named skills in MuJoCo and asserts the expected motion. About ten
minutes.

## Boundary containment (wall-less arena)

`stay_in_boundary` confines the robot inside an open circular boundary marked purely
on the floor with ZERO physical walls. Inside the arena, it roams slowly, alternating
between diverse exploratory maneuvers (`move`, `curve`, `turn`, and `stop`). As it
approaches the boundary line, active vector-field deflection blends the perimeter
tangent with the inward normal and regulates speed to ensure zero boundary breaches:

```python
targets, meta = execute_skill(
    "stay_in_boundary", quat, dirs_body, max_extend,
    ball_xy=env.data.qpos[:2],
    lin_vel=env.data.qvel[:2],
    boundary_radius=2.0,
    speed=0.45,
    safety_margin=0.50,
    step_count=step,
    return_metadata=True,
)
```

## The pillar course (standing hops)

`demos/pillars/runner.py` climbs a ladder of narrow columns -- 0.90 m
pads, up to 3.5x the core in height -- where every ascent is a `jump_to`
standing hop planned by `skills/mid_level/hop_planner.py` from the pad's own geometry,
and every descent is a `fall_down` roll off the lip (the 0.12 m gaps are
narrower than the ball, so no jump is needed downhill). The planner works
from `skills/hop_calibration.json`, measured over RANDOM orientations
(`scripts/skills/calibrate_hop.py`).

Two more layers handle what the servo cannot:

- **Probe-abort.** Nothing at the stance predicts a jump (r = 0.28), but the
  burn predicts itself: vz at step 8 gives lift-off vz at r = 0.97, vx at
  step 5 gives lift-off vx at r = 0.95. The first steps of the real jump are
  the probe; a launch outside the plan's bracket is aborted centimetres up,
  the footing is shuffled, and it goes again.
- **Side-lane recovery.** A ball that falls beside the pillars drives back
  down the lane to the start and climbs again.

Over 14 orientation-randomized seeds (`--seed N`): 14/14 complete, 10 with
no intervention at all. `--demo-recovery` stages a fall so the recovery can
be seen.

## Adding skill 13

Three steps.

1. Write the function in the module for its family: `locomotion.py`,
   `navigation.py`, `terrain_following.py`, `climbing.py`, `jumping.py`,
   `falling.py`, `stair_climbing.py`, `cone_courses.py`, `wall_running.py`
   or `bowl_riding.py`. Never name a module after a skill: see
   `test_no_module_is_named_after_a_skill`.
   Keep the contract: state in, `(n_bars,)` targets out.
2. Add one line to `SKILL_REGISTRY` in `__init__.py`.
3. If it needs state beyond `quat`/`dirs_body`/`max_extend`, add its name to
   the matching set in `runner.py` (`NEEDS_HEADING`, `NEEDS_VELOCITY`,
   `NEEDS_WALL_NORMAL`, or `PHASE_SKILLS`).

Then add a check to `tests/test_skills.py` so it stays verified.
