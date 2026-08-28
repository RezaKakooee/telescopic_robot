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

## One gait, and presets of it

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

## The 14 skills

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

## Jump phases

The three jump skills are state machines. You pass a `phase` string.

- `jump_up`: `crouch` → `takeoff` → `airborne` → `landing`
- `jump_forward_while_stopped`: same phases, rear-biased takeoff
- `jump_forward_while_moving`: `sprint` → `dip` → `launch` → `airborne` → `landing`
- `fall_down`: `edge` → `freefall` → `absorb` → `settle`

`fall_down` is driven by **height**, not by a step count. How long the creep
to the lip takes varies, so a fixed schedule tucks the rods at the wrong
moment.

You do not have to time these yourself. `skills/runner.py` holds the
verified timings and applies them for you.

## Running a skill

```bash
# list all skills
python scripts/skills/run_skill.py --list

# run one skill
python scripts/skills/run_skill.py --skill go_fast --steps 200 --video

# the 7 ground skills as ONE continuous labelled take (best for comparing them)
python scripts/skills/run_skill.py --combo --video --camera fixed_close_dual

# run all 11 in sequence, one 30 s video per skill
python scripts/skills/run_skill.py --demo --seconds 30 --video --per-skill \
    --camera fixed_close_dual

# push against the nearest maze wall (lidar finds it)
python scripts/skills/run_skill.py --skill push_against_wall --kind maze \
    --config configs/rl/config.yaml --seed 3 --seconds 30 --video --per-skill
```

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

## Verify

```bash
MUJOCO_GL=egl PYTHONPATH=. python tests/test_skills.py
```

Runs the named skills in MuJoCo and asserts the expected motion.

## The pillar course (standing hops)

`scripts/skills/run_pillars.py` climbs a ladder of narrow columns -- 0.90 m
pads, up to 3.5x the core in height -- where every ascent is a `jump_to`
standing hop planned by `skills/hop_planner.py` from the pad's own geometry,
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

1. Write the function in `locomotion.py`, `interaction.py`, or `jumping.py`.
   Keep the contract: state in, `(n_bars,)` targets out.
2. Add one line to `SKILL_REGISTRY` in `__init__.py`.
3. If it needs state beyond `quat`/`dirs_body`/`max_extend`, add its name to
   the matching set in `runner.py` (`NEEDS_HEADING`, `NEEDS_VELOCITY`,
   `NEEDS_WALL_NORMAL`, or `PHASE_SKILLS`).

Then add a check to `tests/test_skills.py` so it stays verified.
