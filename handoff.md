# Handoff — current state

Date: 2026-09-09. Supersedes the 2026-08-28 handoff (that story is in
`docs/project_journey/02_skill_library_and_the_skill_course.md`).

A 60-rod spherical robot in MuJoCo. It moves by extending and retracting
telescopic rods; there are no wheels and no legs. On top of that sits a
library of motion skills, a set of demos that prove them, and an RL stack.

## Layout

| Where | What |
|---|---|
| `radial_sphere/` | The robot and the runtime: MuJoCo env, MJCF builder, scenarios, the shared gait maths, the demo runner. |
| `skills/low_level/` | 9 modules. State in, 60 rod targets out. One behaviour each, no branching. |
| `skills/mid_level/` | 4 modules. Choose a low-level skill each step and delegate: `follow_path`, `stay_in_boundary`, `climb_stairs`, plus the jump planners. |
| `skills/high_level/` | Empty. Reserved for planning and RL policies that emit skill commands. |
| `demos/<name>/` | 16 folders. `demo.yaml` plus `runner.py` when the control flow is the point. |
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
PYTHONPATH=. python scripts/run_tests.py              # 82 tests, ~30 s
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

`main` is at `f43f2c8` and pushed. Working tree clean. 82 tests pass.

## Open

- **`boundary-stones` demo fails containment.** The robot leaves its 2.0 m
  circle by 78 cm. It used to pass, but only because a terrain-sensor bug was
  holding it to 0.30 m/s against a commanded 0.75. With the sensor fixed it
  reaches its commanded speed and `safety_margin=0.65` is too small. Scaling
  the margin with speed made the overshoot worse, so the demo needs real
  retuning. `regen_check/README.md` has the numbers.
- **`demos/chimney` fails its climb.** Peak 0.82 m, lip not cleared. Predates
  this work.
- **`scratch/`** is 331 tracked files that are gitignored. Untrack with
  `git rm -r --cached scratch/` when convenient.
- **`docs/backflip_skill.md`** documents code that no longer exists. It has a
  banner saying so; delete the file if you do not want the record.
- **`skills/high_level/`** is empty, with the contract written in its
  `__init__.py`.

## Things worth knowing before changing code

- A tuning number is declared once. `SuspensionGains` holds the nine
  suspension gains; passing one as a keyword raises `TypeError` on purpose.
- `mid_level` may import `low_level`, never the reverse.
  `tests/test_skill_levels.py` enforces it.
- A module is never named after a skill. `skills.slalom` used to resolve to
  the function while `skills.stairs` resolved to the module.
- Demo `expect:` bounds are regression guards measured from a real run, not
  physical ideals. Each demo's yaml records its reference numbers.
