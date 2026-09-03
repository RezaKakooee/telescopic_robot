# Stairs Skill

The stairs skill traverses a configured flight by composing existing robot
skills. It does not maintain a second, stair-specific jump implementation.

## Composition

| Course operation | Existing primitive | Phases |
|---|---|---|
| Position on a tread | `move` + `stop` | `cruise`, `poise` |
| Jump to the next tread | `jump_to` | `hop_crouch`, `hop_takeoff`, `hop_airborne`, `hop_landing` |
| Cross the plateau | `move` + `stop` | `plateau`, `poise` |
| Step down | `fall_down` | `edge`, `freefall`, `absorb`, `settle` |

`skills/stairs.py` is the stateless dispatcher. The course-level state machine
is in `scripts/skills/run_stairs.py`; it reads geometry from
`configs/rl/stairs_course.yaml` through `stairs_course_geometry()`.

## Why the old implementation failed

The original runner advanced its climb and descent counters after fixed time
windows. A reported landing could therefore be on the previous tread, and a
descent could be counted without a freefall or target contact. Its elapsed
time also assumed 0.02 s control steps even though this environment advances
0.01 s per action.

The verified runner instead:

1. Plans every upward hop with the measured envelope in
   `skills/hop_calibration.json`.
2. Uses live velocity to drive `jump_to` and live height/contact to change
   phases.
3. Requires robot contact with the named target geom and a stable final pose
   within that tread's x/z bounds before counting it.
4. Re-deals the sphere's footing with a small sideways move and retries when
   the post-landing orientation gives a weak launch.
5. Runs all four `fall_down` phases for every downward step.
6. Measures time from `model.opt.timestep * action_repeat`.

For the default course, a measured 8 cm preflight side roll before step 2
places a better rod set under the sphere. All three risers then clear on their
first attempt; the retry path remains available for disturbed runs.

## Verified course

Default compact geometry: three 0.25 m rises with 1.30 m treads, a 1.25 m
plateau, 1.80 m course width, and three 0.25 m drops. With seed 42:

| Result | Measurement |
|---|---:|
| Upward tread contacts | 3 / 3 |
| Downward support contacts | 3 / 3 |
| Peak core height | 1.602 m |
| Core-impact steps | 0 |
| Jump attempts | 1, 1, 1 |
| Final position | (11.28, 0.04, 0.20) m |
| Final linear speed | 0.005 m/s |
| Final angular speed | 0.022 rad/s |
| Simulated elapsed time | 32.22 s |

## Visual accessibility

The two flights use alternating dark-blue and teal tread materials. Every
leading edge has a 9 cm safety-yellow horizontal band plus a matching 9 cm
vertical riser band, so the height change remains visible from overhead and
oblique views. These larger bands have collision disabled; the original thin
physical nosing remains unchanged underneath, preserving the calibrated
contacts. The video runner uses a close 3.0 m side camera elevated 30 degrees
above the scene. Its position tracks the robot, while azimuth and elevation
remain fixed, keeping the robot and nearby tread edges large and readable.

## Run and test

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/skills/run_stairs.py --no-video
MUJOCO_GL=egl PYTHONPATH=. python scripts/skills/run_stairs.py
MUJOCO_GL=egl PYTHONPATH=. python tests/test_skills.py
```

Videos are written only when video recording is enabled, under the normal
run directory returned by `make_run_dir()`.
