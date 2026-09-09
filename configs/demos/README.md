# demos

One yaml per skill demo. `scripts/run_demo.py` runs them; the logic lives once
in `radial_sphere/demo.py`.

    python scripts/run_demo.py demo=gap
    python scripts/run_demo.py demo=all video=false
    python scripts/run_demo.py list=true

## The spec

| Key | Meaning |
| --- | --- |
| `scenario` | `kind`, `config` and `seed` for `generate_scenario` |
| `spawn` | optional starting `x`, `y`, `z`, written into qpos |
| `settle` | steps to run before recording, with the fixed skill args only |
| `skill.name` | a name from `SKILL_REGISTRY` |
| `skill.args` | fixed keyword arguments |
| `skill.args_from_scenario` | arguments read off the scenario, such as cones |
| `skill.feedback` | arguments bound to live state, re-read every step |
| `steps` | the step budget |
| `stop_when` | finish early once **every** bound holds |
| `stop_when_any` | a list; finish as soon as **any** entry holds |
| `video.panes` | one or more cameras, each with overlay text |
| `expect` | bounds on the metrics, checked at the end |

## Live state

`feedback` bindings and overlay text share one namespace:

    step time ball_x ball_y ball_z ball_xy vx vy vz lin_vel speed
    distance_x path_length goal_distance goal_x

Overlay lines are format strings over it, so `"x = {ball_x:.2f} m"` needs no
code.

## Metrics and expectations

Every run reports `final_x/y/z`, `distance_x`, `path_length`, `mean_speed`,
`max_speed`, `min_z`, `max_z`, `core_impacts`, `goal_distance`, `steps_run`
and `duration_s`. `expect` places `min` and `max` bounds on any of them.

Those bounds are **regression guards taken from a measured run**, not physical
ideals. Each demo's yaml records the reference numbers in a comment so a
change in behaviour is visible rather than silently re-baselined.

## The settle phase

`settle` runs with the fixed `args` only, never the `feedback` bindings.
Closing the loop while the robot is still dropping onto its rods feeds it a
transient it should not react to, and the run diverges from there. Getting
this wrong cost the gap demo half a metre of travel.

## What stays a script

A demo whose control flow is its point stays in `scripts/skills/`: the course
state machines (`run_stairs`, `run_course`, `run_pillars`, `run_platforms`),
the phase machines (`run_wall_run`, `run_chimney*`, `run_vertical_cylinder`,
`run_cylinder_ramp_launch`, `run_motordrome_wall_of_death`), and the
calibration sweeps. This format describes one skill driven in one loop.
