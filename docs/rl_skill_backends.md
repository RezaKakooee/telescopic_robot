# Choosing the RL skill library

For playground and campus training/evaluation, set this in the scenario config:

```yaml
rl:
  skill_backend: skills  # skills | skills_rl (default)
  action_mode: macro    # macro | hybrid, supported by both backends
```

`skills_rl` preserves the existing seven primitive actions and their parameter
encoding. The policy learns when to drive, thrust, tuck, brake, brace, conform,
or stand, including how to sequence those actions into a jump.

`skills` calls the existing functions in `skills/` through `skills.runner`.
The ten actions, in checkpoint order, are:

1. `move`
2. `stop`
3. `reverse`
4. `follow_path`
5. `straddle_gap`
6. `traverse_rough_terrain`
7. `jump_up`
8. `jump_forward_while_stopped`
9. `jump_forward_while_moving`
10. `jump_to`

Heading comes from the next route waypoint. `follow_path` uses the full route;
the terrain controller receives current contacts, clearances, and suspension
state. `straddle_gap` drives **along** a longitudinal gap with support on both
sides; it is not a jump across a transverse valley.

Each jump selection executes its crouch/run-up, burn, flight, and landing phases
before returning control to PPO. State and rod targets update on every low-level
step, with episode termination and boundary checks throughout. Landing uses
local terrain height, including elevated platforms. A landing phase lasts
0.2 seconds; options have a timeout of 1.6 seconds (2.4 for the running jump).
`jump_to` ends its burn early when its requested vertical velocity is reached.
Timeouts are reported in `info.skill_timed_out`; completing a sequence does not
guarantee that an obstacle was cleared.

Other controllers run for `rl.decision_every` low-level steps. Jump decisions
therefore represent more simulation time than movement decisions. Episode
time limits, step cost, and the no-progress counter account for executed control
steps. PPO's discount remains per policy decision, so gamma is not a per-second
discount. Info includes `skill_backend`, `skill_name`, `skill_phase`,
`skill_control_steps`, and cumulative `control_steps`.

The `skills` macro action is a Box of 10 preference scores (argmax selects the
skill). Hybrid uses 13 values: the same scores followed by heading offset
(-90 to +90 degrees), speed (0.2–1.6 m/s), and jump power (0.35–1).
Unused parameters are ignored. `jump_to` maps speed to horizontal target
velocity and the power channel to vertical target velocity (1.4–3 m/s).
Macro defaults are speed 1.1, full jump power, and `jump_to` targets 0.6/2.6 m/s.

Dedicated arena skills such as `climb_stairs`, `wall_run`, and `wall_of_death`
are not exposed by this adapter: their standalone runners require additional
arena geometry and maneuver planning. Registry aliases are also omitted.
Steering environments for other scenarios keep their existing action interface.

## Commands

Start fresh training with the parkour curriculum and the new backend:

```bash
python scripts/rl/train_rl.py kind=playground config_name=playground_parkour_skills
```

Or choose either backend with an override on the existing config:

```bash
python scripts/rl/train_rl.py kind=playground config_name=playground_parkour rl.skill_backend=skills
python scripts/rl/train_rl.py kind=playground config_name=playground_parkour rl.skill_backend=skills_rl
```

Backend changes require a **fresh policy**, because action spaces and meanings
differ. Do not resume a `skills_rl` checkpoint as `skills`, or vice versa.
The default backend still accepts the existing checkpoints.

Evaluate a run using its saved config and normalization statistics:

```bash
python scripts/rl/eval_rl.py run=storage_local/<training-run> episodes=1
```

Evaluation selects the same environment/backend as training and checks the
checkpoint spaces. It saves videos under the run's `evaluation/renders/`
directory, sampling simulation time inside jump options. Explicit `config` or
`config_name` overrides the saved config; use the backend/action mode that the
checkpoint was trained with.

The switch reduces the sequencing burden on the policy. Learning the correct
skill and launch location still requires training and obstacle-level evaluation.
