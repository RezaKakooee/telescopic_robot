# Handoff: VLA skills, inspection courses, expert demos

Date: 2026-09-14. Supersedes the earlier version of this file.
Environment: conda `roboverse`, run from the repo root with `MUJOCO_GL=egl PYTHONPATH=.`.
LeRobot / SmolVLA live in a separate venv: `/home/storage_group/envs/lerobot/bin/python`.

## 1. What the robot does now

The ball picks one macro skill every 0.1 s (10 control steps). The skill runs
through `SkillArbitrationEnv` with `skill_backend: skills`, which calls the
`skills` library through `skills.runner.skill_targets`. So every option runs
the current skill code.

| Env option | Skill code | Notes |
|---|---|---|
| `move` | `skills.move` | heading from the waypoint tracker |
| `reverse` | `skills.reverse` | used by the expert to back up before a retry |
| `stop` | `skills.stop` | |
| `traverse_rough_terrain` | `skills.traverse_rough_terrain` | stone fields |
| `jump_forward_while_moving` | `skills.jump_forward_while_moving` | see the jump fixes below |
| `crawl_pipe` | `skills.crawl_pipe` | new. Rolling inside a round conduit |

`skills_vla/` is the policy-facing layer: six classes
(`roll, jump_forward, jump_gap, traverse_rough, brake_stop, crawl_pipe`), params
in [-1, 1], egocentric heading. Each class wraps the same `skills` function.
`skills_vla.ENV_SKILL_MAP` maps a class to its env option.

### Jump fixes (radial_sphere/handcrafted_skill_backend.py)

The macro running jump used to be an accident. Its 0.55 s "sprint" phase
kicked a rolling ball into the air with the rods out; the real launch came
0.55 s later, past the obstacle. Fixed:

- sprint is skipped when the ball already rolls faster than 0.5 m/s;
- power 0.9 and landing rollout 0.04 (`MACRO_JUMP_POWER`, `MACRO_ROLLOUT_GAIN`);
  1.0 flies off the 1.5 m boxes, 0.85 misses the box top;
- the option ends only when the ball has settled, not mid-bounce.

### crawl_pipe (skills/low_level/pipe_crawling.py)

Every rod is capped at the free length to the pipe wall along that rod.
Rods pointing down keep a 4 cm push into the floor (that is how the ball
rolls); side rods stop 2 cm short of the wall. The heading is steered to the
axis with velocity damping. Two metres before the mouth the rods are not
capped yet (the rim would stop the ball); this approach zone is on the entry
side only, relative to the travel direction. Works down to a 0.22 m radius;
the tucked ball is 0.40 m wide, so 0.20 m is the physical limit.

## 2. Obstacle hits: the success metric

Reaching the goal is not enough. A hit is a contact with an obstacle face
while the ball moves into it (`radial_sphere/mujoco_env.py`,
`HIT_GEOM_PREFIXES`, `_check_contacts`). Landing on a box top, rolling on a
pipe floor, or brushing a ledge while leaving it are not hits.

- `info["obstacle_hit"]`, `info["obstacle_hit_geoms"]`, `info["episode_hits"]`
- reward: `rl.obstacle_hit_penalty` per macro step (5.0 in
  `configs/rl/playground_parkour_skills.yaml`)
- `eval_skills_vla.py` reports `clean_success` (goal reached with zero hits)

## 3. Courses

### Playground (`playground`)

Dimensions live in `radial_sphere/playground_course.py`; the MJCF, the
scenario, the oracle windows and the tests read them. Current values are at
the measured limits of the running jump:

| Obstacle | Value | Clean limit |
|---|---|---|
| hurdle | 0.18 m | 0.18 m |
| boxes | 0.40 / 0.80 / 1.20 m, 1.5 m long, 0.60 m gaps | +0.40 m per step, 0.65 m gap |
| stairs | 2 x 0.40 m, 1.8 m treads | +0.40 m per step |
| wall (Station 6, replaces the old pipe) | 0.55 m | 0.60 m |

Clearing a thin wall and landing on a platform differ: over a wall the apex
must pass the top (core apex about 0.85 m); onto a platform the ball must
still be above the edge while coming down, with rods out for the landing.

The playground oracle (`scripts/data/generate_skills_vla_dataset.py`) fires
each jump from a measured distance before the obstacle edge
(`JUMP_TRIGGERS`). It reaches the goal with 0 hits. A BC policy trained on
25 of its demos also scores 100 % clean success (`storage_local/20260914_1317__local__train_skills_vla__v2_clean_oracle/`).

### Ten inspection courses (`radial_sphere/inspection_scenarios.py`)

`generate_scenario("inspection_<name>", cfg, seed=..., tour=False|True)`.
Simple objects, industrial situations, each with its own layout:

| Kind | Layout | Obstacles |
|---|---|---|
| warehouse | random shelving maze (seed) | pallets, fallen beam, spill |
| pipe_alley | open yard, 45 deg lane | posts, pipe saddles, conduit |
| tank_farm | open yard, S route | nine tanks, bund curbs, gravel |
| substation | serpentine through four bays | trenches, posts, gravel |
| loading_dock | hall | trucks, ramp, crates, stairs down |
| boiler_house | three rooms with doorways | floor pipes, stairs to a deck, ramp |
| utility_tunnel | 1.5 m tunnel, L with a dead end | two conduits, manhole, cable cover |
| solar_farm | open field, snake | post rows, ditch, mud |
| quarry | switchback hill | slopes, hairpin, rock fall, potholes |
| rubble_site | open square, diagonal | random slabs, beam, crack, collapsed wall |

`tour=True` gives a long route (30 to 150 m) that revisits obstacles and may
cross itself. Tours set `monotonic_path=True`, which makes the env use
`MonotonicWaypointTracker` (tracks the route in order; ties at dead ends go
forward) and count the goal only at the end of the route. Inspection walls
are 1.0 m tall (`WALL_HEIGHT`): the ball jumped a 0.22 m fence once.

Render all ten: `python scripts/scenarios/render_inspection_courses.py`.

## 4. The generic expert (`radial_sphere/inspection_oracle.py`)

Reads the obstacles from the scenario and turns them into stations along the
route (arc length). Same windows as the playground:

| Station | Trigger | Skill |
|---|---|---|
| slab across the route (>= 0.08 m) | 0.55 to 0.85 m before the edge | running jump |
| gap / trench | 0.25 to 0.45 m before the edge | running jump |
| stair riser (ascending) | 0.38 to 0.55 m before each riser | running jump |
| pipe | 2 m before the mouth to the exit | crawl_pipe |
| stone field | inside | traverse_rough_terrain |
| goal (end of route) | within 0.45 m | stop |

A retry rule: stalled for 6 macro steps in front of a station means back up
about 1 m (`reverse`), then jump again. An obstacle crossed several times on
a tour is several stations.

Run it: `python scripts/vla/run_inspection_oracle.py [--tour] [course ...]`
(chase video with a minimap, summary JSON). Results: 10/10 goals on the short
routes and on the tours; 5/10 with zero hits. Tunnel and solar-farm hits are
light touches (pipe walls, posts).

Rule learned the hard way: every obstacle needs 2 to 3 m of straight
approach. A turn right before a jump or a pipe mouth fails.

## 5. Demonstrations for VLA SFT

`scripts/data/generate_inspection_demos.py --episodes 30 --workers 10`
(no chase video; 10 minutes for 300 episodes). Per macro step: policy camera
256x256, 13-D state, skill class, params, the oracle's reason text, hit flag.
Jitter per episode: start pose, camera pose, course seed. Only episodes that
reach the goal are kept.

Latest dataset: `storage_local/20260914_1612__local__generate_inspection_demos__30ep/`
(216 of 300 episodes kept, 108k frames, 2.4 GB). Keep rates: tank farm,
substation, boiler, rubble about 30/30; quarry 25; pipe alley 20; solar 18;
warehouse 17; tunnel 10; loading dock 8. Failures are stalls; the dock ramp
and the tunnel pipes are sensitive to the start jitter.

`scripts/data/convert_inspection_demos_to_lerobot.py --demos <run dir>`
(LeRobot venv) writes a LeRobotDataset: `observation.image`,
`observation.state` (13), `action` (10 = one-hot skill + params), `task`,
`reason`. Argmax of the first six action entries gives the skill.

Rough SFT budget: 30 to 50 demos per course, 300 to 500 episodes, 150k to
250k frames. Jumps are about 1 % of the decisions; weight or oversample them.

## 6. Earlier VLA / RL results (playground, before the inspection courses)

- BC on the clean playground demos: 100 % clean success on 3 seeds.
- PPO fine-tuning of that policy (`scripts/vla/train_rl_skills_vla.py`,
  plain PyTorch, KL to BC, frozen ResNet): keeps 100 %, no gain. With a
  weak KL it forgot the box jump; with the hit penalty on the old (accidental)
  jumps it got stuck. RL has nothing to fix while BC is already clean.

## 7. Storage layout

Every experiment folder is `storage_local/<YYYYMMDD_HHMM>__local__<script>__<tag>/`.
All VLA scripts default to a new timestamped run dir. Exceptions kept on
purpose: `storage_local/_assets` (blog scripts), `storage_local/sci_out`
(SLURM logs).

## 8. Open

- Loading dock and utility tunnel keep rates (8/30, 10/30). Longer straight
  approaches or a smaller start jitter would help.
- Tanks render as short domes (pillars). No tall cylinder object exists yet.
- The hurdle and the wall sit at the physical limit; small changes flip them.
- `jump_to` (aimed hop) is unused: unreliable from a rolling start.
- Hybrid mode (the VLA also picks speed and power) is not wired for demos.
  Today one power serves every jump.
- SmolVLA fine-tuning on the LeRobot dataset is the next step.

## 9. Quick checks

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/run_tests.py                       # 190 tests, ~60 s
MUJOCO_GL=egl PYTHONPATH=. python scripts/vla/run_inspection_oracle.py --tour  # 10 tours with videos
MUJOCO_GL=egl PYTHONPATH=. python scratch/record_oracle_video.py             # playground oracle video
```
