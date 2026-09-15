# Handoff: VLA skills, inspection courses, expert demos, SmolVLA SFT

Date: 2026-09-15 (evening). Supersedes the earlier version of this file.
Committed as `ab6a22e` on `main`; the SmolVLA evaluator, the demo video
export and the round-2 changes are in the working tree (see section 10).
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

Bug fixed on 2026-09-15: `HIT_GEOM_PREFIXES` only listed the playground
names, so on the inspection courses hits on slabs, beams, pallets, trucks
(`wood_plank_`), stair risers and ramp curbs were never counted. Every hit
number recorded before that fix (expert tables, demo `hit_flags`, the
SmolVLA evaluations of rounds 1 and 2) is too low; success flags are right.
Corrected expert hits on the short routes, seed 7: quarry 0; warehouse,
dock 1; tank farm, substation, tunnel 2; boiler, rubble 3; solar 4;
pipe alley 10 (light touches at the conduit mouth; the entry is sensitive to
the approach angle and three route variants did not improve it).

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
| hurdle_lane (drill) | 45 m lane | 14 beams 0.10-0.20 m; tour = there and back, 42 jumps |
| trench_field (drill) | four 20 m lanes, snake | 12 trenches 0.30-0.50 m; tour = the same snake |
| box_steps (drill) | 33 m lane | 5 low walls + 5 gaps; tour = there and back, 30 jumps |

The three drills exist to get jump examples: a pass gives 36 jumps against
about 5 on a normal course. Rule from building them: the first obstacle
after a 90 degree corner needs 5 m; the turn swings the ball almost 1 m off
the lane.

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

`scripts/data/generate_inspection_demos.py --episodes 30 --workers 10 [--seed-offset N]`
(no chase video; 10 minutes for 300 episodes). Per macro step: policy camera
256x256, 13-D state, skill class, params, the oracle's reason text, hit flag.
Jitter per episode: start pose, camera pose, course seed. Only episodes that
reach the goal are kept.

| Run | Seeds | Kept | Frames |
|---|---|---|---|
| `storage_local/20260914_1612__local__generate_inspection_demos__30ep/` | 1000+ | 216 / 300 | 108k |
| `storage_local/20260914_2331__local__generate_inspection_demos__30ep/` | 2000+ | 217 / 300 | ~107k |
| `storage_local/20260915_0942__local__generate_inspection_demos__30ep/` (3 drills only) | 3000+ | running at handoff | ~2,500 jumps |

Keep rates by course (both runs alike): tank farm 30, substation, rubble,
boiler, quarry 27-30, pipe alley 20, solar 18-20, warehouse 16-17,
tunnel 10-12, loading dock 8. Failures are stalls; the dock ramp and the
tunnel pipes are sensitive to the start jitter.

`scripts/data/export_demo_videos.py --demos <run dir> --per-course 2` writes
the policy camera of demo episodes as H.264 with a skill HUD (what the VLA
sees). Chase-camera videos of the same tours: run the expert with
`run_inspection_oracle.py --tour`.

`scripts/data/convert_inspection_demos_to_lerobot.py --demos <run dirs...> [--oversample 4 --window 10]`
(LeRobot venv) writes a LeRobotDataset: `observation.image`,
`observation.state` (13), `action` (10 = one-hot skill + params), `task`,
`reason`. Argmax of the first six action entries gives the skill. With
`--oversample N` every switch into a rare skill (jump, gap, pipe, rough)
becomes a short extra episode written N-1 more times: rare decisions are
about 1 % of the frames and this is the only class weighting the LeRobot
trainer allows without patching it.

## 6. SmolVLA fine-tuning (round 1) and what it showed

Environment: `/home/storage_group/envs/lerobot` (lerobot 0.4.4, torch 2.10,
MuJoCo pinned to 3.8.1: with MuJoCo 3.13 the expert itself failed three
courses, so the skills are tied to 3.8.1; `ops/setup_lerobot_env.sh` pins it).

Training (70 min on the A100 for 20k steps, batch 32):

```bash
lerobot-train --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=roboball/inspection_tours --dataset.root=<lerobot dir> \
  --rename_map='{"observation.image": "observation.images.camera1"}' \
  --batch_size=32 --steps=20000 --save_freq=5000 --output_dir=<run>/train --wandb.enable=false
```

Round 1: `storage_local/20260914_1639__local__train_smolvla__inspection_tours_216ep/`
(216 episodes, no oversampling; loss 0.63 -> 0.026; the 20k checkpoint is
under `train/checkpoints/020000/pretrained_model`).

Closed loop (`scripts/vla/eval_smolvla_inspection.py --checkpoint ... [--tour] [--video] --replan K`,
LeRobot venv; same courses, same hit metric; videos at 25 fps with the
policy camera as an inset; `agreement` = fraction of steps where the policy's
skill equals the expert's on the same state):

| Route | Re-plan | Goals | Clean | Agreement |
|---|---|---|---|---|
| short | every 5 steps | 3/10 | 1 | 95 % |
| short | every step | 5/10 | 1 | 96 % |
| tour | every 5 steps | 3/10 | 2 | 96 % |

The expert reaches 10/10 on the same seed. The policy picks the right skill
class almost always; it misses the moment of the rare ones (warehouse: never
jumps the beam; boiler: fails at the stairs; tunnel: leaves crawl mode early;
rubble: mistimes the crack). Re-planning every step (one forward pass per
macro step, ~3.7x slower evaluation) is clearly better than chunk execution.

## 7. Round 2: more demos, rare windows oversampled

Chain: `ops/round2_smolvla_chain.sh`. Dataset
`storage_local/20260915_0002__local__lerobot_dataset__inspection_tours_x2_os4/lerobot`
(433 full episodes, 221k frames, plus 7,371 short rare-window episodes, 153k
frames, from `--oversample 4`; the conversion took two hours because every
short episode is its own video file). Training:
`storage_local/20260915_0203__local__train_smolvla__inspection_tours_x2_os4/`
(20k steps, 70 min). Evaluation (re-plan every step, short routes, seed 7):
`storage_local/20260915_0313__local__eval_smolvla_inspection__short__playground_parkour_skills/`.

| Policy | Goals | Clean | Agreement |
|---|---|---|---|
| round 1, re-plan 1 | 5/10 | 1 | 96 % |
| round 2, re-plan 1 | 6/10 | 2 | 98 % |
| expert | 10/10 | 5 | - |

Round 2 passes pipe alley, tank farm, substation, boiler house, solar farm,
quarry. Still failing: warehouse (rolls into the beam, never jumps),
loading dock (stalls at the ramp), tunnel (leaves crawl mode early),
rubble (mistimes the crack). More data and oversampling helped a little;
the failure mode is unchanged: the moment of a rare skill.

Round 3 plan (not started): convert all three demo runs (10 courses x 2 +
the 3 drills) with `--oversample 4`, train 20k steps, evaluate with
`--replan 1 --video` on all 13 courses. The drills add about 2,500 jump
decisions, the thing the policy gets wrong. Other levers after that: the
10k/15k checkpoints; a smaller action chunk (`--policy.chunk_size 10`);
a "distance to next obstacle" state feature (the expert uses exactly that);
hybrid mode (the VLA picks speed and power too).

Also fixed on 2026-09-15: a tour that passes the goal position early no
longer ends there (`SkillArbitrationEnv` gates the low-level goal
termination on `path_dist_remaining < 1.0` for monotonic routes).

## 8. Earlier VLA / RL results (playground, before the inspection courses)

- BC on the clean playground demos: 100 % clean success on 3 seeds.
- PPO fine-tuning of that policy (`scripts/vla/train_rl_skills_vla.py`,
  plain PyTorch, KL to BC, frozen ResNet): keeps 100 %, no gain. With a
  weak KL it forgot the box jump; with the hit penalty on the old (accidental)
  jumps it got stuck. RL has nothing to fix while BC is already clean.

## 9. Storage layout

Every experiment folder is `storage_local/<YYYYMMDD_HHMM>__local__<script>__<tag>/`.
All VLA scripts default to a new timestamped run dir. Exceptions kept on
purpose: `storage_local/_assets` (blog scripts), `storage_local/sci_out`
(SLURM logs).

## 10. Working tree at handoff

Uncommitted and mine (evening): the hit-prefix fix in `mujoco_env.py`,
the goal gating in `skill_arbitration_env.py`, the three drill courses and
the dock/pipe-alley tweaks in `inspection_scenarios.py`, the drill task
texts in `generate_inspection_demos.py`, plus (afternoon):
`scripts/vla/eval_smolvla_inspection.py`,
`scripts/data/export_demo_videos.py`, the `--oversample` converter,
`run_inspection_oracle.py` (`--seed=N`, `--no-video`), `ops/watch.sh`,
`ops/setup_lerobot_env.sh`, `skills/__init__.py` and `skills/runner.py`
(crawl_pipe entry-zone fix), `radial_sphere/scenario.py`.

Uncommitted and from another session (not reviewed here):
`configs/rl/wall_jump.yaml`, `demos/wall_jump/`, `demos/chimney/*`,
`skills/mid_level/shaft_climbing.py`, `skills/mid_level/__init__.py`,
`tests/test_skills.py`, `tests/test_zigzag_skill.py`.

The disk filled up once (shared machine); it was expanded to 495 GB.
Each SmolVLA checkpoint is 1.3 GB with its optimizer state.

## 11. Open

- Loading dock and utility tunnel keep rates (8/30, 10/30). Longer straight
  approaches or a smaller start jitter would help.
- Tanks render as short domes (pillars). No tall cylinder object exists yet.
- The hurdle and the wall sit at the physical limit; small changes flip them.
- `jump_to` (aimed hop) is unused: unreliable from a rolling start.
- Hybrid mode (the VLA also picks speed and power) is not wired for demos.
  Today one power serves every jump.
- SmolVLA reaches 6/10 short routes after round 2; the failures are the
  timing of rare skills, not class confusion. See section 7 for the levers.

## 12. Quick checks

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/run_tests.py                       # 190 tests, ~60 s
MUJOCO_GL=egl PYTHONPATH=. python scripts/vla/run_inspection_oracle.py --tour  # 10 tours with videos
MUJOCO_GL=egl PYTHONPATH=. python scratch/record_oracle_video.py             # playground oracle video
MUJOCO_GL=egl PYTHONPATH=. /home/storage_group/envs/lerobot/bin/python scripts/vla/eval_smolvla_inspection.py \
    --checkpoint storage_local/20260914_1639__local__train_smolvla__inspection_tours_216ep/train/checkpoints/020000/pretrained_model \
    --replan 1 --video inspection_tank_farm                                   # one course, ~2 min
```
