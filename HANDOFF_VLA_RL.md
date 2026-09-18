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
| jump_maze (drill) | five 24 m lanes between walls, snake | one jump form per lane: rising beams and a beam pair; widening trenches and beam+gap; stairs up, a beam on the deck, stairs down, a riser and a ramp down; two 3 m platforms to land on; beams and a trench crossed at an angle. 27 jumps; tour = the same snake |
| doubling_boxes (drill) | 28 m lane | five 2.2 m boxes, heights 0.05, 0.10, 0.20, 0.40, 0.80 m, a 0.25 m pit between each pair, stairs down at the end. The only way on is a jump from each box top onto the next; on every deck the expert lands, brakes, backs up, settles, runs and jumps (4 jumps, 12/12 pass); tour = the same single pass |

The drills exist to get jump examples: a pass gives 27 to 36 jumps against
about 5 on a normal course. The first three each drill one form; the jump
maze (2026-09-15) has every form the expert knows, so one demo covers
beams, trenches, risers, platforms, pairs, combinations and angled
approaches. Rules from building them: the first obstacle after a 90 degree
corner needs 5 m (the turn swings the ball almost 1 m off the lane); 2 m
between two beams and 2.5 m between a beam and a trench (at 1.5 m the ball
lands on the second obstacle); a 0.6 m ramp down sends the ball into the
next corner at 3 m/s, 0.3 m with 4 m of run-out is fine. A 3 m platform is
landed on; a 1.5 m one is simply cleared, the running jump flies 3 m.
From the doubling boxes: the running jump is airborne for only about
0.7 m (the rest of the 1.1 m macro travel is the dip and the roll-out),
so a box-to-box pit must be 0.25 m or narrower; a pit before the first,
low box fails from the floor (the ball lands on the lip and stops) while
the same pit works from a box top, so the row starts with a curb; the
0.40 m step up succeeds about three times in four from a 3 m deck.

**Two fixes from the doubling boxes (2026-09-15 evening).** (1) The
heightmap (`map_perception._rasterize_scenario`) had no `steps`, so every
box top read as floor height and the jump option, which ends when the ball
is near the ground under it, never saw its landing on a deck: it coasted
for its whole 2.4 s (2.5 m instead of 1.4 m). Decks (boxes at least 0.8 m
across, `DECK_MIN_HALF_SIZE`) are in the map now; beams stay out because a
beam in the map would start the landing phase mid-flight. (2) The expert has
the platform routine on short decks (`SHORT_DECK` 2.6 m): after a jump
lands on one it brakes 4 macro steps, reverses to 0.7 m past the deck's
near edge, stands 3 steps, then runs and jumps from `DECK_GAP_WINDOW`
(0.30-0.50 m before the pit; the usual 0.25-0.45 from cruise). A box right
behind a pit no longer gets its own jump window: it fired half a metre too
early and put the ball into the face. With both, 2.2 m decks pass 12/12;
1.5 m decks only sometimes (the run-up is too short), 2.0 m never without
the routine. The 13 other courses give the same hit counts at seed 7 and
the maze stays 12/12. These are the `demos/platforms` ideas (back up for a
run-up) done at the macro level, so the demos record them as
`stop` / `reverse` decisions.

**Short decks are a different skill.** `demos/doubling_boxes` (config
`configs/rl/doubling_boxes.yaml`) hops up five 1.2 m decks and down four
the pillar way, a pit between every pair: stand at a planned point,
`jump_to` with a planned velocity, brake, creep in pulses, hop again
(`skills/mid_level/hop_planner`, long-stroke build). Found while building
it: the running-jump macros cannot do decks under 2 m (the flight plus
roll-out is 2.3 m); the standing hop needs a deck at least about 0.3 m
tall, else the 0.41 m rods stand on the floor beside it and the push is
weak (hops from 0.05 / 0.10 m decks abort); rises past about 0.45 m are
outside the hop calibration; so the decks are 0.45, 0.50, 0.60, 0.80,
1.20, 0.80, 0.60, 0.50, 0.45 m (the rises and drops double, the heights
cannot). Hops land up to 0.6 m off the planner's bracket, so 1.0 m decks
are at the scatter limit (1 of 5 starts made the way up) and 1.2 m is the
default. The creep cannot go below 1 m/s on this build (`gain_for_speed`
clamps to the measured curve), so the line-up is pulsed. Pits: 0.25 m up,
0.35 m down (wider than the 0.30 m core, so the ball must hop down; the
weakest hop cannot cross 0.45 m). Descent hops scatter about 1 m (longer
flight, strong cells only), so each lands about 85 % of the time and the
whole nine-hop course passes for 1 of 8 start orientations. With
`valley_down: 0.25` the descent is rolled off the lip (`fall_down`, 1.5 m/s
creep, line-up 0.5 m back) and 4 of 6 starts make the whole course.
This is a demo, not expert data: the macro env has no planned hop yet.

**`flip`, the seventh skill (2026-09-15 night).** There is no reverse
decision any more. The ball always drives "forward"; to back up it says
`flip` (the env brakes for that macro step and turns the travel direction
around), then `move`, then `flip` again to face forward. The expert's deck
routine and its stall retry both work this way now. Before, `reverse` was
recorded as `roll` with the heading param turned by 180 degrees, and the
evaluator only reads the skill class, so every learned back-up was played
back as a forward roll: 1,791 frames (0.67 %) in 101 of the 510 existing
episodes, all the expert's retries. Changes: `SkillArbitrationEnv.flipped`
and the `flip` option in `handcrafted_skill_backend.SKILL_NAMES` (appended,
old indices stay valid); `skills_vla.FlipSkill` (index 6, appended);
`generate_inspection_demos` records it; the LeRobot action is now 11-D
(7 one-hot + 4 params; the round 1 and 2 datasets and checkpoints are
10-D, and `eval_smolvla_inspection` reads the width off the action);
`convert_inspection_demos_to_lerobot` oversamples flips like the other
rare skills. Round 3 must be trained from the base model on a fresh
conversion. Turning is not a decision: on these courses the route tracker
sets the heading, so `roll` always follows the route; a turn action would
mean the policy takes over navigation, which is a later step.

**Floor seams (fixed 2026-09-15).** A course with trenches gets a floor of
box slabs. The old cut put every trench's x-edges across the whole arena,
so a trench in one lane left a seam under every other lane at that x, and
a running jump launched on a seam veered sideways and failed (jump maze:
3 of 3 seeds at the first beam; with the seams gone 12 of 12 episodes pass).
`_decompose_floor_slabs` now cuts around each hole along its long side, so
the seams stay at the trench's own lane edges. Trench field went from
18/30 kept to 12/12 in a check; the 13 other courses give the same hit
counts as before at seed 7.

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

Round 3 plan (not started): collect the jump maze and the doubling boxes
(`generate_inspection_demos.py --courses inspection_jump_maze inspection_doubling_boxes --episodes 30`,
27 and 4 jumps per pass; 12/12 each in a check), then convert all demo runs
(10 courses x 2 + the 5 drills) with `--oversample 4`, train 20k steps, evaluate with
`--replan 1 --video` on all 13 courses. The drills add about 2,500 jump
decisions, the thing the policy gets wrong. Other levers after that: the
10k/15k checkpoints; a smaller action chunk (`--policy.chunk_size 10`);
a "distance to next obstacle" state feature (the expert uses exactly that);
hybrid mode (the VLA picks speed and power too).

Also fixed on 2026-09-15: a tour that passes the goal position early no
longer ends there (`SkillArbitrationEnv` gates the low-level goal
termination on `path_dist_remaining < 1.0` for monotonic routes).

## 7b. Code as policy: the skills own their timing (2026-09-18)

The target architecture is in `docs/architecture.md`. Three layers: scripted
gaits and servos at the bottom, skills that own their own timing and their
own end in the middle, RL or a VLA choosing the next skill on top.

What changed:

- **The running jump times itself.** `radial_sphere/terrain_probe.py` rays
  the ground ahead along the heading and finds the next edge (a beam, a
  tread, a platform, a trench, a wall). `SkillOption` (in
  `handcrafted_skill_backend.py`) first approaches: it rolls along the
  route, re-measuring the edge every control step, and fires the jump
  when the edge is at the calibrated distance (beam 0.75 m, tread 0.50,
  platform 0.75, trench 0.40). With nothing to aim at the decision costs
  one macro step of rolling (`skill_result: no_target`). Flag
  `rl.self_timed_jumps` (default true).
- **The skill contract.** `skills_vla.VLASkill.can_start(terrain)` and
  `plan(terrain)`; the executor reports `info["skill_result"]` (success,
  timed_out, approach_timeout, no_target) and `info["skill_plan"]`.
- **The expert arms, it does not time.** `inspection_oracle.py` says
  "jump" once a station is within `ARM_DIST` 1.6 m. The old 0.2-0.3 m
  windows are gone. Any decision in the last 1.6 m is a right one: at
  1.1 m/s and 10 Hz that is 15 decisions instead of 2 or 3.
- Demos record `results` (the option result per step). New tools:
  `scripts/vla/trace_inspection_expert.py` (decisions step by step, with
  the plan and the result) and `scripts/vla/check_inspection_expert.py`
  (keep rates over jittered tour episodes, the collector's conditions).

Measured, expert, seed 7 short routes: 14 of 15 courses pass; hits fell on
most (boiler 3 -> 0, hurdle 2 -> 0, box steps 1 -> 0, maze 4 -> 2).
Pipe alley fails on that seed inside the conduit (the crawl, not a jump).

Jittered tour episodes (the collector's conditions), 8 per course, after
the review fixes:

| Course | Kept | Old keep rate (of 30) |
|---|---|---|
| tank farm, substation, boiler house, rubble, hurdle lane, trench field, box steps, jump maze, doubling boxes | 8/8 | 27-30, drills 18-30 |
| quarry | 7/8 | 27-30 |
| loading dock | 6/8 | 8 |
| solar farm | 6/8 | 18-20 |
| warehouse | 4/8 | 16-17 |
| pipe alley | 4/8 | 20 |
| utility tunnel | 4/12 | 10-12 |

The tunnel's cable cover moved from 0.4 m to 1.4 m past the branch
corner: no jump can be timed from a turn 0.35 m before the edge, and that
one spot was most of the tunnel's failures. Its failures now are in the
second conduit (the crawl). A/B on 24 episodes against the previous
commit, before the review fixes: warehouse 9 -> 12, solar 11 -> 12, box
steps 24 -> 22.

Review fixes (three reviewers, all findings closed): "success" now means
the ball came down past the edge it aimed at, else "short"; an edge under
the rods is rolled, not looked past (`MIN_TARGET` 0.35); a jump armed
while the ball still bounces from the last landing waits until it is
settled; a ramp is followed, not read as a chain of beams; a pit's far
wall is not a target; the shape and trigger refresh on every re-probe;
three ray lines (centre and shoulders); the ball's own floor is the
reference level (inside a pipe the first ray hits the roof); the stall
test needs no travel, not just no arc progress (a ball past a corner read
as stalled); the retry only for a station still ahead; a fast retry when
the station is under the rods; the collector draws the arm distance per
episode from 0.9-1.6 m, sets the jump class from the executed plan and
records `results`, `plan_shapes`, `plan_dists`; the converter writes a
`result` feature; the evaluator tallies results and keeps the expert's
flip state in step with the env.

The jump maze has a sixth lane: five 2.2 m decks, 0.10, 0.20, 0.40, 0.20,
0.10 m, a 0.25 m pit between each pair, jumped on with a pause on every
deck, up and down (a pit before a lower deck counts as a gap now).

Known limits: crawl_pipe and traverse_rough still run per macro step;
`no_target` jump frames keep the jump label (the `results` column says
what ran; the converter does not filter on it yet).

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
