# Architecture: code as policy

This is the target design for the robot. Other engineers should follow it.
Every claim here matches the working tree on 2026-09-18.

Related files:

| File | What it holds |
|---|---|
| `HANDOFF_VLA_RL.md` | the courses, the demos, the SmolVLA results |
| `docs/rl_skill_backends.md` | the RL config for the `skills` backend |
| `skills/README.md` | the low-level skill API |

## The idea

- The policy picks a skill.
- The skill owns its timing and its end.
- The policy says "jump the next edge". It never says "jump now".
- RL and the VLA decide at skill boundaries.
- The timing problem leaves the policy. It moves into the skill.

## 1. The three layers

| Layer | Owner | Rate | Input | Output | Where in the repo |
|---|---|---|---|---|---|
| Low: gaits and servos | scripted code | every control step (10 ms) | robot state (`quat`, `dirs_body`, `max_extend`, velocity) plus the skill's kwargs (`d_hat`, `speed`, `phase`) | `(n_bars,)` rod targets in metres | `skills/low_level/` (`locomotion.py`, `jumping.py`, `pipe_crawling.py`, `terrain_following.py`), `skills/runner.py` (`skill_targets`) |
| Mid: skills with their own end | scripted phase machines | from the skill's start to its end (see the clock table) | terrain edges ahead (`TerrainProbe`), robot state, the route heading | rod targets every control step, then one result | `skills_vla/` (the contract), `radial_sphere/handcrafted_skill_backend.py` (`SkillOption`), `radial_sphere/terrain_probe.py`, `skills/mid_level/` |
| High: pick the next skill | PPO, SmolVLA, or the scripted expert | one decision per skill end | RL: local map patch, guidance, proprioception. VLA: camera image. Expert: arc position on the route and the course's stations | a skill name (hybrid mode adds heading offset, speed, power) | `radial_sphere/skill_arbitration_env.py` (executor), `radial_sphere/inspection_oracle.py` (expert), `scripts/rl/train_rl.py`, `scripts/vla/` |

The clocks:

| Clock | Value | Where |
|---|---|---|
| physics step | 2 ms | `radial_sphere/mujoco_mjcf.py`, `timestep` |
| control step | 10 ms (5 physics steps) | `MujocoRadialSphereEnv.action_repeat` |
| plain skill | 10 control steps (0.1 s) | `rl.decision_every` |
| jump approach | 1.0 to 4.0 s | `APPROACH_MIN_S`, `APPROACH_MAX_S` |
| running jump after the approach | 2.4 s | `SkillOption.__init__` |
| other jumps | 1.6 s | `SkillOption.__init__` |

Why these layers:

- Learned rod control gave speed. It also gave wall hits and no jumps.
- Scripted gaits are reliable and easy to test.
- A mid-level skill is one decision with one meaning.
- The high level is where the data and the reward live.

## 2. The skill contract

The contract is `VLASkill` in `skills_vla/base.py`. Abridged:

```python
@dataclass
class SkillResult:
    """Output from executing a single control substep of a skill."""

    targets: np.ndarray         # (n_bars,) float32 rod extension targets
    done: bool = False          # True if atomic multi-step sequence is complete
    info: dict[str, Any] = None # diagnostic metadata (phase, power, etc.)


class VLASkill(ABC):
    name: str = "base"
    params: tuple[ParamSpec, ...] = ()      # each scaled from [-1, 1]
    is_multi_step: bool = False

    @abstractmethod
    def act(self, state: RobotState, camera_heading: float = 0.0, **kwargs) -> np.ndarray:
        """Compute single-step rod extension targets (shape: (n_bars,))."""

    def step(self, state: RobotState, substep: int, camera_heading: float = 0.0, **kwargs) -> SkillResult:
        """Execute one control step of a multi-step sequence."""
        targets = self.act(state, camera_heading=camera_heading, **kwargs)
        return SkillResult(targets=targets, done=True)

    # The option contract. The policy chooses a skill; the skill owns its
    # own timing and its own end.
    def can_start(self, terrain) -> bool:
        """Is there something for this skill to do from here?"""
        return True

    def plan(self, terrain) -> dict:
        """What the skill intends, from the terrain ahead."""
        return {}
```

What each method promises:

| Method | Promise | Base answer | Jump answer (`jump_forward.py`, `jump_gap.py`) |
|---|---|---|---|
| `can_start(terrain)` | True when the skill has something to do here. `terrain` is the list of `Edge` ahead, or None. | `True` | `terrain is None or plan_jump(terrain) is not None` |
| `plan(terrain)` | What the skill intends. Empty when it needs no plan. | `{}` | `plan_jump(terrain) or {}`: the edge, its shape, the trigger distance |
| `step(state, substep, ...)` | Rod targets for one control step. `done` is True at the skill's end. | calls `act`, `done=True` | phase from the verified schedule; `done` at the budget, or in landing once settled |
| `act(state, ...)` | Rod targets for one step, without phase state. | abstract | the launch phase |

Notes:

- `can_start` and `plan` read terrain only. They never move the ball.
- `is_multi_step` is True for `jump_forward` and `jump_gap`.
- It is False for `roll`, `brake_stop`, `traverse_rough`, `crawl_pipe`, `flip`.
- `skills_vla.ENV_SKILL_MAP` maps each class to an env option name.

What the executor reports in `info` after each decision:

| Key | Meaning |
|---|---|
| `skill_name` | the decision, as decoded from the action |
| `skill_result` | `"success"`, `"short"`, `"timed_out"`, `"approach_timeout"`, `"too_close"`, `"interrupted"`, `"no_target"`, or None |
| `skill_plan` | `{"shape", "trigger", "edge_dist", "edge_change"}` at the decision, or None |
| `skill_phase` | the option's last phase |
| `skill_control_steps` | control steps the option ran |
| `skill_timed_out` | a jump used its whole budget without settling |
| `flipped` | the travel direction is reversed |

One honest note. The env runs `SkillOption`, not the `VLASkill` objects.
It calls `TerrainProbe.plan` for the jump plan.
That calls the same `plan_jump` as `VLASkill.plan`.
But `TerrainProbe.plan` adds two things.
It passes the ball's own floor height.
It reads three ray lines, not one.
`VLASkill.plan` passes one edge list and no floor.
The two must stay in step.

## 3. What the executor does with a plan

The executor is `SkillArbitrationEnv.step()` in `radial_sphere/skill_arbitration_env.py`.
The option is `SkillOption` in `radial_sphere/handcrafted_skill_backend.py`.

One decision, step by step:

1. Decode the action into a skill name, params and a heading.
2. The heading is the route's waypoint heading. Hybrid mode adds an offset.
3. `flip` becomes one macro step of `stop`. It toggles `env.flipped`.
4. `move` while flipped runs the `reverse` gait.
5. `jump_forward_while_moving` asks the probe for a plan. Only when `rl.self_timed_jumps` is true.
6. No plan means no target. The option becomes one macro step of `move`. The result is `"no_target"`.
7. With a plan, the option starts in phase `"approach"`.
8. The env runs `option.targets(substep)` until the option is complete.
9. `option.finish()` fixes the result.
10. The env writes the info keys from section 2. `skill_plan` is the plan at the decision.

The approach (`SkillOption._approach_step`):

- Each control step reads the route's live heading (`heading_fn`).
- `heading_fn` comes from the waypoint tracker.
- It drives the macro `move` gait along that heading.
- It holds the line through `line_origin` with a cross-track error.
- `line_origin` is where the option started. It never moves.
- Remaining distance is the edge distance minus progress along the heading.
- Every control step it probes again along the current heading.
- A fresh edge within 0.4 m of the estimate is the same edge.
- Then the whole plan is replaced: edge, shape and trigger.
- Odometry re-anchors on the fresh edge.
- `edge_xy` is the edge's position in the world. It moves with the fresh edge.
- The success or short check uses `edge_xy`.
- A lost edge is filled in by odometry alone.
- A frozen heading once drifted the ball into a post. That is why the heading is live.

The approach numbers:

| Name | Value | Meaning |
|---|---|---|
| `APPROACH_SPEED` | 1.1 m/s | macro `move` speed during the approach |
| `APPROACH_MIN_S`, `APPROACH_MAX_S` | 1.0 s, 4.0 s | bounds of the approach budget |
| budget formula | `(dist + 0.5) / (APPROACH_SPEED / 2)`, clipped | worst case: the ball starts from rest |
| `REPROBE_EVERY` | 1 control step | how often the edge is measured again |
| re-probe tolerance | 0.4 m | a fresh edge this near the estimate is the same edge |
| `RUNNING_SPEED` | 0.5 m/s | above this the sprint phase is skipped |

The trigger. Each control step checks, in this order:

| Order | Check | What happens |
|---|---|---|
| 1 | remaining under `MIN_TARGET` (0.35 m) | result `"too_close"`; the option brakes with `stop` and ends |
| 2 | the approach budget is used up | result `"approach_timeout"`; the option brakes with `stop` and ends |
| 3 | remaining at or under the trigger | the jump schedule starts |

The late-edge rule. It applies at the option's first control step only:

- The edge may already be inside the trigger at the decision.
- Then the jump fires at once only when the ball is settled.
- It also fires when the edge is more than 0.1 m inside the trigger.
- Else the ball rolls one control step. Then check 3 fires it.
- Settled: core within 0.05 m of the ground plus the radius. Vertical speed under 0.3 m/s.
- Why: a jump armed mid-bounce hit the next crate every time.

The results:

| Result | Set by | Meaning |
|---|---|---|
| `"success"` | `complete()` | landed and settled past the edge |
| `"short"` | `complete()` | landed and settled before the edge; it hit it or fell short |
| `"timed_out"` | `finish()` | the jump budget ran out mid-air or mid-bounce |
| `"approach_timeout"` | `_approach_step()` | the approach budget ran out; the option braked |
| `"too_close"` | `_approach_step()` | the edge got under the rods before the jump fired; the option braked |
| `"interrupted"` | `finish()` | the episode ended mid-option |
| `"no_target"` | the env | no edge to aim at; one macro step of `move` instead |
| None | | a plain skill; it has no result |

The past-the-edge check (`_landing_result`):

- Take the ball's position minus `edge_xy`.
- Project it on the heading.
- Above zero is `"success"`. Else `"short"`.
- A jump without a plan has no `edge_xy`. Settled means `"success"`.

The completion rule (`complete()`):

| Condition | Value |
|---|---|
| time after touchdown | at least 0.2 s |
| height | core within 0.10 m of the ground plus the radius |
| vertical speed | below 0.5 m/s |

The option also ends when:

| Event | Where |
|---|---|
| the low-level env terminates or truncates | `env.step` |
| the ball leaves the playground bounds | `_outside_playground` |
| the ball is within 0.50 m of the goal | `step()` loop |
| the ball falls below z = -0.15 m | `step()` loop |
| the episode's control-step budget is used | `max_steps * k` |

The config flag:

| Flag | Default | Effect |
|---|---|---|
| `rl.self_timed_jumps` (`configs/rl/config.yaml`) | `true` | the running jump plans, approaches and fires itself |
| set to `false` | | the old behaviour: the jump fires at once from where the ball is |

## 4. The terrain probe

The probe is `radial_sphere/terrain_probe.py`.

- Rays go down onto the model along the travel heading.
- They use the same ray groups as the rays under the ball.
- The core body is excluded from the rays.
- `TerrainProbe(env)` binds this to one env: `edges(heading, offset)` and `plan(heading)`.

| Name | Value | Why |
|---|---|---|
| `PROBE_RANGE` | 4.0 m | a 1.8 m tread seen from 1.6 m still shows its far end |
| `PROBE_STEP` | 0.05 m | the same spacing as the terrain rays under the ball |
| ray start | 3 m above the ball | above every wall and deck |

An `Edge` is the first place the ground steps up or down:

| Field | Meaning |
|---|---|
| `dist` | from the ball's centre to the edge, along the heading (m) |
| `kind` | `"rise"` or `"drop"` |
| `change` | height change at the edge, signed (m) |
| `length` | how far the new level runs before the next step; inf if not seen |
| `returns` | a drop that comes back up within `MAX_GAP`, to a level at most `MAX_DROP` lower |
| `level` | ground height before the edge (m) |

`edges_in` finds every step of at least `MIN_EDGE`. The edge sits half way between two ray samples.

The ramp rule in `edges_in`:

- A change can build up over many samples.
- One sample may differ from the last by under `SHARP_FRACTION` x `MIN_EDGE`.
- Then the ground is a ramp or a curve. It is not an edge.
- The probe follows the ramp. The ball rolls it.

The shapes (`Edge.shape`):

| Shape | Rule | Examples |
|---|---|---|
| `gap` | a drop that returns | trench, crack, pit before a deck |
| None | a drop that does not return | cliff, descending stair: rolled, not jumped |
| None | a rise above `MAX_JUMP_RISE` | a wall |
| `beam` | a rise with a top shorter than `BEAM_MAX_TOP` | hurdle, floor pipe, crate, pipe saddle |
| `riser` | a top between `BEAM_MAX_TOP` and `TREAD_MAX_TOP` | a stair tread |
| `platform` | a top longer than `TREAD_MAX_TOP` | a deck to land on and roll across |

The triggers. The jump fires this far before the edge:

| Shape | `TRIGGER` (m) | From |
|---|---|---|
| `beam` | 0.75 | the expert's old slab window, 0.55 to 0.85 m |
| `riser` | 0.50 | the old riser window, 0.38 to 0.55 m |
| `platform` | 0.75 | the same as a beam |
| `gap` | 0.40 | the old gap window, 0.25 to 0.45 m; 0.42 once fell into a 0.39 m gap |

These are the windows the expert used to fire by hand.
They were measured on the playground and the inspection courses.
Now they live in the skill.

The limits:

| Name | Value | Why |
|---|---|---|
| `MIN_EDGE` | 0.08 m | smaller steps are curbs; the ball rolls over them. Same as `inspection_oracle.MIN_JUMP_HEIGHT` |
| `SHARP_FRACTION` | 0.6 | a change slower than this share of `MIN_EDGE` per sample is a ramp |
| `MAX_JUMP_RISE` | 0.55 m | taller is a wall, not a target. The playground wall is 0.55 m; the clean limit is 0.60 m |
| `MAX_GAP` | 0.80 m | a drop that returns within this is a trench. A longer one is a lower level to roll down to. The measured clean gap is 0.65 m |
| `MAX_DROP` | 0.55 m | the far side may be this much lower and still count as a gap: a pit before a lower deck |
| `MIN_TARGET` | 0.35 m | nearer is too late. The launch travels about 0.3 m. Jumps fired at 0.2 m into a cable cover hit it every time |
| `BEAM_MAX_TOP` | 1.2 m | shorter tops are things to fly over |
| `TREAD_MAX_TOP` | 2.5 m | the courses' treads are 1.5 to 1.8 m; longer tops are platforms |

`plan_jump(edges, ground)` picks the target:

1. `ground` is the floor the ball stands on. Without it, the first sample's level is used.
2. An edge nearer than `MIN_TARGET` returns None. It is rolled, not looked past.
3. A wall returns None. It blocks the view.
4. A drop that does not return is skipped. The probe looks past it.
5. A rise whose base is below `ground` by `MIN_EDGE` returns None. It is the far wall of a pit or a lower level.
6. The first remaining edge is the target.
7. Return `{"edge", "shape", "trigger"}`.
8. Nothing found returns None.

`TerrainProbe.plan(heading)` runs `plan_jump` on three ray lines:

| Item | Value |
|---|---|
| `ground` | ball z minus the radius minus 0.03 m |
| why not the first sample | inside a pipe the first ray sample is the roof |
| ray lines | the centre, and one at each shoulder (the core radius to the side) |
| rises | the nearest rise of the three lines; a shoulder wins only when nearer by more than 0.10 m |
| drops | from the centre line only; that is where the ball's weight goes |
| no centre target, and a wall on the centre line | None; a shoulder seeing past the wall is not a target |

## 5. What the high level does now

The scripted expert is `InspectionOracle` in `radial_sphere/inspection_oracle.py`.
It stands in for RL or the VLA. It makes the demos.

The stations, built from the scenario:

| Station | Built from | Condition |
|---|---|---|
| `jump` | `steps` (slabs, beams, curbs, bund walls) | height at least `MIN_JUMP_HEIGHT` |
| `deck` | `steps` at least `DECK_MIN_HALF_SIZE` (0.4 m) across | shorter than `SHORT_DECK` (2.6 m) along the route |
| `gap` | `gaps` | the route crosses it, or passes within 0.25 m |
| `stair` | ascending `staircases` | one per riser, in the climbing direction |
| `pipe` | `pipes` | from `PIPE_APPROACH` (2.0 m) before the mouth to 0.3 m past the end |
| `rough` | `stones` | inside the field plus 0.3 m |

A slab within 0.4 m after a gap loses its own station. The gap jump clears both.

The decision order in `_select`:

1. At the goal: `stop`.
2. A queued retry step, if any.
3. The deck routine, when on a short deck.
4. Stalled on a deck: the deck routine again.
5. Stalled before a jump station: queue a retry.
6. A jump, gap or stair station in reach and in view: `jump_forward_while_moving`.
7. Inside a pipe station: `crawl_pipe`.
8. Inside a rough station: `traverse_rough_terrain`.
9. Else: `move`.

### Arm within `arm_dist`, no windows

| Name | Value | Meaning |
|---|---|---|
| `arm_dist` | constructor argument, default `ARM_DIST` = 1.6 m | say "jump" once the station's edge is this near |
| `ARM_MIN` | -0.1 m | and not once the ball is past it |
| `ARM_VIEW_DEG` | 35 degrees | and only when the station lies within this bearing of the route heading at the ball |
| `ARM_ALWAYS` | 0.5 m | nearer than this the bearing is noise; arm regardless |

Why the view test: the probe looks along the heading. A station round a corner is not on it yet.

The old windows against the new arming range:

| Station | Old window | Old width | Now |
|---|---|---|---|
| slab | 0.55 to 0.85 m | 0.30 m | -0.1 to 1.6 m |
| gap | 0.25 to 0.45 m | 0.20 m | -0.1 to 1.6 m |
| riser | 0.38 to 0.55 m | 0.17 m | -0.1 to 1.6 m |
| decisions inside, at 1.1 m/s and 10 Hz | | 2 to 3 | about 15 |

Why this is enough:

- Any decision inside the last 1.6 m is a right one.
- The skill measures the edge and fires at the trigger.
- The policy no longer needs to hit a window of 2 decisions.
- The VLA learned the skill class but missed the moment.
- A wide window makes the moment easy.

### Flip instead of reverse

- The ball always drives "forward".
- To back up, the expert says `flip`.
- The env brakes for one macro step. It turns the travel direction around.
- Then `move` runs the reverse gait.
- `flip` again faces forward.
- `_go(backwards)` issues one flip per change of direction.
- Why: the VLA reads only the skill class. `reverse` used to replay as a forward roll.

### The deck routine as a high-level sequence

`_deck_routine` runs on a short deck after a jump landing. A gap or jump must follow within 0.6 m.

| Phase | Decisions | Length |
|---|---|---|
| stop | `stop` | `DECK_STOP_STEPS` = 4 macro steps |
| back | `flip`, then `move` | until `DECK_REAR` = 0.7 m past the deck's near edge |
| settle | `flip`, then `stop` | `DECK_SETTLE_STEPS` = 3 macro steps |
| run | `jump_forward_while_moving` | at once: the next pit is within `arm_dist` |

Every step in it is a normal high-level decision. The demos record them as such.

Measured on the doubling boxes:

| Deck length | Result |
|---|---|
| 2.2 m | passes with the routine |
| 2.0 m | never without it |
| 1.5 m | only sometimes |

The stall retry:

| Item | Value |
|---|---|
| slow stall | 6 macro steps with under 0.05 m of arc progress AND under `STALL_XY` = 0.15 m of travel |
| fast stall | a station under the rods (`TOO_CLOSE` = 0.35 m ahead or less), the ball crept under 0.03 m in one step, after a jump decision |
| station range | still ahead or under the ball: `s_here <= s_end + 0.3`, and at most 1.0 m ahead |
| retry script | `flip` (if not flipped), `move` x 7, `flip`, `jump_forward_while_moving` |
| retry reason | names the station, for example `retry: jump slab 3 (0.16 m)` |

Why two stall tests: arc length alone read a ball rolling past a corner as stalled.
Why the fast one: pushing into a station for 6 steps was wasted time.

Tools:

| Script | What it does |
|---|---|
| `scripts/data/generate_inspection_demos.py` | records the demos (see below) |
| `scripts/data/convert_inspection_demos_to_lerobot.py` | writes the LeRobot dataset; adds a `result` string feature per frame (empty in old demos) |
| `scripts/vla/eval_smolvla_inspection.py` | runs SmolVLA closed loop; tallies `info["skill_result"]` per course; keeps the expert's flip state in step with the env |
| `scripts/vla/trace_inspection_expert.py` | prints the decisions step by step, with each plan and result; flags `--seed`, `--tour`, `--jitter`, `--probe`, `--all` |
| `scripts/vla/check_inspection_expert.py` | runs N jittered tour episodes per course and prints keep rates; flags `--episodes`, `--seed-offset`, `--workers`, `--short` |

What the collector records:

| Item | Value |
|---|---|
| `ARM_RANGE` | (0.9, 1.6) m, drawn per episode as `arm_dist` |
| jump class | from the executed plan: `gap` is `jump_gap`, else `jump_forward` |
| `results` | the option result per step |
| `plan_shapes` | what the jump aimed at per step, else empty |
| `plan_dists` | distance to that edge at the decision, else nan |
| attr `arm_dist` | the episode's arming distance |

## 6. Evidence

| Skill set | Policy | Where | Result |
|---|---|---|---|
| `skills_rl` primitives (drive, thrust, tuck, brake, brace, conform, stance), no memory | PPO, 800k steps | playground | 0 to 1 % success |
| `skills` macro options; jumps run their phases inside one decision | PPO, 200k steps | playground | 12 % success |
| `skills_vla` over macro options | BC on 25 clean expert demos | playground | 100 % clean success, 3 seeds |
| same | PPO fine-tune with KL to BC | playground | keeps 100 %, no gain |
| `skills_vla`, hand-timed expert | SmolVLA round 1, 216 episodes | 10 inspection courses, short routes | 5/10 goals, 1 clean, 96 % agreement |
| same | SmolVLA round 2, 433 episodes, rare windows x 4 | same | 6/10 goals, 2 clean, 98 % agreement |
| same | the expert itself | same | 10/10 goals, 5 clean |
| self-timed jump (this change) | the expert, seed 7 | 15 courses, short routes | 14/15 pass |

Hits before and after the self-timed jump (expert, seed 7, short routes):

| Course | Before | After |
|---|---|---|
| boiler house | 3 | 0 |
| hurdle lane | 2 | 0 |
| jump maze | 4 | 2 |

Jittered tours with the self-timed jump, 8 episodes per course, current code:

| Course | Kept | Old keep rate (of 30) |
|---|---|---|
| tank farm, substation, boiler house, rubble site, hurdle lane, trench field, box steps, jump maze, doubling boxes | 8/8 | |
| quarry | 7/8 | |
| loading dock | 6/8 | 8 |
| solar farm | 6/8 | 18 to 20 |
| warehouse | 4/8 | 16 to 17 |
| pipe alley | 4/8 | 20 |
| utility tunnel | 3/8 | 10 to 12 |

The utility tunnel number is from before the cable cover moved.
The cover sat 0.4 m past a corner. The course turned 0.35 m before it.
No jump can be timed from there. The cover now sits 1.4 m past the corner.

The jump maze has a sixth lane now:

| Lane | What it drills |
|---|---|
| 5 | five 1.6 x 1.2 m decks, 0.30, 0.50, 0.90, 0.50, 0.30 m high, 0.25 m pits between them |

The ball jumps onto each deck. It pauses on each (the deck routine). Up, then down.
The deck size is a config knob: `scenario.jump_maze.deck_length`, `deck_width`.

Measured deck sizes, 8 jittered tour episodes each:

| Length x width | Kept | Hits per episode |
|---|---|---|
| 2.2 x 1.2 m | 8 | 2 |
| 1.6 x 1.2 m (default) | 6 | 4 |
| 1.4 x 1.2 m | 5 | 11 |
| 1.2 x 1.2 m, 1.0 x 1.2 m | 0 | |
| 1.6 x 0.8 m | 5 | 15 |
| 1.6 x 0.6 m | 6 | 19 |

The running jump needs its run-up after the back-up. Below 1.4 m there is none.
Below 1.2 m wide the ball falls off the side.

The standing hop does not need a run-up. The option has a hop mode for it.
With under 0.8 m of run-up and a slow ball it plans an aimed hop (`jump_to`, `hop_planner`).
It runs only on the long-stroke build (0.26 m rods). On the standard build it cannot work:

| Standard build, from a standstill | Result |
|---|---|
| aimed hop, calibrated (`hop_calibration_standard.json`) | rise 0.20 to 0.33 m on a bad orientation; the planner finds no plan for any rise |
| aimed hop onto a 0.20 m deck from 0.55 m | 0 of 8, bounced off the face |
| full-power standing jump (`jump_forward_while_stopped`) | rise 0.47 to 0.69 m, but 0.4 to 1.9 m forward |
| full-power standing jump onto a 1 m deck | 3 of 8 at best |

Unit tests: 234 pass.

The numbers behind the timing problem:

| Fact | Value |
|---|---|
| old firing window | 0.2 to 0.3 m |
| cruise speed | 1.1 m/s |
| decision rate | 10 Hz |
| decisions inside the old window | 2 to 3 |
| rare skill frames in the demos | 1.6 % |

What it showed:

- Memoryless primitives cannot sequence a jump.
- Macro options help RL a little.
- BC over skills is perfect on the playground.
- The VLA picks the right skill class almost always.
- It misses the moment of the rare skills.
- So the moment moved into the skill.

## 7. What is not done yet

- `crawl_pipe` and `traverse_rough_terrain` still run per macro step. They have no end of their own.
- `roll`, `brake_stop` and `flip` are the same. Their result is None.
- `jump_gap` and `jump_forward` share one gait. `ENV_SKILL_MAP` sends both to `jump_forward_while_moving`.
- `jump_gap` wraps `jump_to` in `skills_vla`. The env never runs that.
- Only `jump_forward_while_moving` is in `TIMED_JUMPS`. The other three jumps fire at once.
- Turning is the route tracker's. The heading comes from the waypoints.
- The approach follows the waypoints too. A turn action would hand navigation to the policy. That is a later step.
- The env calls `TerrainProbe.plan`, not `VLASkill.plan`. The two share `plan_jump`. Only the env passes the floor height and three ray lines.
- Decks under 1.4 m need a standing hop. On the standard build no standing jump lands on one (section 6). The hop mode runs only on the long-stroke build; `demos/doubling_boxes` shows it.
- Warehouse, pipe alley and the utility tunnel keep under 5 of 8.
- SmolVLA round 3 has not been trained on self-timed demos.
- An LLM writing skill programs is a later step. It would sit on top of the same skills.

Next: collect demos with the self-timed jump and train SmolVLA round 3.
