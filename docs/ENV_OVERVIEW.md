# RadialSphere — Environment Overview

## The robot

A core sphere (radius 0.15 m) with 60 telescopic bars placed on a Fibonacci
sphere. Each bar is a slide joint driven by a position actuator:

- **sleeve** — a guide tube ending flush with the ball surface,
- **rod** — slides through the sleeve (0 … 0.12 m extension),
- **foot** — a rounded cap on the tip; only the feet and the core collide.

Each bar has its own color, so extension is visible in videos. A retracted
bar sinks into the ball; a fully extended bar reaches out almost one ball
radius. Generated as MJCF by `radial_sphere/mjcf.py`; physics runs on the
RoboVerse metasim MuJoCo handler.

## Locomotion

The scripted controller (`radial_sphere/controller.py`) scores every bar by
how backward-facing (push) and downward-facing (stance) it is, then min-max
normalises the scores so the full extension stroke is used each step. Given a
desired direction, the sphere rolls that way.

## Tasks (`radial_sphere/scenario.py`)

| kind | description |
|---|---|
| `path` | follow a sinusoidal path to its end |
| `roundtrip` | sine out, semicircle turn, return lane back — ball returns toward the camera |
| `goal` | reach a random goal point |
| `obstacle` | reach a random goal; heavy pillars block the straight line (2 always on it) |
| `maze` | level 1 fixed corridor or level 3 random perfect maze with dead ends |

`goal`/`obstacle` and maze level 3 re-randomise every reset when `randomize`
is on. Level 3 uses constant-count fixed wall pieces, 16-ray lidar, and a
geodesic reward; it does not expose its solution route through `path_pts`.
Scenarios serialise to JSON via `scenario_generator.py`.

## Low-level env (`RadialSphereEnv`, `RadialSphere-v0`)

- **Action** (60): normalised extension target per bar, [-1, 1].
- **Observation** (73): quat (4) + lin vel (3) + ang vel (3) + bar
  extensions (60) + goal direction (2) + goal distance (1).
- **Reward**: `progress_coef * (prev_dist - dist)` + `success` bonus only on
  physical MuJoCo contact between the goal and the robot core/feet. Geodesic
  distance is shaping only. Termination on contact; truncation at `max_steps`.
- The chase camera is optional (`camera.enabled` / `enable_camera=False`);
  rendering dominates step time (~3 steps/s with, ~176 steps/s without).

## High-level env (`SteeringEnv`)

RL plans, the scripted controller executes:

- **Action** (3): desired direction in the goal frame (2) + drive (1).
  Held for `rl.decision_every` low-level steps.
- **Observation** (7 + 3·K): velocity in the goal frame, yaw rate, goal
  distance, last command, and the K nearest pillars (offset + surface gap).
- **Reward**: low-level reward summed over the hold.

Trained with SB3 PPO (`scripts/rl/train_rl.py`): VecNormalize obs, camera
off, task re-randomised per episode; wandb (project `telescopic_robot`) +
tensorboard logging. Evaluation (`scripts/rl/eval_rl.py`) loads the
checkpoint + `vecnormalize.pkl` and records videos.

Level 3 preset:

```bash
python scripts/rl/train_rl.py -cn maze_level3 --kind maze
python scripts/heuristic/heuristic_agent.py -cn maze_level3 --kind maze
```

Fixed-layout curriculum with random start/goal points, resumed from a prior
checkpoint:

```bash
python scripts/rl/train_rl.py -cn maze_level3_fixed --kind maze \
  --resume storage_local/<prior-train-run>
```

## Baseline results (2026-08)

- Heuristic on `path` / `roundtrip` / `goal`: 100 % success.
- Heuristic on `obstacle`: 0 % — drives into a blocking pillar and stalls.
- PPO steering on `goal`: 100 % success after 100k decisions (easy task).
- PPO steering on `obstacle`: in progress.
