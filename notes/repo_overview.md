# telescopic_robot — Repo Overview

*Written 2026-08-12 (Claude Code walkthrough of the full source tree).*

## What this repo is

A custom reinforcement-learning environment for a **telescopic sphere robot**: a
rigid core sphere with 60 prismatic ("telescoping") bars pointing outward in a
Fibonacci-sphere arrangement. By extending back/bottom-facing bars and
retracting front-facing ones, the sphere pushes itself along the floor and
rolls. The task is either to follow a sinusoidal path to its end (`path`) or to
reach a random goal point (`goal`).

Physics runs in MuJoCo through **RoboVerse's `metasim`** handler (an external
dependency, not vendored in this repo), and the environment is exposed as a
standard Gym/Gymnasium env registered as `RadialSphere-v0` (plus a classic
4-tuple `RadialSphere-v0-compat` variant).

## Architecture

Everything is driven by `config.yaml` — the single source of truth for robot
geometry, controller gains, path shape, reward weights, camera, and episode
length. Nothing is hard-coded in the package.

Data flow:

```
config.yaml → mjcf (robot XML) → ScenarioCfg → metasim handler (MuJoCo)
            → ActionModel / ObservationModel / RewardModel / Renderer
```

### Package modules (`radial_sphere/`)

| Module | Role |
|---|---|
| `config.py` | Loads `config.yaml` into a nested `SimpleNamespace`. |
| `geometry.py` | Pure math: Fibonacci sphere, sinusoidal path sampling, quaternion → rotation matrix. No sim/config deps. |
| `mjcf.py` | Generates the robot MJCF programmatically: one core body; per bar a non-colliding sleeve capsule + inner capsule on a slide joint (range `[0, max_extend]`) with a position actuator **named after its joint** (RoboVerse's mujoco handler looks actuators up by joint name). No freejoint — the handler adds one when `fix_base_link=False`. |
| `scenario.py` | Serialisable `Scenario` dataclass (kind, spawn, goal, waypoints, markers, path length) with `path`/`goal` generators and JSON save/load. |
| `action.py` | Action space `[-1, 1]^n_bars`; `decode` maps to metre targets (`dof_pos_target` dict), `encode` is the inverse (used to feed the scripted controller through the same interface). |
| `observation.py` | Flat float32 vector, 13 + n_bars dims: quat (4) + lin vel (3) + ang vel (3) + normalized bar extensions (n) + goal direction (2) + normalized goal distance (1). |
| `reward.py` | `progress_coef * (prev_dist − dist)` dense shaping + sparse `success` bonus when within `goal_eps` (0.45 m) of the goal. |
| `controller.py` | Scripted expert: `desired_direction` (look-ahead path tracker → unit xy direction) and `bar_targets` (scores each bar by backward-facing "push" and downward-facing "stance" components in world frame). |
| `radial_sphere.py` | `RadialSphereEnv`: builds MJCF, wraps in RoboVerse `ScenarioCfg` (chase/bird camera, red breadcrumb markers, green goal sphere), settles under gravity for `n_settle_steps` on reset, tracks the ball each step by writing MuJoCo's `cam_pos` directly. Also `GymCompatWrapper` / `make_compat_env`. |
| `render.py` | TensorState camera output → uint8 `(H, W, 3)` frame. |
| `snapshot.py` | Timestamped run dirs under `storage_local/` (named with the SLURM job ID or `local`) + code/config snapshot for reproducibility. |
| `_gym.py` | gym/gymnasium dual-import shim. |

### Root entry points

- `heuristic_agent.py` — scripted look-ahead controller fed through the
  normalized action interface via `ActionModel.encode`. Deterministic baseline.
- `random_agent.py` — uniform random actions; sanity check.
- `scenario_generator.py` — writes scenario JSONs (+ optional preview PNGs)
  under a run dir; agents consume them via `--scenario <path>.json`.

All scripts log per-episode return/steps/success and save MP4s via metasim's
`ObsSaver`. Outputs land in
`storage_local/radial_sphere__<YYYYMMDD_HHMM>__<jobid|local>__<tag>/`
(`renders/`, `code/`, `scenarios/`, `previews/`).

## Current state / caveats (as of 2026-08-12)

- **A large modular refactor is uncommitted.** Last commit
  (`a30ec15 "initial refactoring"`, 2026-06-04) predates most of the tree:
  the old monolithic root `radial_sphere.py` demo is deleted (staged), and the
  new modules (`action`, `observation`, `reward`, `render`, `scenario`,
  `snapshot`, `_gym`), the three root scripts, and `config.yaml` are all
  **untracked**. Losing this working tree loses the refactor.
- Minor staleness: `config.yaml` comments still reference the deleted `run()`
  demo in `radial_sphere.py`; the `demo:` section it served looks vestigial.
  `README.md` is just the repo name.
- The system Python (`/usr/bin/python`, 3.10) does **not** have `metasim`
  installed — runs happen in some other environment (conda/Apptainer on the
  scicore cluster; June run dirs carry SLURM job IDs).
- `nohup.out` records failed attempts to run a nonexistent `render_success.py`.
- `storage_local/`, `roboverse_data/`, and build artefacts are gitignored.
