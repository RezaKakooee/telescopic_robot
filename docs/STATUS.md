# Project status — what was done and where we are

*Last update: 2026-08-13. Written as a handover so work can continue on any
server without the old chat history.*

## What this project is

A telescopic sphere robot ("RadialSphere"): a ball with 60 bars that extend
and retract to make it roll. Physics: RoboVerse `metasim` + MuJoCo (CPU).
Control is two-level: RL picks the direction ("go 45° right"), a scripted
controller moves the 60 bars. See `docs/ENV_OVERVIEW.md` for the env details.

## Story so far (chronological)

1. **Modular refactor** — one package `radial_sphere/` with config-driven env.
2. **Visible telescoping** — bars rebuilt as sleeve/rod/foot units, per-bar
   colors, full-travel controller. Fixed a MuJoCo contact bug (tiny feet sank
   into the floor → explicit `solref/solimp` + `priority=1`).
3. **Scenarios** — `path` (sine), `roundtrip` (out and back toward the
   camera), `goal` (random target), `obstacle` (random pillars, 2 always
   block the straight line), `maze` (level 1: ~28.5 m serpentine corridor of
   thin fixed iron walls).
4. **RL steering layer** — `SteeringEnv`: action = direction (+drive) in the
   goal frame, held 10 low-level steps; obs = velocity/goal features +
   3 nearest pillars + 16 lidar rays. PPO (SB3) + VecNormalize.
5. **Reward fix for mazes** — distance to goal is now **geodesic** (through
   the walls' free space, Dijkstra field), not straight-line.
6. **Cameras** — `chase` (follows behind), `bird` (top-down, follows),
   `bird_fixed` (static, sees the whole maze; default for maze).
7. **Project layout** — mirrors `~/ant_swarm`: `configs/rl/config.yaml`
   (OmegaConf + Hydra, `-cn` variants, `key=value` overrides,
   `RADIAL_SPHERE_CONFIG` env var), `scripts/{heuristic,rl,env}/`,
   `ops/sb_train.sh` (+ `sb_train_gpu.sh`), shared run id
   (`radial_sphere/run_id.py`) across run dir / wandb / .out log,
   loguru logging with per-run `train.log`, wandb inside the run dir.

## Results

| task | heuristic (scripted) | PPO steering |
|---|---|---|
| path / roundtrip | 100 % success | (not needed) |
| goal (random target) | 100 % | 100 % after 100k decisions (easy) |
| obstacle (pillars) | **0 %** — stalls at a pillar | **solved**, reward ~13 |
| maze level 1 (corridor) | **0 %** — pushes into the wall under the goal | **solved**: reward 33.9 of ~34.4 max, goal in ~1800 of 4000 steps, eval 100 % |

## Where things are (key runs under storage_local/)

- Maze RL model: `radial__20260813_1132__20431881__train_rl/`
  (`checkpoints/final.zip` + `vecnormalize.pkl` — both needed for eval)
- Maze RL video: `radial__20260813_1351__local__eval_rl__maze/renders/`
- Heuristic stuck-in-maze video: `radial__20260813_1129__local__heuristic_agent__maze/renders/`
- Obstacle RL model: `radial_sphere__20260812_2305__20381975__rl_train__obstacle/`
- SLURM job logs: `storage_local/sci_out/<run id>.out`
- wandb project: `telescopic_robot` (run name = run id)

## How to run

    # scripted baseline (fails in the maze — expected)
    python scripts/heuristic/heuristic_agent.py --kind maze

    # train (SLURM; log → storage_local/sci_out/<run id>.out)
    sbatch ops/sb_train.sh train_rl "" --kind maze --steps 150000
    # GPU variant (rtx4090): sbatch ops/sb_train_gpu.sh ... rl.device=cuda

    # evaluate a trained run with videos
    python scripts/rl/eval_rl.py --run storage_local/<train run dir> --kind maze

    # any config value can be overridden:  key=value  (Hydra dotlist)
    python scripts/rl/train_rl.py --kind maze rl.total_steps=3e5 rl.n_envs=8

## New server setup

    git clone <repo>
    conda create -n roboverse python=3.10 && conda activate roboverse
    pip install -r requirements.txt
    export MUJOCO_GL=egl        # or osmesa for headless video rendering
    wandb login                 # optional

## Known facts and traps

- The camera makes stepping ~50x slower (renders 1280×720 inside
  `get_states`). Training must run with `enable_camera=False` (the training
  script does this).
- Old scicore login node: 10 GB per-user memory cap. Training there: only
  `--n-envs 1`. Use SLURM for real runs.
- metasim's `ObsSaver` buffers frames in RAM (OOM risk) — we use our
  streaming `radial_sphere.render.VideoRecorder` instead.
- A SLURM timeout once killed the final save; checkpoints now save their
  VecNormalize stats too (`save_vecnormalize=True`), so any checkpoint is
  evaluable.

## Next steps (agreed direction)

1. Maze **level 3 (random maze)** is implemented from
   `docs/env_mockups/maze_env_proposal.html`: constant-count fixed wall pieces
   are repositioned every episode, with dead ends, lidar, geodesic reward,
   and raw goal direction rather than a solution-route `path_pts` crutch.
   Level 2 (rooms) remains unimplemented.
2. Train level 3: on random mazes the policy cannot memorise, and
   the heuristic has no chance.
3. Possible extras: cross-track penalty for path tasks, end-to-end 60-bar
   PPO baseline for comparison, sparse-reward variant.
