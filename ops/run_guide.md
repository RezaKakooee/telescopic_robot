# RadialSphere Ops Run Guide

How to run, monitor, and post-process experiments. All commands run from the
repo root, with the `roboverse` conda env:

```bash
cd ~/telescopic_robot
conda activate roboverse
```

## Install on a new server

```bash
git clone https://github.com/RezaKakooee/telescopic_robot.git && cd telescopic_robot
conda create -n roboverse python=3.10 -y && conda activate roboverse
pip install -r requirements.txt
export MUJOCO_GL=egl                     # or osmesa for headless video rendering
wandb login                              # or add rl.wandb=false to commands

# smoke test (no camera, fast), then a real run:
python scripts/rl/train_rl.py --kind maze --steps 2000 --n-envs 1 rl.wandb=false
sbatch ops/sb_train.sh train_rl "" --kind maze --steps 150000
```

The geodesic field needs no separate step — it is computed inside the maze
scenario generator automatically.

```
ops/
├── sb_train.sh        # SLURM training job (scicore CPU, 1 day) — main launcher
├── sb_train_gpu.sh    # same, on rtx4090 (use with rl.device=cuda)
└── run_guide.md       # this file
```

Python entry points live in `scripts/{rl,heuristic,env}`; configs in
`configs/rl/` (see the READMEs). Reusable logic lives in the
`radial_sphere` package.

## Train on the cluster

```bash
# default config (configs/rl/config.yaml), maze task
sbatch ops/sb_train.sh train_rl "" --kind maze

# more steps / other tasks
sbatch ops/sb_train.sh train_rl "" --kind maze --steps 300000
sbatch ops/sb_train.sh train_rl "" --kind obstacle
sbatch ops/sb_train.sh train_rl "" --kind goal

# a config variant (parallel sweeps = submit several of these)
sbatch ops/sb_train.sh train_rl configs/rl/my_variant.yaml

# variant + hydra-style overrides
sbatch ops/sb_train.sh train_rl "" --kind maze rl.n_steps=512 rl.lr=1e-4

# GPU (only useful for big nets / image obs — the current MLP is CPU-bound):
sbatch ops/sb_train_gpu.sh train_rl "" --kind maze rl.device=cuda
# or a100: sbatch --partition=a100 --qos=a100-1day --gres=gpu:1 ops/sb_train.sh ...
```

One run id (see `radial_sphere/run_id.py`) names three things identically:

- log:       `storage_local/sci_out/<run_id>.out`
- run dir:   `storage_local/<run_id>/`  (checkpoints, renders, tb, train.log,
             `code/` snapshot with the RESOLVED config, wandb/)
- wandb run: project `telescopic_robot` — same name

## Train locally (debug)

```bash
# login node has a 10 GB per-user memory cap: keep --n-envs 1
python scripts/rl/train_rl.py --kind maze --steps 50000 --n-envs 1 rl.wandb=false
```

## Monitor

```bash
squeue -u $USER -n radial                           # job states
tail -f storage_local/sci_out/<run_id>.out          # live log
grep -a "ep_rew_mean" storage_local/sci_out/<run_id>.out | tail   # learning curve
```

What to watch: `rollout/ep_rew_mean` should climb toward the task ceiling
(maze level 1: ~34 = 24.4 m geodesic progress + 10 success bonus;
obstacle/goal: ~13). `ep_len_mean` should fall as the policy gets faster.

## Evaluate a checkpoint

```bash
# uses checkpoints/final.zip + vecnormalize.pkl from the training run dir
python scripts/rl/eval_rl.py --run storage_local/<train run dir> --kind maze --episodes 3

# mid-training checkpoints work too (each ppo_<N>_steps.zip has a matching
# ppo_vecnormalize_<N>_steps.pkl saved next to it)
```

Videos land in the eval run dir under `renders/`. Warm-start / resume of
training is NOT implemented yet — if a job dies, evaluate its last
checkpoint or start a fresh run.

## Watch what was learned

```bash
# RL policy video (bird view sees the whole maze; config: camera.view)
python scripts/rl/eval_rl.py --run storage_local/<run> --kind maze

# scripted baseline for comparison (fails in maze/obstacle — expected)
python scripts/heuristic/heuristic_agent.py --kind maze

# scene previews without an episode (PNG per scenario)
python scripts/env/scenario_generator.py --kind maze
```

Video knobs in `configs/rl/config.yaml`: `video.frame_every` (lower = slower
motion), `video.fps`, `camera.view` (`chase` | `bird` | `bird_fixed`).

## Cameras and speed

The camera is rendered inside `get_states` every step → ~50x slower.
Training always disables it (`enable_camera=False`); only eval/heuristic
runs render. Rough rates: ~176 env-steps/s without camera, ~3/s with.

## When a node fails or times out

Checkpoints (+ their VecNormalize stats) save every `rl.checkpoint_every`
(50k) steps, so at most 50k steps are lost. The sbatch time limit is 1 day;
a 150k-decision maze run takes ~2 h on 6 envs.
