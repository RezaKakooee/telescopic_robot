# telescopic_robot

A telescopic sphere robot ("RadialSphere") that rolls by extending and
retracting 60 bars. Built on RoboVerse/metasim + MuJoCo, exposed as a
Gymnasium env, with a scripted low-level controller and a high-level RL
steering policy (PPO).

## Layout

- `radial_sphere/` — the library: env, MJCF robot, scenarios, controller,
  steering wrapper, rendering. See `docs/ENV_OVERVIEW.md`.
- `configs/` — YAML configs (OmegaConf). `configs/rl/config.yaml` is the
  single source of truth; see `configs/README.md`.
- `scripts/` — entry points, grouped by family; see `scripts/README.md`.
- `ops/` — SLURM wrappers (`sbatch ops/sb_train.sh train_rl [config] [args]`);
  job logs land in `storage_local/sci_out/<run id>.out`.
- `docs/` — environment documentation.
- `notes/` — personal notes (vocabulary, overviews).
- `storage_local/` — all run outputs (gitignored): videos, checkpoints,
  code+config snapshots.

## Quick start

    # scripted baseline, one episode with video
    python scripts/heuristic/heuristic_agent.py

    # train the RL steering policy on the obstacle task
    python scripts/rl/train_rl.py --kind obstacle
    sbatch ops/sb_train.sh train_rl "" --kind obstacle  # on the cluster

    # evaluate a trained policy with videos
    python scripts/rl/eval_rl.py --run storage_local/<rl_train run dir> --kind obstacle

Any config value can be overridden on the command line (OmegaConf dotlist),
or a whole variant selected via the env var:

    python scripts/rl/train_rl.py rl.total_steps=500000 controller.back_gain=0.4
    RADIAL_SPHERE_CONFIG=configs/rl/variant.yaml python scripts/rl/train_rl.py
