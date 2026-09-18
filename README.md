# telescopic_robot

A telescopic sphere robot ("RadialSphere") that rolls by extending and
retracting 60 bars. Built on RoboVerse/metasim + MuJoCo, exposed as a
Gymnasium env, with a scripted low-level controller and a high-level RL
steering policy (PPO).

The target architecture (three layers, the skill contract) is [`docs/architecture.md`](docs/architecture.md).

## Layout

- `radial_sphere/` — the library: env, MJCF robot, scenarios, controller,
  steering wrapper, rendering. See `docs/ENV_OVERVIEW.md`.
- `configs/` — YAML configs (OmegaConf). `configs/rl/config.yaml` is the
  single source of truth; see `configs/README.md`.
- `skills/` — the skill library, grouped by what a skill decides:
  `low_level/` computes one behaviour, `mid_level/` chooses a low-level
  skill each step and delegates, `high_level/` is reserved for planning.
  See `skills/README.md`.
- `skills_vla/` — six policy-facing skills (bounded params, egocentric
  heading), each a thin wrapper over `skills`. Used by the VLA scripts.
- `demos/` — one folder per skill demo: `demo.yaml`, plus `runner.py`
  when the control flow is the point. `python scripts/run_demo.py
  list=true` shows them. See `demos/README.md`.
- `scripts/` — entry points, grouped by family; see `scripts/README.md`.
- `ops/` — SLURM wrappers (`sbatch ops/sb_train.sh train_rl [config] [args]`);
  job logs land in `storage_local/sci_out/<run id>.out`.
- `docs/` — environment documentation. `HANDOFF_VLA_RL.md` (repo root) covers
  the VLA work: the playground course, ten inspection courses, the generic
  expert, and demo collection for SFT.
- `notes/` — personal notes (vocabulary, overviews).
- `storage_local/` — all run outputs (gitignored): videos, checkpoints,
  code+config snapshots.

## Quick start

    # the scripted expert on the ten inspection courses, with videos
    MUJOCO_GL=egl PYTHONPATH=. python scripts/vla/run_inspection_oracle.py --tour

    # a fine-tuned SmolVLA policy on the same courses (LeRobot venv)
    MUJOCO_GL=egl PYTHONPATH=. /home/storage_group/envs/lerobot/bin/python \
        scripts/vla/eval_smolvla_inspection.py --checkpoint <run>/train/checkpoints/020000/pretrained_model --replan 1


    # scripted baseline, one episode with video
    python scripts/heuristic/heuristic_agent.py

    # train the RL steering policy on the obstacle task
    python scripts/rl/train_rl.py kind=obstacle
    sbatch ops/sb_train.sh train_rl "" --kind obstacle  # on the cluster

    # evaluate a trained policy with videos
    python scripts/rl/eval_rl.py run=storage_local/<rl_train run dir> kind=obstacle

Any config value can be overridden on the command line (OmegaConf dotlist),
or a whole variant selected via the env var:

    python scripts/rl/train_rl.py rl.total_steps=500000 controller.back_gain=0.4
    RADIAL_SPHERE_CONFIG=configs/rl/variant.yaml python scripts/rl/train_rl.py
