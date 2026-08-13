# configs

YAML configs (OmegaConf), grouped like `scripts/`:

- `rl/config.yaml` — the project default and single source of truth
  (robot, controller, scenarios, reward, camera, rl, video). The
  `radial_sphere` package loads it when `RADIAL_SPHERE_CONFIG` is not set.

Entry scripts accept trailing `key=value` overrides (OmegaConf dotlist):

    python scripts/rl/train_rl.py rl.n_envs=2 rl.total_steps=500000
    RADIAL_SPHERE_CONFIG=configs/rl/variant.yaml python scripts/rl/train_rl.py

Each run snapshots its RESOLVED config (with overrides applied) into
`storage_local/<run>/code/config.yaml` — that copy is what reproduces the run.
