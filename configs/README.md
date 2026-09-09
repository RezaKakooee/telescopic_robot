# configs

YAML configs (OmegaConf), grouped like `scripts/`:

- `rl/config.yaml` — the project default and single source of truth
  (robot, controller, scenarios, reward, camera, rl, video). The
  `radial_sphere` package loads it when `RADIAL_SPHERE_CONFIG` is not set.

- `scripts/<name>.yaml` — the knobs for the entry script of the same name.
  These replaced the old `argparse` flags, so every knob a script takes is
  now visible in one file instead of being buried in a parser.

## Running a script

There are no flags. Every knob is a `key=value` override:

    python scripts/skills/run_skill.py skill=move speed=0.6 steps=500
    python docs/blog/render_rough_terrain.py seconds=20 crf=30
    python scripts/run_tests.py all=true

`--help` prints the composed config, which is the complete knob list with its
current values. A key the script does not declare is an error, so a typo or a
stale flag name is reported rather than ignored.

The RL and heuristic entry scripts take both kinds of key at once: their own
knobs, and scenario overrides bound for the main config.

    python scripts/rl/train_rl.py seed=7 rl.n_steps=512

`seed` is a script knob; `rl.n_steps` is passed through to the scenario
config. They are told apart by whether the script's yaml declares the key.

Entry scripts accept trailing `key=value` overrides (OmegaConf dotlist):

    python scripts/rl/train_rl.py rl.n_envs=2 rl.total_steps=500000
    RADIAL_SPHERE_CONFIG=configs/rl/variant.yaml python scripts/rl/train_rl.py

Each run snapshots its RESOLVED config (with overrides applied) into
`storage_local/<run>/code/config.yaml` — that copy is what reproduces the run.
