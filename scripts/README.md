# scripts

Entry points, grouped by method family (library code stays in `radial_sphere/`):

- `rl/` — `train_rl.py` (PPO on the SteeringEnv), `eval_rl.py`
  (checkpoint → stats + videos).
- `heuristic/` — non-learning baselines: `heuristic_agent.py` (scripted
  look-ahead controller), `random_agent.py`.
- `env/` — `scenario_generator.py` (create + preview task scenarios as JSON).

All scripts run from the repo root:

    python scripts/heuristic/heuristic_agent.py kind=obstacle
    python scripts/rl/train_rl.py kind=obstacle steps=300000
    python scripts/rl/eval_rl.py run=storage_local/<run>
    python scripts/env/scenario_generator.py --kind goal --count 5

All accept `--config <yaml>` and trailing `key=value` config overrides.
Outputs land in `storage_local/<run>/` (videos, checkpoints, code snapshot).

## Demos

`run_demo.py` runs any demo declared in `configs/demos/`. Prefer adding a yaml
there over writing another runner: the shared plumbing (run dir, video panes,
overlay, metrics, expectations) lives once in `radial_sphere/demo.py`.

    python scripts/run_demo.py demo=gap
    python scripts/run_demo.py list=true
