"""Scenario generator — create a task for the agent to do.

Two kinds (for now):
  path : path navigation — follow the sinusoidal path to its end.
  goal : goal finding    — reach a single random goal point in the arena.

Generates one or more scenario specs (JSON) under a ``storage_local`` run dir,
plus an optional preview PNG of each scene (sphere at spawn, red breadcrumbs,
green goal).  An agent then *does* a generated scenario via its ``--scenario``:

    python scenario_generator.py --kind goal --seed 1
    python heuristic_agent.py --scenario storage_local/<run>/scenarios/goal.json
    python random_agent.py    --scenario storage_local/<run>/scenarios/goal.json

Usage:
    python scripts/env/scenario_generator.py                   # kind from config
    python scripts/env/scenario_generator.py --kind obstacle --count 5
    python scripts/env/scenario_generator.py --kind goal --no-preview
"""
from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import numpy as np
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from radial_sphere import (KINDS, build_run_id, generate_scenario,  # noqa: E402
                           load_config_cli, make_run_dir, setup_logging)
from radial_sphere.config import script_config  # noqa: E402

setup_logging()


def main():
    args = script_config("scenario_generator", passthrough=True)

    cfg = load_config_cli(path=args.config, name=args.config_name,
                          overrides=args.scenario_overrides)
    kind = args.kind or getattr(cfg.scenario, "kind", "path")
    if kind not in KINDS:
        raise SystemExit(f"unknown kind {kind!r}; expected one of {list(KINDS)}")

    run_dir = make_run_dir(build_run_id("scenario_generator", tag=kind))
    setup_logging(run_dir)
    sc_dir = run_dir / "scenarios"
    sc_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    for i in range(args.count):
        name = f"{kind}_{i:02d}" if args.count > 1 else kind
        sc = generate_scenario(kind, cfg, seed=args.seed + i, name=name)
        path = sc.save(sc_dir / f"{name}.json")
        scenarios.append(sc)
        log.info(f"generated {kind!r} scenario → {path}  "
                 f"spawn={np.round(sc.spawn_xy, 2).tolist()}  "
                 f"goal={np.round(sc.goal, 2).tolist()}  path_len={sc.path_length:.2f}")

    if args.preview:
        import imageio.v2 as iio

        from radial_sphere import RadialSphereEnv
        prev_dir = run_dir / "previews"
        prev_dir.mkdir(parents=True, exist_ok=True)
        for sc in scenarios:
            env = RadialSphereEnv(cfg, scenario=sc, output_dir=run_dir, seed=args.seed)
            env.reset()
            frame = env.render()
            if frame is not None:
                out = prev_dir / f"{sc.name}.png"
                iio.imwrite(out, frame)
                log.info(f"preview → {out}")
            env.close()

    log.info(f"Run dir : {run_dir}")


if __name__ == "__main__":
    main()
