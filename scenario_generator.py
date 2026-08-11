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
    python scenario_generator.py                      # default kind from config
    python scenario_generator.py --kind path
    python scenario_generator.py --kind goal --count 5
    python scenario_generator.py --kind goal --no-preview
"""
from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse

import numpy as np
import rootutils
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from radial_sphere import KINDS, generate_scenario, load_config, make_run_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Generate a scenario for the RadialSphere agent")
    p.add_argument("--kind", choices=KINDS, default=None,
                   help="task kind (default: config scenario.kind)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count", type=int, default=1, help="how many scenarios to generate")
    p.add_argument("--config", default=None,
                   help="path to a config.yaml (default: project config.yaml)")
    p.add_argument("--no-preview", dest="preview", action="store_false",
                   help="skip rendering preview PNGs (faster; no sim build)")
    args = p.parse_args()

    cfg = load_config(args.config)
    kind = args.kind or getattr(cfg.scenario, "kind", "path")

    run_dir = make_run_dir(f"scenario__{kind}")
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
