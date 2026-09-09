"""Run a skill demo declared in configs/demos/.

    python scripts/run_demo.py demo=gap
    python scripts/run_demo.py demo=gap video=false
    python scripts/run_demo.py list=true

Each demo is one yaml: the scenario, the skill, how long to run, which
cameras to record, and what counts as success. The logic lives once, in
`radial_sphere.demo`. See configs/demos/README.md for the spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from radial_sphere.config import script_config  # noqa: E402
from radial_sphere.demo import run_demo  # noqa: E402

DEMOS = Path(__file__).resolve().parent.parent / "configs" / "demos"


def load_spec(name: str):
    from omegaconf import OmegaConf
    path = DEMOS / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"no demo {name!r}; available: {', '.join(available())}")
    spec = OmegaConf.load(path)
    spec.name = spec.get("name", name)
    return spec


def available() -> list[str]:
    return sorted(p.stem for p in DEMOS.glob("*.yaml"))


def main() -> int:
    args = script_config("run_demo")
    if args.list:
        print("demos in configs/demos/:")
        for name in available():
            spec = load_spec(name)
            print(f"  {name:26s} {spec.get('description', '')}")
        return 0
    if not args.demo:
        raise SystemExit(f"pass demo=<name>. Available: {', '.join(available())}")

    names = available() if args.demo == "all" else [str(args.demo)]
    failed = []
    for name in names:
        spec = load_spec(name)
        if args.steps:
            spec.steps = int(args.steps)
        print(f"\n=== {name}: {spec.get('description', '')}")
        result = run_demo(spec, video=None if args.video is None else bool(args.video))
        for key, value in result.metrics.items():
            print(f"    {key:14s} {value:9.3f}")
        if result.video:
            print(f"    video          {result.video}")
        if result.passed:
            print(f"    -> pass")
        else:
            failed.append(name)
            for line in result.failures:
                print(f"    -> FAIL  {line}")
    if failed:
        print(f"\n{len(failed)} demo(s) failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
