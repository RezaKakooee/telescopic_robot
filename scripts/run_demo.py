"""Run a skill demo from demos/.

    python scripts/run_demo.py demo=gap
    python scripts/run_demo.py demo=all declarative=true video=false
    python scripts/run_demo.py list=true

Each demo is one folder holding everything about showing one skill:
`demo.yaml` and, when the control flow is the point, its own `runner.py`.

A demo with a runner is executed by that runner. A demo without one is
executed declaratively by `radial_sphere.demo.run_demo`, which reads the
same yaml. Pass `declarative=true` to force the declarative path for a
demo that has both; that is what `tests/test_demos.py` checks against.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from radial_sphere.config import script_config  # noqa: E402
from radial_sphere.demo import run_demo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMOS = ROOT / "demos"


def available() -> list[str]:
    return sorted(p.parent.name for p in DEMOS.glob("*/demo.yaml"))


def load_spec(name: str):
    from omegaconf import OmegaConf

    path = DEMOS / name / "demo.yaml"
    if not path.exists():
        raise SystemExit(f"no demo {name!r}; available: {', '.join(available())}")
    spec = OmegaConf.load(path)
    spec.name = spec.get("name", name)
    return spec


def runner_for(name: str) -> Path | None:
    path = DEMOS / name / "runner.py"
    return path if path.exists() else None


def is_declarative(spec) -> bool:
    """A spec the shared runner can execute on its own."""
    return "scenario" in spec and "skill" in spec


def main() -> int:
    args = script_config("run_demo")
    if args.list:
        print("demos in demos/:")
        for name in available():
            spec = load_spec(name)
            marks = []
            if runner_for(name):
                marks.append("runner")
            if is_declarative(spec):
                marks.append("declarative")
            print(f"  {name:24s} [{', '.join(marks)}]  {spec.get('description', '')}")
        return 0
    if not args.demo:
        raise SystemExit(f"pass demo=<name>. Available: {', '.join(available())}")

    names = available() if args.demo == "all" else [str(args.demo)]
    failed = []
    for name in names:
        spec = load_spec(name)
        runner = runner_for(name)
        print(f"\n=== {name}: {spec.get('description', '')}")

        if runner is not None and not args.declarative:
            extra = [] if args.video is None else [f"knobs.video={bool(args.video)}"]
            done = subprocess.run([sys.executable, str(runner), *extra], cwd=ROOT)
            if done.returncode:
                failed.append(name)
            continue

        if not is_declarative(spec):
            print("    -> skipped: needs its runner, and declarative was forced")
            continue
        if args.steps:
            spec.steps = int(args.steps)
        result = run_demo(spec, video=None if args.video is None else bool(args.video))
        for key, value in result.metrics.items():
            print(f"    {key:14s} {value:9.3f}")
        if result.video:
            print(f"    video          {result.video}")
        if result.passed:
            print("    -> pass")
        else:
            failed.append(name)
            for line in result.failures:
                print(f"    -> FAIL  {line}")

    if failed:
        print(f"\n{len(failed)} demo(s) failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
