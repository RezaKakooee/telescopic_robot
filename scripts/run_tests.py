"""Single entry point for the test suites.

Two things live under ``tests/`` and they run very differently.

* The fast suite. Everything ``unittest discover`` collects, including the
  script-style modules that opt in through ``tests/_function_suite.py``.
  Around twenty seconds. This is what runs by default.
* The long integration drivers. ``tests/test_skills.py`` walks the whole
  skill library through real MuJoCo runs and takes about ten minutes.
  ``tests/test_jump_height.py`` measures jump heights. Both are scripts with
  bare asserts and both are documented as commands you run by hand, so they
  stay scripts. Set ``all=true`` to run them too.

Usage::

    PYTHONPATH=. python scripts/run_tests.py             # fast suite
    PYTHONPATH=. python scripts/run_tests.py all=true    # + long drivers
    PYTHONPATH=. python scripts/run_tests.py list=true   # what would run
"""
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

from radial_sphere.config import script_config

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Long-running integration drivers. Scripts, not collected suites.
DRIVERS = (
    ("tests/test_skills.py", "whole skill library, real physics, ~10 min"),
    ("tests/test_jump_height.py", "jump height measurements"),
)


def discover():
    return unittest.defaultTestLoader.discover(
        start_dir=str(REPO_ROOT / "tests"), pattern="test_*.py",
        top_level_dir=str(REPO_ROOT),
    )


def run_fast(verbosity: int) -> bool:
    suite = discover()
    print(f"Fast suite: {suite.countTestCases()} tests")
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return result.wasSuccessful()


def run_driver(path: str) -> bool:
    print(f"\n{'=' * 70}\n  {path}\n{'=' * 70}", flush=True)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
    started = time.time()
    completed = subprocess.run([sys.executable, path], cwd=REPO_ROOT, env=env)
    print(f"  {path}: {'passed' if completed.returncode == 0 else 'FAILED'} "
          f"in {time.time() - started:.0f}s")
    return completed.returncode == 0


def main() -> int:
    args = script_config("run_tests")

    if args.list:
        flat = []
        stack = [discover()]
        while stack:
            item = stack.pop()
            if isinstance(item, unittest.TestSuite):
                stack.extend(reversed(list(item)))
            else:
                flat.append(str(item))
        print(f"Fast suite ({len(flat)} tests):")
        for name in flat:
            print(f"  {name}")
        print("\nLong drivers (run with all=true):")
        for path, note in DRIVERS:
            print(f"  {path}  -  {note}")
        return 0

    ok = run_fast(args.verbose)
    if args.all:
        for path, _ in DRIVERS:
            ok = run_driver(path) and ok
    else:
        print("\nSkipped the long integration drivers. Set all=true to include:")
        for path, note in DRIVERS:
            print(f"  {path}  -  {note}")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
