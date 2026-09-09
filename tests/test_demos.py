"""Every demo in demos/ runs, and meets its own expectations.

Each demo's yaml carries an `expect` block: bounds on the metrics, taken from
a measured run. That makes the demo its own regression test, so a behaviour
change shows up here instead of in a video nobody watches.

These run real physics with video off, about six seconds each.
"""
import os
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
from omegaconf import OmegaConf

from radial_sphere.demo import run_demo
from skills import SKILL_REGISTRY

DEMOS = Path(__file__).resolve().parent.parent / "demos"
SPEC_NAMES = sorted(p.parent.name for p in DEMOS.glob("*/demo.yaml"))
#: Specs the shared runner can execute on its own. The rest have a
#: runner.py because their control flow is the point.
DECLARATIVE = [n for n in SPEC_NAMES
               if {"scenario", "skill"} <= set(OmegaConf.load(DEMOS / n / "demo.yaml"))]

#: Names the demo runner exposes to feedback bindings and overlay text.
LIVE_STATE = {
    "step", "time", "ball_x", "ball_y", "ball_z", "ball_xy", "vx", "vy", "vz",
    "lin_vel", "speed", "distance_x", "path_length", "goal_distance", "goal_x",
}
METRICS = {
    "final_x", "final_y", "final_z", "distance_x", "path_length", "mean_speed",
    "max_speed", "min_z", "max_z", "core_impacts", "goal_distance",
    "steps_run", "duration_s",
}


def load(name):
    spec = OmegaConf.load(DEMOS / name / "demo.yaml")
    spec.name = spec.get("name", name)
    return spec


class DemoSpecTests(unittest.TestCase):
    """Cheap checks on the yaml, so a typo fails fast instead of mid-run."""

    def test_there_are_demos(self):
        self.assertTrue(SPEC_NAMES, "demos/ has no specs")
        self.assertTrue(DECLARATIVE, "no demo can run without its own runner")

    def test_every_demo_has_a_runner_or_a_runnable_spec(self):
        """A folder with neither cannot be run at all."""
        for name in SPEC_NAMES:
            with self.subTest(demo=name):
                self.assertTrue((DEMOS / name / "runner.py").exists() or name in DECLARATIVE,
                                f"demos/{name}/ has no runner.py and no runnable spec")

    def test_each_spec_is_well_formed(self):
        for name in DECLARATIVE:
            with self.subTest(demo=name):
                spec = load(name)
                self.assertIn(str(spec.skill.name), SKILL_REGISTRY)
                self.assertGreater(int(spec.steps), 0)
                self.assertTrue(spec.get("description"), "give the demo a description")
                for key, source in (spec.skill.get("feedback") or {}).items():
                    self.assertIn(str(source), LIVE_STATE,
                                  f"{name}: feedback {key} reads unknown state {source}")
                bounded = list(spec.get("expect") or {})
                for group in [spec.get("stop_when") or {}] + list(spec.get("stop_when_any") or []):
                    bounded += list(group)
                for key in bounded:
                    self.assertIn(key, METRICS | LIVE_STATE, f"{name}: unknown key {key}")

    def test_every_demo_declares_expectations(self):
        """A demo without bounds is a video, not a check."""
        for name in DECLARATIVE:
            with self.subTest(demo=name):
                self.assertTrue(load(name).get("expect"),
                                "add an expect block, measured from a real run")


class DemoRunTests(unittest.TestCase):
    """Run each demo for real and hold it to its own bounds."""

    def test_demos_meet_their_expectations(self):
        for name in DECLARATIVE:
            with self.subTest(demo=name):
                result = run_demo(load(name), video=False, quiet=True)
                self.assertEqual(result.failures, [],
                                 f"{name}: " + "; ".join(result.failures))


if __name__ == "__main__":
    unittest.main()
