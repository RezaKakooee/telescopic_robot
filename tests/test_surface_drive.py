"""Behaviour checks for `surface_drive`, the gait re-aimed at any surface.

This skill had no test of its own. It was reached only by the registry sweep
in `test_skill_interface.py`, which checks the calling convention and nothing
about what the skill does.

Each check here pins one claim the docstring makes.
"""
import os
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

from radial_sphere.geometry import fibonacci_sphere
from skills.locomotion import move, surface_drive

MAX_EXTEND = 0.16
MIN_OFFSET = 0.025
FLOOR = np.array([0.0, 0.0, -1.0])


def _quats():
    rng = np.random.default_rng(5)
    out = [np.array([1.0, 0.0, 0.0, 0.0])]
    for row in rng.normal(size=(4, 4)):
        out.append(row / np.linalg.norm(row))
    return out


class SurfaceDriveTests(unittest.TestCase):
    def setUp(self):
        self.dirs = fibonacci_sphere(60).astype(np.float32)
        self.headings = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                         np.array([-0.6, 0.8])]

    def test_pointing_at_the_floor_reproduces_move_exactly(self):
        """The docstring's central claim, and the reason the skill is trusted."""
        for quat in _quats():
            for heading in self.headings:
                for speed in (0.6, 1.2, 2.0):
                    rolling = move(quat, self.dirs, MAX_EXTEND, heading, speed=speed)
                    surface = surface_drive(
                        quat, self.dirs, MAX_EXTEND, FLOOR,
                        np.array([heading[0], heading[1], 0.0]), speed=speed)
                    np.testing.assert_allclose(
                        surface, rolling, atol=1e-9,
                        err_msg=f"diverged at speed={speed} heading={heading}")

    def test_travel_direction_is_projected_into_the_surface(self):
        """`along` may be passed loosely; its normal component is removed."""
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        clean = surface_drive(quat, self.dirs, MAX_EXTEND, FLOOR,
                              np.array([1.0, 0.0, 0.0]))
        with_climb = surface_drive(quat, self.dirs, MAX_EXTEND, FLOOR,
                                   np.array([1.0, 0.0, -0.7]))
        np.testing.assert_allclose(with_climb, clean, atol=1e-9)

    def test_reach_cap_is_a_hard_ceiling_per_rod(self):
        """Without the cap a rod commanded past the surface keeps pushing."""
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        free = surface_drive(quat, self.dirs, MAX_EXTEND, FLOOR,
                             np.array([1.0, 0.0, 0.0]), speed=2.0)
        self.assertGreater(free.max(), 0.05, "expected a real push to cap")
        cap = np.full(60, 0.04)
        capped = surface_drive(quat, self.dirs, MAX_EXTEND, FLOOR,
                               np.array([1.0, 0.0, 0.0]), speed=2.0,
                               reach_cap=cap)
        self.assertLessEqual(capped.max(), 0.04 + 1e-9)

    def test_a_wall_normal_moves_the_push_off_the_floor_rods(self):
        """Re-aiming the normal must re-aim which rods carry the wave."""
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        floor = surface_drive(quat, self.dirs, MAX_EXTEND, FLOOR,
                              np.array([1.0, 0.0, 0.0]), speed=1.2)
        wall = surface_drive(quat, self.dirs, MAX_EXTEND,
                             np.array([0.0, 1.0, 0.0]),
                             np.array([1.0, 0.0, 0.0]), speed=1.2)
        self.assertGreater(float(np.max(np.abs(floor - wall))), 0.02)
        # The rods pushing at a wall must be the ones pointing at it.
        pushing = wall > MIN_OFFSET + 1e-6
        self.assertTrue(np.all(self.dirs[pushing][:, 1] > -0.1),
                        "a rod pushing away from the wall would throw the ball off")

    def test_output_stays_inside_the_stroke(self):
        for quat in _quats():
            out = surface_drive(quat, self.dirs, MAX_EXTEND,
                                np.array([0.3, -0.6, -0.74]),
                                np.array([1.0, 0.5, 0.0]), speed=1.6)
            self.assertEqual(out.shape, (60,))
            self.assertTrue(np.all(out >= 0.0) and np.all(out <= MAX_EXTEND))


if __name__ == "__main__":
    unittest.main()
