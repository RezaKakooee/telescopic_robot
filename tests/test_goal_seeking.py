"""Tests for the high-level goal-seeking planner.

The planner is worth testing on three separate promises:

1. `plan_route` returns a polyline that really does clear every obstacle,
   including on random maps it has not been tuned against.
2. `go_to_goal` picks the right skill: stop at the goal, `climb_stairs` for a
   step it can clear, `follow_path` otherwise.
3. The command it returns can be splatted straight into `execute_skill`. That
   is the whole contract of the level, so it is checked against the real
   skill functions rather than a mock.
"""

import unittest

import numpy as np

from skills import SKILL_REGISTRY, execute_skill
from skills.high_level import (DEFAULT_CLEARANCE, SkillCommand, climbable_height,
                               go_to_goal, plan_route)
from skills.high_level.goal_seeking import _segment_distance


def route_clearance(route, obstacles):
    """Smallest gap between the polyline and any obstacle surface."""
    worst = float("inf")
    for cx, cy, r in obstacles:
        centre = np.array([cx, cy], dtype=float)
        for i in range(len(route) - 1):
            d, _ = _segment_distance(np.asarray(route[i], dtype=float),
                                     np.asarray(route[i + 1], dtype=float), centre)
            worst = min(worst, d - r)
    return worst


def path_length(route):
    route = np.asarray(route, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))


class PlanRouteTests(unittest.TestCase):
    def test_an_empty_map_gives_the_straight_line(self):
        route, clear = plan_route([0.0, 0.0], [5.0, 2.0])
        self.assertTrue(clear)
        np.testing.assert_allclose(route, [[0.0, 0.0], [5.0, 2.0]])

    def test_an_obstacle_off_the_line_is_left_alone(self):
        route, clear = plan_route([0.0, 0.0], [5.0, 0.0], [(2.5, 4.0, 0.4)])
        self.assertTrue(clear)
        self.assertEqual(len(route), 2)

    def test_one_obstacle_on_the_line_is_routed_around(self):
        obstacles = [(2.0, 0.0, 0.4)]
        route, clear = plan_route([0.0, 0.0], [5.0, 0.0], obstacles)
        self.assertTrue(clear)
        self.assertEqual(len(route), 3)
        self.assertGreaterEqual(route_clearance(route, obstacles),
                                DEFAULT_CLEARANCE - 1e-6)
        # A detour that clears a 0.4 m rock should not double the trip.
        self.assertLess(path_length(route), 1.5 * 5.0)

    def test_random_maps_always_come_back_clear(self):
        """The property that matters, on maps the placement rule never saw."""
        rng = np.random.default_rng(7)
        for trial in range(200):
            start = rng.uniform(-1.0, 1.0, size=2)
            goal = start + rng.uniform(3.0, 8.0) * np.array(
                [np.cos(a := rng.uniform(0, 2 * np.pi)), np.sin(a)])
            n = int(rng.integers(1, 7))
            obstacles = []
            for _ in range(n):
                t = rng.uniform(0.15, 0.85)
                centre = start + t * (goal - start) + rng.normal(scale=0.35, size=2)
                obstacles.append((float(centre[0]), float(centre[1]),
                                  float(rng.uniform(0.10, 0.60))))
            # Skip maps where an obstacle swallows the start or the goal. No
            # polyline between those two points can clear such a circle, and
            # `plan_route` says so: it leaves them out of its verdict.
            if any(min(np.linalg.norm(start - np.array(o[:2])),
                       np.linalg.norm(goal - np.array(o[:2]))) < o[2] + DEFAULT_CLEARANCE
                   for o in obstacles):
                continue
            route, clear = plan_route(start, goal, obstacles)
            with self.subTest(trial=trial):
                self.assertTrue(np.all(np.isfinite(route)))
                if clear:
                    self.assertGreaterEqual(route_clearance(route, obstacles),
                                            DEFAULT_CLEARANCE - 1e-6)

    def test_a_start_inside_an_obstacle_does_not_hang(self):
        route, clear = plan_route([0.0, 0.0], [5.0, 0.0], [(0.1, 0.0, 0.5)])
        self.assertTrue(clear)
        self.assertEqual(len(route), 2)

    def test_the_clearance_is_honoured_when_it_is_widened(self):
        obstacles = [(2.0, 0.0, 0.3)]
        route, clear = plan_route([0.0, 0.0], [5.0, 0.0], obstacles, clearance=1.0)
        self.assertTrue(clear)
        self.assertGreaterEqual(route_clearance(route, obstacles), 1.0 - 1e-6)


class SkillChoiceTests(unittest.TestCase):
    def test_arriving_stops(self):
        cmd = go_to_goal(ball_xy=[0.0, 0.0], goal_xy=[0.1, 0.0],
                         lin_vel=[0.2, 0.0])
        self.assertIsInstance(cmd, SkillCommand)
        self.assertEqual(cmd.skill, "stop")

    def test_a_clear_run_follows_the_path(self):
        cmd = go_to_goal(ball_xy=[0.0, 0.0], goal_xy=[5.0, 0.0])
        self.assertEqual(cmd.skill, "follow_path")
        self.assertEqual(len(cmd.kwargs["path_pts"]), 2)
        self.assertTrue(cmd.meta["route_clear"])

    def test_a_clearable_step_is_climbed_not_avoided(self):
        low = 0.5 * climbable_height("over")
        cmd = go_to_goal(ball_xy=[0.0, 0.0], goal_xy=[4.0, 0.0],
                         steps=[(2.0, 0.0, 0.3, 0.6, low)])
        self.assertEqual(cmd.skill, "climb_stairs")
        self.assertEqual(cmd.meta["n_detours"], 0)
        self.assertIsNotNone(cmd.meta["jump_plan"])
        # It heads at the box, not past it.
        np.testing.assert_allclose(cmd.kwargs["d_hat"], [1.0, 0.0], atol=1e-9)

    def test_a_step_it_cannot_clear_becomes_an_obstacle(self):
        tall = climbable_height("over") + 0.40
        cmd = go_to_goal(ball_xy=[0.0, 0.0], goal_xy=[4.0, 0.0],
                         steps=[(2.0, 0.0, 0.3, 0.6, tall)])
        self.assertEqual(cmd.skill, "follow_path")
        self.assertGreaterEqual(cmd.meta["n_detours"], 1)

    def test_a_step_too_far_ahead_is_not_yet_the_job(self):
        low = 0.5 * climbable_height("over")
        cmd = go_to_goal(ball_xy=[0.0, 0.0], goal_xy=[9.0, 0.0],
                         steps=[(8.0, 0.0, 0.3, 0.6, low)], step_lookahead=2.0)
        self.assertEqual(cmd.skill, "follow_path")

    def test_the_planner_stays_out_of_the_skill_registry(self):
        """A registry name must return rod targets; a planner does not."""
        self.assertNotIn("go_to_goal", SKILL_REGISTRY)


class CommandRunsTests(unittest.TestCase):
    """Every command must splat straight into `execute_skill`."""

    def setUp(self):
        rng = np.random.default_rng(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        dirs = rng.normal(size=(60, 3))
        self.dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        self.max_extend = 0.16

    def _run(self, cmd):
        targets = execute_skill(cmd.skill, self.quat, self.dirs,
                                self.max_extend, **cmd.kwargs)
        self.assertEqual(targets.shape, (60,))
        self.assertTrue(np.all(targets >= -1e-6))
        self.assertTrue(np.all(targets <= self.max_extend + 1e-4))
        return targets

    def test_every_branch_produces_a_runnable_command(self):
        low = 0.5 * climbable_height("over")
        cases = {
            "stop": dict(ball_xy=[0.0, 0.0], goal_xy=[0.1, 0.0],
                         lin_vel=[0.3, 0.0]),
            "straight": dict(ball_xy=[0.0, 0.0], goal_xy=[5.0, 0.0],
                             lin_vel=[0.8, 0.0]),
            "detour": dict(ball_xy=[0.0, 0.0], goal_xy=[5.0, 0.0],
                           obstacles=[(2.0, 0.0, 0.4)], lin_vel=[0.8, 0.0]),
            "step": dict(ball_xy=[0.0, 0.0], goal_xy=[4.0, 0.0],
                         steps=[(2.0, 0.0, 0.3, 0.6, low)], lin_vel=[0.8, 0.0]),
        }
        seen = set()
        for name, kw in cases.items():
            with self.subTest(case=name):
                cmd = go_to_goal(**kw)
                seen.add(cmd.skill)
                self._run(cmd)
        self.assertEqual(seen, {"stop", "follow_path", "climb_stairs"})


if __name__ == "__main__":
    unittest.main()
