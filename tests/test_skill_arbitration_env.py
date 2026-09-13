"""Regression tests for route rewards and hierarchical episode boundaries."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from radial_sphere.map_perception import WaypointTracker
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv


class WaypointProgressTests(unittest.TestCase):
    def test_progress_is_continuous_and_measured_in_metres(self):
        tracker = WaypointTracker([[0, 0], [1, 0], [9, 0], [9, 9]], [9, 9])
        idx, remaining = tracker.get_path_progress([0.2, 0, 0.3])
        self.assertEqual(idx, 0)
        self.assertAlmostEqual(remaining, 17.8)
        self.assertAlmostEqual(tracker.get_path_progress([0.3, 0, 2])[1], 17.7)
        self.assertAlmostEqual(tracker.get_path_progress([8, 0])[1], 10)
        self.assertAlmostEqual(tracker.get_path_progress([9, 1])[1], 8)
        self.assertAlmostEqual(tracker.get_path_progress([9, 9])[1], 0)

    def test_playground_rewards_every_leg_despite_goal_distance(self):
        scenario = generate_scenario('playground', None, seed=0)
        tracker = WaypointTracker(scenario.path_pts, scenario.goal)
        remaining = [tracker.get_path_progress(p)[1] for p in scenario.path_pts]
        self.assertTrue(np.all(np.diff(remaining) < 0))
        self.assertAlmostEqual(remaining[-1], 0)
        self.assertGreater(np.linalg.norm(scenario.path_pts[10] - scenario.goal),
                           np.linalg.norm(scenario.path_pts[0] - scenario.goal))

    def test_duplicate_points_and_single_point_are_finite(self):
        tracker = WaypointTracker([[0, 0], [0, 0], [2, 0]], [2, 0])
        self.assertAlmostEqual(tracker.get_path_progress([0.5, 0])[1], 1.5)
        tracker = WaypointTracker([[2, 0]], [2, 0])
        self.assertAlmostEqual(tracker.get_path_progress([0.5, 0])[1], 1.5)
        self.assertTrue(np.isfinite(tracker.get_guidance([2, 0])).all())


class ArbitrationRewardTests(unittest.TestCase):
    def setUp(self):
        self.scenario = generate_scenario('playground', None, seed=0)
        self.low = SimpleNamespace(
            cfg=SimpleNamespace(rl=SimpleNamespace(action_mode='macro'),
                                robot=SimpleNamespace(rod_mechanism='single_stage')),
            scenario=self.scenario,
            data=SimpleNamespace(qpos=np.array([0., 0., .3, 1., 0., 0., 0.]),
                                 qvel=np.zeros(6)),
            dirs_body=np.tile([0., 0., 1.], (60, 1)),
            max_extend=.16,
        )
        def reset(**kwargs):
            self.low.data.qpos[:3] = [0, 0, .3]
            self.low.data.qvel[:] = 0
            return None, {}
        self.low.reset = reset
        with patch('radial_sphere.skill_arbitration_env.MujocoRadialSphereEnv',
                   return_value=self.low):
            self.env = SkillArbitrationEnv(decision_every=10)
        self.env.reset()
        self.action = np.zeros(self.env.action_space.shape)

    def move(self, pos, vel=(0, 0), term=False, trunc=False, **info):
        def step(targets):
            self.low.data.qpos[:3] = pos
            self.low.data.qvel[:2] = vel
            return None, 0., term, trunc, info
        self.low.step = Mock(side_effect=step)
        return self.env.step(self.action)

    def test_forward_reward_and_backward_penalty_on_first_leg(self):
        _, reward, term, trunc, info = self.move([.2, 0, .3])
        self.assertAlmostEqual(reward, 3.99)
        self.assertAlmostEqual(info['path_progress'], .2)
        self.assertGreater(info['dist_to_goal'], 16.5)
        self.assertFalse(term or trunc)
        self.assertEqual(self.low.step.call_count, 10)
        _, reward, _, _, info = self.move([.1, 0, .3])
        self.assertAlmostEqual(reward, -2.01)
        self.assertAlmostEqual(info['path_progress'], -.1)

    def test_velocity_reward_follows_waypoint_not_goal(self):
        self.env.forward_incentive = True
        _, east_reward, _, _, _ = self.move([0, 0, .3], vel=(1, 0))
        self.env.reset()
        _, north_reward, _, _, _ = self.move([0, 0, .3], vel=(0, 1))
        self.assertGreater(east_reward, north_reward)
        self.assertAlmostEqual(north_reward, -.01)

    def test_reset_clears_previous_episode_progress(self):
        self.move([4, 0, .3])
        _, info = self.env.reset()
        self.assertGreater(info['path_dist_remaining'], 30)
        _, reward, _, _, _ = self.move([.2, 0, .3])
        self.assertAlmostEqual(reward, 3.99)

    def test_shaking_against_obstacle_does_not_reset_stall_deadline(self):
        self.env.no_progress_limit = 4
        self.move([4.28, 0, .2])
        for i in range(4):
            _, _, term, _, info = self.move([4.28 + .01 * (i % 2), .02 * (i % 2), .2], vel=(.5, .5))
            self.assertEqual(term, i == 3)
        self.assertTrue(info['stalled'])
        self.env.reset()
        self.assertEqual(self.env.no_progress_steps, 0)

    def test_small_continuous_progress_prevents_stall(self):
        self.env.no_progress_limit = 4
        for i in range(1, 10):
            _, _, term, _, info = self.move([i * .04, 0, .2])
            self.assertFalse(term or info['stalled'])

    def test_primitive_targets_follow_pose_at_each_control_step(self):
        poses = []
        def act(name, state, **kwargs):
            poses.append(state.quat.copy())
            return np.zeros(60)
        def step(targets):
            self.low.data.qpos[3:7] = [0., 0., 0., 1.]
            return None, 0., False, False, {}
        self.low.step = step
        with patch('radial_sphere.skill_arbitration_env.S.act', side_effect=act):
            self.env.step(self.action)
        self.assertEqual(len(poses), self.env.k)
        np.testing.assert_array_equal(poses[0], [1, 0, 0, 0])
        np.testing.assert_array_equal(poses[1], [0, 0, 0, 1])

    def test_macro_thrust_aims_rods_behind_waypoint(self):
        import skills_rl as skills
        action = np.full(self.env.action_space.shape, -1.)
        action[skills.SKILL_NAMES.index('thrust')] = 1.
        self.low.step = Mock(return_value=(None, 0., False, False, {}))
        with patch('radial_sphere.skill_arbitration_env.S.act', return_value=np.zeros(60)) as act:
            self.env.step(action)
        self.assertAlmostEqual(act.call_args.kwargs['azimuth'], np.pi)

    def test_airborne_escape_stops_at_first_substep(self):
        _, reward, term, trunc, info = self.move([-1.25, -.1, .62])
        self.assertTrue(term)
        self.assertFalse(trunc or info['success'])
        self.assertTrue(info['out_of_bounds'])
        self.assertEqual(self.low.step.call_count, 1)
        self.assertLess(reward, -20)

    def test_inner_courtyard_escape_does_not_reward_shortcut(self):
        self.env.stage_rewards = True
        _, reward, term, _, info = self.move([5, 5, 1.5])
        self.assertTrue(term and info['out_of_bounds'])
        self.assertEqual(info['path_progress'], 0)
        self.assertLess(reward, -20)

    def test_escape_cannot_collect_stage_bonus_or_goal_success(self):
        self.env.stage_rewards = True
        _, reward, term, _, info = self.move([1.21, 16.5, .6], term=True, success=True)
        self.assertTrue(term and info['out_of_bounds'])
        self.assertFalse(info['success'])
        self.assertEqual(self.env.cleared_milestones, set())
        self.assertLess(reward, -20)

    def test_jumps_and_turns_inside_course_remain_legal(self):
        for xy in [(0, 0), (9, 0), (9, 4.45), (9, 9), (4, 9),
                   (0, 9), (0, 14.2), (0, 16.5), (10, 10)]:
            with self.subTest(xy=xy):
                self.assertFalse(self.env._outside_playground(np.array([*xy, 2.])))

    def test_open_scenarios_do_not_use_playground_boundary_check(self):
        self.scenario.kind = 'campus'
        self.assertFalse(self.env._outside_playground(np.array([-2, 0, .3])))

    def test_low_level_end_flags_stop_substeps_and_propagate(self):
        for term, trunc in [(True, False), (False, True), (True, True)]:
            with self.subTest(term=term, trunc=trunc):
                self.env.reset()
                _, _, terminated, truncated, info = self.move(
                    [0, 0, .3], term=term, trunc=trunc, wall_contact=1)
                self.assertEqual(self.low.step.call_count, 1)
                self.assertEqual((terminated, truncated), (term, trunc))
                self.assertFalse(info['success'])
                self.assertEqual(info['wall_contact'], 1)

    def test_goal_contact_success_outside_wrapper_radius(self):
        _, _, terminated, _, info = self.move(
            [0, 15.9, .3], term=True, success=True)
        self.assertTrue(terminated)
        self.assertTrue(info['success'])
        self.assertEqual(self.low.step.call_count, 1)

    def test_wrapper_goal_pit_and_time_limit_still_end_episodes(self):
        for pos, limit, expected in [([0, 16.4, .3], 1500, (True, False, True)),
                                     ([0, 0, -.2], 1500, (True, False, False)),
                                     ([0, 0, .3], 1, (False, True, False))]:
            with self.subTest(pos=pos, limit=limit):
                self.env.reset()
                self.env.max_steps = limit
                _, _, term, trunc, info = self.move(pos)
                self.assertEqual((term, trunc, info['success']), expected)


if __name__ == '__main__':
    unittest.main()
