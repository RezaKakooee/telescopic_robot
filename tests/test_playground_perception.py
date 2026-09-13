"""The navigation map must include the actual playground collision obstacles."""
import unittest

import numpy as np

from radial_sphere.config import load_config
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv


class PlaygroundPerceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_config('configs/rl/playground_explore.yaml')
        cls.env = SkillArbitrationEnv(cfg, scenario=generate_scenario('playground', cfg, seed=100))
        cls.env.reset(seed=100)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def height(self, xy):
        m = self.env.patch_extractor
        x, y = m._world_to_grid(*xy)
        return m.global_elevation[y, x]

    def test_pipe_and_parkour_boxes_are_visible(self):
        self.assertGreater(self.height((4.5, 0)), .19)
        for y in (2.8, 4.45, 6.1):
            self.assertAlmostEqual(self.height((9, y)), .24, places=3)

    def test_downsampling_preserves_thin_hurdle_from_approach(self):
        m = self.env.patch_extractor
        for x in (3., 3.3, 3.6, 3.9):
            patch = m.get_patch(np.array([x, 0., .2]))
            # Middle rows exclude guide walls; the crossbar must remain visible.
            self.assertGreater(float(patch[0, 6:10].max() + .2), .19)

    def test_downsampling_preserves_both_narrow_valleys(self):
        m = self.env.patch_extractor
        for y in (3.625, 3.675, 5.275, 5.325):
            with self.subTest(y=y):
                patch = m.get_patch(np.array([9., y, .18]))
                terrain = patch[0, 6:10, 7:9] + .18
                self.assertLess(float(terrain.min()), .05)
                self.assertGreater(float(terrain.max()), .22)

    def test_map_covers_finish_and_excludes_robot(self):
        self.assertGreater(self.env.patch_extractor.ymax, 17.5)
        self.assertGreater(self.height((0, 14.2)), .7)
        self.assertLess(self.height((0, 0)), .02)

    def test_calibration_matches_actuator_model(self):
        self.assertIsNone(self.env.env.hardware)
        self.assertEqual(self.env.profile, 'ideal')

    def test_macro_jump_sequence_clears_real_crossbar(self):
        import skills_rl as skills
        self.env.reset(seed=100)
        fired = False
        flight_steps = 0
        for _ in range(100):
            if not fired and self.env.env.data.qpos[0] >= 4.0:
                fired = True
                flight_steps = 6
            name = 'thrust' if flight_steps > 5 else 'tuck' if flight_steps else 'drive'
            if flight_steps:
                flight_steps -= 1
            action = np.full(self.env.action_space.shape, -1., dtype=np.float32)
            action[skills.SKILL_NAMES.index(name)] = 1.
            _, _, term, trunc, info = self.env.step(action)
            self.assertFalse(term or trunc, info)
            if self.env.env.data.qpos[0] > 5.2:
                break
        self.assertGreater(self.env.env.data.qpos[0], 5.2)

    def test_drive_only_stall_ends_instead_of_running_six_minutes(self):
        import skills_rl as skills
        self.env.reset(seed=100)
        action = np.full(self.env.action_space.shape, -1., dtype=np.float32)
        action[skills.SKILL_NAMES.index('drive')] = 1.
        for step in range(250):
            _, _, term, trunc, info = self.env.step(action)
            if term or trunc:
                break
        self.assertTrue(info['stalled'])
        self.assertFalse(info['success'] or info['out_of_bounds'])
        self.assertLess(step, 200)


class PlaygroundTrainingStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config('configs/rl/playground_parkour.yaml')
        cls.cfg.rl.training_start_probability = 1.0
        cls.env = SkillArbitrationEnv(
            cls.cfg, scenario=generate_scenario('playground', cls.cfg), training=True)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_box_and_valley_starts_are_settled_and_reset_progress(self):
        seen = set()
        for seed in range(20):
            obs, info = self.env.reset(seed=seed)
            index = info['training_start']
            seen.add(index)
            expected_xy = self.env.training_start_points[index]
            pos = self.env.env.data.qpos[:3]
            np.testing.assert_allclose(pos[:2], expected_xy, atol=.08)
            ground = .24 if index in (0, 2) else .004
            self.assertGreater(pos[2], ground + .12)
            self.assertLess(pos[2], ground + .25)
            self.assertFalse(self.env._outside_playground(pos))
            remaining = self.env.waypoint_tracker.get_path_progress(pos)[1]
            self.assertAlmostEqual(info['path_dist_remaining'], remaining)
            self.assertAlmostEqual(self.env.progress_anchor, remaining)
            self.assertEqual(self.env.no_progress_steps, 0)
            self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(seen, {0, 1, 2, 3})

    def test_evaluation_still_starts_at_original_spawn(self):
        env = SkillArbitrationEnv(self.cfg, scenario=generate_scenario('playground', self.cfg))
        try:
            _, info = env.reset(seed=100)
            self.assertIsNone(info['training_start'])
            np.testing.assert_allclose(env.env.data.qpos[:2], [0, 0], atol=.01)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
