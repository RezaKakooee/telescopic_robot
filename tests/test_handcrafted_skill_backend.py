"""Behavioral checks for the alternative RL skill library and jump options."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from radial_sphere import handcrafted_skill_backend as H
from radial_sphere.config import load_config
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv


class OptionTests(unittest.TestCase):
    def option(self, name="jump_up", ground=0., dt=.01):
        env = SimpleNamespace(
            data=SimpleNamespace(qpos=np.array([0., 0., ground + .18, 1., 0., 0., 0.]),
                                 qvel=np.zeros(6)),
            model=SimpleNamespace(opt=SimpleNamespace(timestep=dt)),
            action_repeat=1, sphere_radius=.15,
        )
        return H.SkillOption(env, name, {"vz_target": 2.6} if name == "jump_to" else {},
                             np.array([1., 0.]), 10, lambda xy: ground)

    def test_complete_jump_includes_crouch_burn_flight_and_landing_on_platform(self):
        for ground in (0., .24, .8):
            option = self.option(ground=ground)
            self.assertEqual(option._jump_phase(0), "crouch")
            self.assertEqual(option._jump_phase(20), "takeoff")
            option.env.data.qpos[2] = ground + .5
            option.env.data.qvel[2] = 1.
            self.assertEqual(option._jump_phase(32), "airborne")
            option.env.data.qvel[2] = -1.
            self.assertEqual(option._jump_phase(40), "airborne")
            option.env.data.qpos[2] = ground + .23
            self.assertEqual(option._jump_phase(50), "landing")
            option.env.data.qvel[2] = .5  # bouncing must not restart flight
            self.assertEqual(option._jump_phase(51), "landing")
            self.assertFalse(option.complete(60))
            self.assertFalse(option.complete(71))  # still bouncing: the next option must not fire mid-air
            option.env.data.qvel[2] = .1
            self.assertTrue(option.complete(71))

    def test_running_jump_and_changed_control_interval(self):
        option = self.option("jump_forward_while_moving", dt=.02)
        self.assertEqual(option._jump_phase(0), "sprint")
        self.assertEqual(option._jump_phase(28), "dip")
        self.assertEqual(option._jump_phase(32), "launch")
        self.assertEqual(option.max_steps, 120)

    def test_velocity_servo_ends_jump_to_burn_early(self):
        option = self.option("jump_to")
        self.assertEqual(option._jump_phase(20), "takeoff")
        option.env.data.qvel[2] = 2.7
        self.assertEqual(option._jump_phase(21), "airborne")
        option.env.data.qvel[2] = 1.
        self.assertEqual(option._jump_phase(22), "airborne")

    def test_hybrid_parameters_are_bounded_and_heading_is_relative(self):
        action = np.zeros(H.action_space("hybrid").shape)
        action[H.SKILL_NAMES.index("jump_to")] = 1
        action[-3:] = [10, -10, 10]
        _, name, params, heading = H.decode(action, "hybrid", np.pi / 2)
        self.assertEqual(name, "jump_to")
        np.testing.assert_allclose(heading, [-1, 0], atol=1e-7)
        self.assertAlmostEqual(params["vx_target"], .2)
        self.assertAlmostEqual(params["vz_target"], 3.)
        for invalid in (np.zeros(7), np.full(13, np.nan)):
            with self.assertRaises(ValueError):
                H.decode(invalid, "hybrid", 0.)


class BackendPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config("configs/rl/playground_parkour_skills.yaml")
        cls.env = SkillArbitrationEnv(cls.cfg, scenario=generate_scenario("playground", cls.cfg, seed=100))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def action(self, name):
        action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        action[self.env.skill_names.index(name)] = 1
        return action

    def test_every_registered_option_runs_real_physics_in_both_modes(self):
        for mode in ("macro", "hybrid"):
            self.env.action_mode = mode
            self.env.action_space = H.action_space(mode)
            for name in self.env.skill_names:
                with self.subTest(mode=mode, name=name):
                    self.env.reset(seed=100)
                    obs, reward, _, _, info = self.env.step(self.action(name))
                    self.assertTrue(self.env.observation_space.contains(obs))
                    self.assertTrue(np.isfinite(reward))
                    self.assertEqual(info["skill_name"], name)
                    self.assertEqual(info["skill_backend"], "skills")
                    if name in H.TIMED_JUMPS:
                        # From the spawn nothing is within the probe's reach:
                        # the self-timed jump has no target and costs one
                        # macro step of rolling, not a jump into open floor.
                        self.assertEqual(info["skill_result"], "no_target")
                        self.assertEqual(info["skill_control_steps"], self.env.k)
                    elif name in H.JUMPS:
                        self.assertGreater(info["skill_control_steps"], self.env.k)
                        self.assertEqual(info["skill_phase"], "landing")
                        self.assertFalse(info["skill_timed_out"])

    def test_timed_jump_approaches_the_hurdle_and_lands_past_it(self):
        """Armed 1.5 m early, the running jump rolls up to the hurdle, fires at
        the calibrated distance and reports success on landing."""
        self.env.action_mode = "macro"
        self.env.action_space = H.action_space("macro")
        self.env.reset(seed=100)
        hurdle_x = float(self.env.scenario.steps[0][0]) if getattr(self.env.scenario, "steps", None) is not None else None
        for _ in range(60):
            if float(self.env.env.data.qpos[0]) > 2.9:
                break
            self.env.step(self.action("move"))
        x0 = float(self.env.env.data.qpos[0])
        _, _, _, _, info = self.env.step(self.action("jump_forward_while_moving"))
        self.assertIsNotNone(info["skill_plan"], "the hurdle must be a target from here")
        self.assertGreater(info["skill_plan"]["edge_dist"], 0.9)
        self.assertEqual(info["skill_result"], "success")
        self.assertGreater(info["skill_control_steps"], self.env.k)
        self.assertGreater(float(self.env.env.data.qpos[0]), x0 + info["skill_plan"]["edge_dist"],
                           "the ball must come down past the edge it aimed at")
        self.assertEqual(info["obstacle_hit"], 0)

    def test_option_stops_immediately_on_low_level_termination(self):
        self.env.reset(seed=100)
        callback = []
        self.env.on_control_step = lambda env: callback.append(env.control_steps)
        try:
            with patch.object(self.env.env, "step", return_value=(None, 0., True, False, {})) as step:
                _, _, terminated, _, info = self.env.step(self.action("jump_up"))
            self.assertTrue(terminated)
            self.assertEqual(step.call_count, 1)
            self.assertEqual(info["skill_control_steps"], 1)
            self.assertEqual(callback, [1])
        finally:
            self.env.on_control_step = None

    def test_long_option_consumes_episode_time_budget(self):
        self.env.reset(seed=100)
        old_max = self.env.max_steps
        self.env.max_steps = 2
        try:
            _, _, _, truncated, info = self.env.step(self.action("jump_up"))
            self.assertTrue(truncated)
            self.assertEqual(info["skill_control_steps"], 2 * self.env.k)
        finally:
            self.env.max_steps = old_max


if __name__ == "__main__":
    unittest.main()
