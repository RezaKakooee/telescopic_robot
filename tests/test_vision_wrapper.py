"""Unit and integration tests for RoboBallVisionWrapper and multimodal Gym spaces."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import unittest
import numpy as np

from radial_sphere.config import load_config_cli
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from radial_sphere.vision_wrapper import RoboBallVisionWrapper, make_vision_env


class VisionWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config_cli(name="playground_parkour_skills")
        cls.scenario = generate_scenario("playground", cls.cfg, seed=42)

    def test_front_camera_observation_space_and_reset(self):
        env = SkillArbitrationEnv(self.cfg, scenario=self.scenario, seed=42)
        wrapped = RoboBallVisionWrapper(
            env,
            camera_name="front",
            image_size=(128, 128),
            channels_first=False,
            state_type="kinematics",
            language_instruction="Follow the parkour corridor to the goal.",
        )

        obs, info = wrapped.reset(seed=42)

        # Check observation space structure
        self.assertIn("image", obs)
        self.assertIn("state", obs)
        self.assertEqual(obs["image"].shape, (128, 128, 3))
        self.assertEqual(obs["image"].dtype, np.uint8)
        self.assertEqual(obs["state"].shape, (11,))
        self.assertEqual(obs["state"].dtype, np.float32)

        # Observation space validation
        self.assertTrue(wrapped.observation_space.contains(obs))
        self.assertEqual(info["language_instruction"], "Follow the parkour corridor to the goal.")

        # Test step
        action = np.zeros(wrapped.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapped.step(action)
        self.assertEqual(obs["image"].shape, (128, 128, 3))
        self.assertEqual(obs["state"].shape, (11,))
        self.assertTrue(wrapped.observation_space.contains(obs))

        wrapped.close()

    def test_channels_first_and_birdview_camera(self):
        env = SkillArbitrationEnv(self.cfg, scenario=self.scenario, seed=42)
        wrapped = RoboBallVisionWrapper(
            env,
            camera_name="birdview",
            image_size=(96, 96),
            channels_first=True,
            state_type="both",
        )

        obs, info = wrapped.reset(seed=42)
        self.assertEqual(obs["image"].shape, (3, 96, 96))
        self.assertEqual(obs["state"].shape, (11,))
        self.assertIn("raw_state", obs)
        self.assertEqual(obs["raw_state"].shape, env.observation_space.shape)
        self.assertTrue(wrapped.observation_space.contains(obs))

        wrapped.close()

    def test_make_vision_env_helper(self):
        wrapped = make_vision_env(
            cfg=self.cfg,
            scenario=self.scenario,
            camera_name="front",
            image_size=(64, 64),
            seed=42,
        )
        obs, info = wrapped.reset(seed=42)
        self.assertEqual(obs["image"].shape, (64, 64, 3))
        self.assertEqual(obs["state"].shape, (11,))
        wrapped.close()


if __name__ == "__main__":
    unittest.main()
