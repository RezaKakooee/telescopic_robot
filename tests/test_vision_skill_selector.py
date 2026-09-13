import json
import unittest

import numpy as np

from radial_sphere.handcrafted_skill_backend import SKILL_NAMES
from radial_sphere.vision_skill_selector import (
    VisionSkillSelector, execution_feedback, make_prompt, parse_skill, skill_action,
)


class VisionSkillTests(unittest.TestCase):
    def test_every_skill_maps_to_correct_executor_action(self):
        for index, name in enumerate(SKILL_NAMES):
            self.assertEqual(parse_skill(json.dumps({"skill": name})), name)
            self.assertEqual(int(np.argmax(skill_action(name))), index)
        self.assertEqual(parse_skill('```json\n{"skill":"stop"}\n```'), "stop")

    def test_invalid_commands_cannot_select_other_skills_or_parameters(self):
        invalid = ['{"skill":"climb_stairs"}', '{"skill":"move","speed":100}',
                   '{"skill":["move","jump_up"]}', '["move"]', 'move',
                   '{"skill":"move"} then jump', '{"skill":null}']
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_skill(response)

    def test_invalid_output_brakes_and_records_error_without_teacher_fallback(self):
        selector = VisionSkillSelector(lambda images, text: '{"skill":"fly"}')
        selected = selector.select([np.zeros((8, 8, 3), np.uint8)], "Reach the finish", {})
        self.assertEqual(selected.skill, "stop")
        self.assertIsNotNone(selected.error)
        self.assertEqual(selected.raw_response, '{"skill":"fly"}')

    def test_predictor_receives_images_instruction_and_previous_feedback(self):
        frames = [np.zeros((8, 8, 3), np.uint8), np.ones((8, 8, 3), np.uint8)]
        def predict(images, text):
            self.assertIs(images, frames)
            self.assertIn("Stop before the pipe", text)
            self.assertIn('"skill_timed_out": true', text)
            self.assertIn("chronological order", text)
            return '{"skill":"stop"}'
        selector = VisionSkillSelector(predict)
        self.assertEqual(selector.select(frames, "Stop before the pipe", {"skill_timed_out": True}).skill, "stop")

    def test_feedback_does_not_leak_privileged_map_or_teacher_information(self):
        info = {"skill_name": "move", "wall_contact": 1,
                "path_dist_remaining": 25., "path_progress": 3., "dist_to_goal": 4.,
                "teacher_action": "jump_up"}
        self.assertEqual(execution_feedback(info), {"skill_name": "move", "wall_contact": 1})

    def test_missing_instruction_is_rejected(self):
        with self.assertRaises(ValueError):
            make_prompt(" ", {}, 1)


if __name__ == "__main__":
    unittest.main()
