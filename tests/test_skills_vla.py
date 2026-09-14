"""Unit tests for skills_vla package."""
from __future__ import annotations

import unittest
import numpy as np

from radial_sphere.gait import MIN_OFFSET
from skills_vla import (
    RobotState,
    ParamSpec,
    SkillResult,
    ego_to_world_heading,
    dispatch_vla_action,
    get_skill,
    SKILL_NAMES,
    roll,
    jump_forward,
    jump_gap,
    traverse_rough,
    brake_stop,
)
from skills_vla.roll import RollSkill
from skills_vla.jump_forward import JumpForwardSkill
from skills_vla.jump_gap import JumpGapSkill
from skills_vla.traverse_rough import TraverseRoughSkill
from skills_vla.brake_stop import BrakeStopSkill


def make_dummy_state(n_bars: int = 60, max_extend: float = 0.16) -> RobotState:
    """Create a realistic dummy state for unit testing."""
    # Fibonacci sphere directions
    indices = np.arange(0, n_bars, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_bars)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    dirs_body = np.column_stack([x, y, z])

    return RobotState(
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        dirs_body=dirs_body,
        max_extend=max_extend,
        lin_vel=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        core_z=0.22,
        core_vz=0.0,
        contact_forces=np.zeros(n_bars, dtype=np.float32),
        min_offset=MIN_OFFSET,
    )


class TestSkillsVLA(unittest.TestCase):
    def setUp(self):
        self.state = make_dummy_state(60, 0.16)

    def test_param_spec_scaling(self):
        param = ParamSpec("speed", low=0.2, high=1.6, default=1.1)
        self.assertAlmostEqual(param.scale(-1.0), 0.2)
        self.assertAlmostEqual(param.scale(1.0), 1.6)
        self.assertAlmostEqual(param.scale(0.0), 0.9)
        self.assertAlmostEqual(param.unscale(0.2), -1.0)
        self.assertAlmostEqual(param.unscale(1.6), 1.0)

    def test_ego_to_world_heading(self):
        # Camera facing East (0 rad), ego heading forward (0 rad) -> world East [1, 0]
        fwd = ego_to_world_heading(ego_angle=0.0, camera_heading=0.0)
        np.testing.assert_allclose(fwd, [1.0, 0.0], atol=1e-6)

        # Camera facing East (0 rad), ego heading left (+pi/2 rad) -> world North [0, 1]
        left = ego_to_world_heading(ego_angle=np.pi / 2, camera_heading=0.0)
        np.testing.assert_allclose(left, [0.0, 1.0], atol=1e-6)

        # Camera facing North (pi/2 rad), ego heading forward (0 rad) -> world North [0, 1]
        cam_north = ego_to_world_heading(ego_angle=0.0, camera_heading=np.pi / 2)
        np.testing.assert_allclose(cam_north, [0.0, 1.0], atol=1e-6)

    def test_roll_skill(self):
        targets = roll(self.state, camera_heading=0.0, heading_ego=0.0, speed=1.2)
        self.assertEqual(targets.shape, (60,))
        self.assertTrue(np.all(targets >= self.state.min_offset - 1e-6))
        self.assertTrue(np.all(targets <= self.state.max_extend + 1e-6))
        self.assertTrue(np.all(np.isfinite(targets)))

    def test_jump_forward_lifecycle(self):
        skill = JumpForwardSkill()
        phases_seen = set()

        # Step through the verified skills.runner schedule (200-step budget)
        for substep in range(205):
            res = skill.step(self.state, substep=substep, camera_heading=0.0, power=0.9)
            self.assertIsInstance(res, SkillResult)
            self.assertEqual(res.targets.shape, (60,))
            self.assertTrue(np.all(res.targets >= self.state.min_offset - 1e-6))
            self.assertTrue(np.all(res.targets <= self.state.max_extend + 1e-6))
            phases_seen.add(res.info["phase"])

            if res.done:
                break

        # The dummy state never leaves the ground, so "airborne" cannot appear.
        self.assertIn("sprint", phases_seen)
        self.assertIn("dip", phases_seen)
        self.assertIn("launch", phases_seen)
        self.assertIn("landing", phases_seen)
        self.assertTrue(res.done)

    def test_jump_gap_lifecycle(self):
        skill = JumpGapSkill()
        res_start = skill.step(self.state, substep=0, camera_heading=0.0)
        self.assertEqual(res_start.info["phase"], "crouch")
        self.assertFalse(res_start.done)

        res_end = skill.step(self.state, substep=79, camera_heading=0.0)
        self.assertTrue(res_end.done)

    def test_traverse_rough_skill(self):
        targets = traverse_rough(self.state, camera_heading=np.pi / 4, heading_ego=0.0, curb_boost=3.0)
        self.assertEqual(targets.shape, (60,))
        self.assertTrue(np.all(targets >= self.state.min_offset - 1e-6))
        self.assertTrue(np.all(targets <= self.state.max_extend + 1e-6))

    def test_brake_stop_skill(self):
        # Moving state
        targets_moving = brake_stop(self.state, strength=2.0)
        self.assertEqual(targets_moving.shape, (60,))
        self.assertTrue(np.all(targets_moving >= self.state.min_offset - 1e-6))

        # Stationary state
        at_rest = make_dummy_state(60, 0.16)
        object.__setattr__(at_rest, "lin_vel", np.zeros(3))
        targets_rest = brake_stop(at_rest, strength=2.0)
        self.assertEqual(targets_rest.shape, (60,))
        # Downward rods should be extended for stance
        self.assertTrue(np.any(targets_rest > at_rest.min_offset))

    def test_dispatcher_continuous_vector(self):
        # Test dispatching with normalized [-1, 1] continuous action vector
        for name in SKILL_NAMES:
            res = dispatch_vla_action(
                name,
                action_params=np.array([0.5, -0.2, 0.8]),
                state=self.state,
                camera_heading=0.0,
                substep=0,
            )
            self.assertEqual(res.targets.shape, (60,))
            self.assertTrue(np.all(np.isfinite(res.targets)))

    def test_dispatcher_aliases(self):
        # 'move' should map to 'roll', 'jump' to 'jump_forward', 'brake' to 'brake_stop'
        self.assertEqual(get_skill("move").name, "roll")
        self.assertEqual(get_skill("jump").name, "jump_forward")
        self.assertEqual(get_skill("brake").name, "brake_stop")


if __name__ == "__main__":
    unittest.main()
