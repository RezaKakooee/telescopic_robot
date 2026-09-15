"""The wall-jump climbs' phase machine, checked without physics.

`zigzag_climb` (the 0.40 m chimney) and `wall_jump_climb` (the 1 m gap)
are sequencers: given where the ball is and how it moves, they pick the
phase to run. These checks feed them hand-written positions and
velocities and look at the phases they pick. The real climbs, under
MuJoCo, are `tests/test_skills.py::test_chimney_climb_vertical` and
`::test_wall_jump_climb`.
"""
import os
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from radial_sphere.geometry import fibonacci_sphere
from skills import SKILL_REGISTRY, execute_skill
from skills.mid_level.shaft_climbing import (
    PHASES, TUNING, WIDE_TUNING, Shaft, ZigzagState, initial_state, next_phase,
    on_wall_top, shaft_from_boxes, wall_jump_climb, zigzag_climb,
)

E = 0.16
#: The chimney scenario's two wall boxes: [x, y, hx, hy, height].
BOXES = [[0.0, 0.85, 0.6, 0.65, 3.3], [0.0, -0.85, 0.6, 0.65, 4.0]]
#: The wall_jump course: a 1.0 m gap, both walls 3.0 m, tops 4.0 m wide.
WIDE_BOXES = [[0.0, 2.5, 2.0, 2.0, 3.0], [0.0, -2.5, 2.0, 2.0, 3.0]]


def _drive(state, shaft, steps, tuning=TUNING):
    """Run `next_phase` over a list of (pos, vel) and return the phases run."""
    seen = []
    for pos, vel in steps:
        state = next_phase(state, pos=np.array(pos, float), vel=np.array(vel, float),
                           shaft=shaft, max_extend=E, tuning=tuning)
        seen.append(state.phase)
    return state, seen


class ShaftTests(unittest.TestCase):
    def test_shaft_from_boxes_picks_the_lower_wall(self):
        shaft = shaft_from_boxes(BOXES)
        self.assertEqual(shaft.top, 3.3)
        self.assertEqual(shaft.low_sign, +1)
        lo, hi = shaft.box_lat
        self.assertAlmostEqual(lo, 0.20, places=6)   # the shaft's half width
        self.assertAlmostEqual(hi, 1.50, places=6)

    def test_on_wall_top(self):
        shaft = shaft_from_boxes(BOXES)
        self.assertTrue(on_wall_top([0.0, 0.7, 3.5], shaft))
        self.assertFalse(on_wall_top([0.0, 0.0, 3.5], shaft))       # over the shaft
        self.assertFalse(on_wall_top([0.0, 0.7, 2.0], shaft))       # below the top
        self.assertFalse(on_wall_top([0.0, -0.7, 3.5], shaft))      # the tall wall
        self.assertFalse(on_wall_top([0.0, 0.3, 3.5], shaft))       # on the lip edge

    def test_equal_walls_make_either_top_the_exit(self):
        shaft = shaft_from_boxes(WIDE_BOXES)
        self.assertEqual(shaft.low_sign, 0)
        self.assertAlmostEqual(shaft.half_width, 0.5)
        self.assertEqual(shaft.top, 3.0)
        self.assertTrue(on_wall_top([0.0, 1.5, 3.2], shaft))
        self.assertTrue(on_wall_top([0.0, -1.5, 3.2], shaft))
        self.assertFalse(on_wall_top([0.0, 0.0, 3.2], shaft))


class PhaseMachineTests(unittest.TestCase):
    def setUp(self):
        self.shaft = shaft_from_boxes(BOXES)

    def test_launch_then_zigzag_alternates_sides(self):
        # Launch runs at least five steps, then flies; the first wall is
        # picked from which side of the centre the ball is on.
        still = [([0.0, 0.01, 0.20], [0.0, 0.0, 2.0])] * 3
        state, seen = _drive(ZigzagState(), self.shaft, still)
        self.assertEqual(seen, ["launch"] * 3)
        # Above launch_z and no longer rising: fly, then a push once the
        # ball is past the centre and still moving toward the -y wall.
        state, seen = _drive(state, self.shaft, [
            ([0.0, 0.01, 0.40], [0.0, 0.0, 0.1]),
            ([0.0, 0.01, 0.40], [0.0, 0.0, 0.1]),
            ([0.0, 0.01, 0.40], [0.0, 0.0, 0.1]),
            ([0.0, -0.05, 0.42], [0.0, -0.3, 0.0]),
            ([0.0, -0.10, 0.42], [0.0, -0.3, 0.0]),
        ])
        self.assertEqual(seen[-3:], ["fly", "fly", "push"])
        self.assertEqual(state.side, -1)
        # The push ends once the ball leaves that wall fast enough, and the
        # next push is off the other wall.
        leaving = [([0.0, -0.05 + 0.02 * k, 0.5], [0.0, 0.6, 0.5]) for k in range(6)]
        state, seen = _drive(state, self.shaft, leaving)
        self.assertIn("fly", seen)
        state, seen = _drive(state, self.shaft, [([0.0, 0.08, 0.6], [0.0, 0.3, 0.0])] * 2)
        self.assertEqual(seen[-1], "push")
        self.assertEqual(state.side, +1)

    def test_exit_and_land_on_the_lower_wall(self):
        # Flying from the lower (+y) wall toward the tall (-y) wall, above
        # the lower lip: the next push becomes the exit.
        top = self.shaft.top
        state = ZigzagState(phase="fly", side=+1, timer=5)
        state, seen = _drive(state, self.shaft, [([0.0, -0.05, top + 0.05], [0.0, -0.3, 0.5])])
        self.assertEqual(state.successor[:2], ("push", -1))
        state, seen = _drive(state, self.shaft, [([0.0, -0.08, top + 0.06], [0.0, -0.4, 0.5])])
        self.assertEqual(seen, ["exit"])
        # Flying out and coming down over the wall top.
        state, seen = _drive(state, self.shaft, [
            ([0.0, 0.30, top + 0.30], [0.0, 1.5, 1.0]),
            ([0.0, 0.50, top + 0.30], [0.0, 1.5, -1.0]),
        ])
        self.assertEqual(seen[-1], "land")
        # Slow and above the top: brake. Then, once settled, stand and done.
        settled = [([0.0, 0.70, top + 0.20], [0.0, 0.05, 0.0])] * (TUNING.land_steps + 2)
        state, seen = _drive(state, self.shaft, settled)
        self.assertEqual(seen[0], "brake")
        self.assertEqual(seen[-1], "stand")
        state, seen = _drive(state, self.shaft, settled[: TUNING.stand_steps + 2])
        self.assertEqual(seen[-1], "done")

    def test_back_on_the_floor_relaunches_or_recentres(self):
        on_floor = [([0.0, 0.0, 0.18], [0.0, 0.0, 0.0])] * (TUNING.floor_fly_steps + 2)
        state, seen = _drive(ZigzagState(phase="fly", side=+1), self.shaft, on_floor)
        self.assertEqual(seen[-1], "launch")
        drifted = [([0.5, 0.0, 0.18], [0.0, 0.0, 0.0])] * (TUNING.floor_fly_steps + 2)
        state, seen = _drive(ZigzagState(phase="fly", side=+1), self.shaft, drifted)
        self.assertEqual(seen[-1], "recentre")
        # Back at the middle and still: settle, then launch again.
        home = [([0.0, 0.0, 0.18], [0.0, 0.0, 0.0])] * (TUNING.settle_steps + 2)
        state, seen = _drive(state, self.shaft, home)
        self.assertEqual(seen[0], "settle")
        self.assertEqual(seen[-1], "launch")

    def test_target_height_holds_then_descends(self):
        shaft = Shaft(target_z=2.0)
        state, seen = _drive(ZigzagState(phase="fly", side=+1), shaft,
                             [([0.0, 0.0, 2.05], [0.0, 0.2, 0.5])])
        self.assertEqual(seen, ["hold"])
        self.assertAlmostEqual(state.hold_z, 2.05)
        hang = [([0.0, 0.0, 2.04], [0.0, 0.0, -0.01])] * TUNING.hold_steps
        state, seen = _drive(state, shaft, hang)
        self.assertEqual(seen[-1], "descend")
        # The clamp starts at full stroke and is servoed from the first step.
        self.assertAlmostEqual(state.clamp_ext,
                               E - TUNING.descent_gain * (-0.01 - TUNING.descent_vz))
        # Falling faster than wanted: the clamp tightens (extension grows).
        state2, _ = _drive(state, shaft, [([0.0, 0.0, 1.5], [0.0, 0.0, -1.0])])
        self.assertGreaterEqual(state2.clamp_ext, state.clamp_ext)
        # Falling slower than wanted: it loosens.
        state3, _ = _drive(state, shaft, [([0.0, 0.0, 1.5], [0.0, 0.0, -0.1])])
        self.assertLess(state3.clamp_ext, E)
        state, seen = _drive(state, shaft, [([0.0, 0.0, 0.20], [0.0, 0.0, -0.1])] * 2)
        self.assertEqual(seen[-1], "stand")

    def test_wide_gap_jumps_at_the_nearer_wall_and_eases_the_exit(self):
        shaft = shaft_from_boxes(WIDE_BOXES)
        t = WIDE_TUNING
        self.assertEqual(initial_state(t).phase, "crouch")
        # Standing a little toward the +y wall: crouch, then take off at it.
        still = [([0.0, 0.05, 0.20], [0.0, 0.0, 0.0])] * t.crouch_steps
        state, seen = _drive(initial_state(t), shaft, still, t)
        self.assertEqual(seen[0], "crouch")
        self.assertEqual(state.side, -1)                # leaving the -y side
        state, seen = _drive(state, shaft, [
            ([0.0, 0.06, 0.22], [0.0, 0.5, 1.0]),
            ([0.0, 0.08, 0.25], [0.0, 1.0, 1.9]),       # vz past 0.9 * launch_up
            ([0.0, 0.10, 0.28], [0.0, 1.2, 2.0]),
        ], t)
        self.assertEqual(seen, ["takeoff", "takeoff", "fly"])
        # The push starts only within push_reach of the wall face.
        state, seen = _drive(state, shaft, [
            ([0.0, 0.20, 0.8], [0.0, 1.5, 1.0]),
            ([0.0, 0.30, 0.9], [0.0, 1.5, 0.8]),        # 0.20 m from the wall
            ([0.0, 0.32, 0.9], [0.0, 1.5, 0.8]),
        ], t)
        self.assertEqual(seen, ["fly", "fly", "push"])
        self.assertEqual(state.side, +1)
        # A push from within exit_from of the lip is the eased exit, off
        # either wall.
        state = ZigzagState(phase="fly", side=+1, timer=5)
        state, seen = _drive(state, shaft, [([0.0, -0.32, 2.5], [0.0, -1.5, 0.5])], t)
        self.assertEqual(state.successor[:2], ("push", -1))
        state, seen = _drive(state, shaft, [([0.0, -0.34, 2.5], [0.0, -1.5, 0.5])], t)
        self.assertEqual(seen, ["exit"])
        # Below that window it is a full push.
        state = ZigzagState(phase="fly", side=+1, timer=5)
        state, seen = _drive(state, shaft, [([0.0, -0.32, 1.5], [0.0, -1.5, 0.5]),
                                            ([0.0, -0.34, 1.5], [0.0, -1.5, 0.5])], t)
        self.assertEqual(seen[-1], "push")

    def test_wide_gap_lands_on_either_top_and_keeps_braking(self):
        shaft = shaft_from_boxes(WIDE_BOXES)
        t = WIDE_TUNING
        top = shaft.top
        state = ZigzagState(phase="fly", side=+1, timer=5)
        state, seen = _drive(state, shaft, [([0.0, -0.8, top + 1.0], [0.0, -2.0, -1.0])], t)
        self.assertEqual(seen, ["land"])
        # Still rolling on the top after land_steps: keep braking, no `done`.
        rolling = [([0.0, -1.5, top + 0.2], [0.0, -0.8, 0.0])] * (t.land_steps + 5)
        state, seen = _drive(state, shaft, rolling, t)
        self.assertEqual(seen[-1], "brake")
        settled = [([0.0, -1.6, top + 0.2], [0.0, 0.0, 0.0])] * 3
        state, seen = _drive(state, shaft, settled, t)
        self.assertEqual(seen[-1], "stand")

    def test_the_exit_test_takes_effect_one_step_later(self):
        """A phase's own pick runs on the following step, never the same one."""
        rising = ([0.0, 0.01, 0.40], [0.0, 0.0, 0.1])
        state = ZigzagState(phase="launch", timer=TUNING.launch_max_steps)
        state = next_phase(state, pos=np.array(rising[0]), vel=np.array(rising[1]),
                           shaft=self.shaft, max_extend=E)
        self.assertEqual(state.phase, "launch")
        self.assertEqual(state.successor[0], "fly")


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.dirs = fibonacci_sphere(60).astype(np.float32)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])

    def test_every_phase_returns_targets(self):
        for phase in PHASES:
            with self.subTest(phase=phase):
                out = zigzag_climb(self.quat, self.dirs, E, phase=phase, side=-1,
                                   lin_vel=np.array([0.2, 0.1]), along_off=0.2,
                                   core_z=0.3, clamp_ext=0.05)
                self.assertEqual(out.shape, (60,))
                self.assertTrue(np.all(out >= 0.0) and np.all(out <= E + 1e-6))

    def test_push_fires_the_rods_into_the_chosen_wall(self):
        R = np.eye(3)
        for side in (+1, -1):
            out = zigzag_climb(self.quat, self.dirs, E, phase="push", side=side)
            fired = self.dirs[out > 0.5 * E] @ R.T
            self.assertGreater(len(fired), 0)
            self.assertTrue(np.all(side * fired[:, 1] > 0), "rods point at the wrong wall")
            self.assertTrue(np.all(fired[:, 2] < 0), "push rods point downward")

    def test_unknown_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            zigzag_climb(self.quat, self.dirs, E, phase="somersault")

    def test_registered_under_its_aliases(self):
        for name in ("zigzag_climb", "zigzag", "chimney_zigzag"):
            self.assertIs(SKILL_REGISTRY[name], zigzag_climb)
        for name in ("wall_jump_climb", "wall_jump", "wide_zigzag"):
            self.assertIs(SKILL_REGISTRY[name], wall_jump_climb)
        _, meta = execute_skill("zigzag", self.quat, self.dirs, E, phase="fly",
                                return_metadata=True)
        self.assertEqual(meta["skill"], "zigzag_climb")
        _, meta = execute_skill("wall_jump", self.quat, self.dirs, E, phase="fly",
                                return_metadata=True)
        self.assertEqual(meta["skill"], "wall_jump_climb")

    def test_wide_takeoff_uses_part_of_the_stroke(self):
        vel = np.array([0.0, 0.0, 0.0])
        out = wall_jump_climb(self.quat, self.dirs, E, phase="takeoff", side=-1, lin_vel=vel)
        self.assertGreater(out.max(), 0.0)
        self.assertLessEqual(out.max(), WIDE_TUNING.launch_power * E + 1e-6)


if __name__ == "__main__":
    unittest.main()
