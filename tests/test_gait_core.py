"""The two gait callers must keep sharing one drive wave.

``radial_sphere.controller.bar_targets`` and
``skills.low_level.locomotion.traverse_rough_terrain`` used to hold separate copies of
the same arithmetic. They drifted: a widened curb-vault sector landed in one
and not the other. Both now call ``radial_sphere.gait``, and these checks fail
if either grows a private copy again.
"""
import os
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

from radial_sphere import gait
from radial_sphere.controller import bar_targets
from radial_sphere.geometry import fibonacci_sphere
from skills.low_level.terrain_following import traverse_rough_terrain

MAX_EXTEND = 0.16
MIN_OFFSET = 0.025


def _quats():
    rng = np.random.default_rng(0)
    out = [np.array([1.0, 0.0, 0.0, 0.0])]
    for _ in range(4):
        q = rng.normal(size=4)
        out.append(q / np.linalg.norm(q))
    return out


class GaitCoreTests(unittest.TestCase):
    def setUp(self):
        self.dirs = fibonacci_sphere(60).astype(np.float32)
        self.headings = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                         np.array([-0.6, 0.8])]

    def test_both_callers_agree_on_the_shared_core(self):
        """With every extra switched off, the two gaits must match exactly."""
        for quat in _quats():
            for d_hat in self.headings:
                for back_gain in (1.0, 1.6, 2.4):
                    for vault, belly in ((False, False), (True, True)):
                        legacy = bar_targets(
                            quat, self.dirs, MAX_EXTEND, d_hat, drive=1.0,
                            min_offset=MIN_OFFSET, back_gain=back_gain,
                            enable_curb_vaulting=vault, curb_boost_gain=2.6,
                            enable_underbelly_contact=belly,
                            underbelly_stance_gain=0.42,
                            underbelly_threshold_z=-0.20)
                        skill = traverse_rough_terrain(
                            quat, self.dirs, MAX_EXTEND, d_hat=d_hat,
                            back_gain=back_gain, min_offset=MIN_OFFSET,
                            enable_curb_vaulting=vault, curb_boost_gain=2.6,
                            enable_underbelly_contact=belly,
                            underbelly_stance_gain=0.42,
                            underbelly_threshold_z=-0.20)
                        np.testing.assert_allclose(
                            skill, legacy, atol=1e-9,
                            err_msg=f"gaits diverged at gain={back_gain} "
                                    f"vault={vault} belly={belly}")

    def test_leading_and_top_rods_never_extend(self):
        u_long = np.linspace(-1.0, 1.0, 41)
        u_z = np.zeros_like(u_long)
        wave = gait.lock_out_leading(np.ones_like(u_long), u_long, u_z)
        self.assertTrue(np.all(wave[u_long > gait.LEADING_LOCKOUT] == 0.0))
        self.assertTrue(np.all(wave[u_long < -0.5] == 1.0))

        u_long = np.full(5, -1.0)
        u_z = np.array([-1.0, 0.0, 0.09, 0.11, 1.0])
        wave = gait.lock_out_leading(np.ones(5), u_long, u_z)
        self.assertTrue(np.all(wave[u_z > gait.TOP_LOCKOUT] == 0.0))

    def test_braking_rods_are_exempt_from_the_leading_lockout(self):
        u_long = np.array([0.5, 0.5])
        u_z = np.array([-0.6, -0.6])
        keep = np.array([True, False])
        wave = gait.lock_out_leading(np.ones(2), u_long, u_z, keep=keep)
        self.assertEqual(wave[0], 1.0)
        self.assertEqual(wave[1], 0.0)

    def test_curb_vault_covers_rods_just_above_the_waist(self):
        """A rod at u_z = 0.08 still bears on a tall rock, so it gets boosted."""
        u_long = np.full(3, -0.5)
        u_z = np.array([-0.5, 0.08, 0.20])
        wave = gait.curb_vault(np.full(3, 0.2), u_long, u_z, 2.6)
        np.testing.assert_allclose(wave[:2], 0.52)
        self.assertEqual(wave[2], 0.2)

    def test_support_weight_excludes_leading_rods(self):
        u_long = np.array([-1.0, -0.5, -0.05, 0.5])
        weight = gait.support_weight(u_long)
        self.assertEqual(weight[0], 1.0)
        self.assertEqual(weight[2], 0.0)
        self.assertEqual(weight[3], 0.0)
        self.assertTrue(0.0 < weight[1] <= 1.0)

    def test_travel_frame_matches_a_hand_computed_case(self):
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        u_long, u_lat, u_z = gait.travel_frame(quat, dirs, [1.0, 0.0])
        np.testing.assert_allclose(u_long, [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(u_lat, [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(u_z, [0.0, 0.0, -1.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
