"""Unit tests for the traverse_rough_terrain skill primitive and active suspension."""
import unittest
import numpy as np
import os
os.environ["MUJOCO_GL"] = "egl"

from skills.terrain_following import traverse_rough_terrain
from skills import execute_skill, SKILL_REGISTRY
from skills.runner import run_skill
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv


class TestRoughTerrainSkill(unittest.TestCase):
    def setUp(self):
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        from radial_sphere.geometry import fibonacci_sphere
        self.dirs_body = fibonacci_sphere(60).astype(np.float32)
        self.max_extend = 0.16
        self.min_offset = 0.025

    def test_registered_in_registry(self):
        """Verify skills are registered in SKILL_REGISTRY."""
        self.assertIn("traverse_rough_terrain", SKILL_REGISTRY)
        self.assertIn("rough_terrain", SKILL_REGISTRY)
        self.assertIn("active_suspension", SKILL_REGISTRY)
        self.assertEqual(SKILL_REGISTRY["traverse_rough_terrain"], traverse_rough_terrain)
        self.assertEqual(SKILL_REGISTRY["rough_terrain"], traverse_rough_terrain)

    def test_anti_snag_lockout(self):
        """Forward and upward rods must be locked down to min_offset."""
        d_hat = np.array([1.0, 0.0], dtype=np.float32)
        targets = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            min_offset=self.min_offset, speed=1.5
        )
        self.assertEqual(len(targets), 60)
        # Leading rods (x > -0.05) or top rods (z > 0.10)
        for i, d in enumerate(self.dirs_body):
            if d[0] > -0.05 or d[2] > 0.10:
                self.assertAlmostEqual(targets[i], self.min_offset, places=4,
                                       msg=f"Rod {i} pointing forward/up should be retracted")

    def test_skyhook_heave_compensation(self):
        """High core height should retract downward rods; low core should extend them."""
        d_hat = np.array([1.0, 0.0], dtype=np.float32)
        target_z = 0.28
        
        down_idx = [i for i, d in enumerate(self.dirs_body) if d[2] < -0.30 and d[0] <= -0.10]
        self.assertTrue(len(down_idx) > 0, "Expected downward pointing rods")

        # Nominal height
        targets_nom = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=target_z, core_vz=0.0, target_ride_height=target_z
        )

        # High core (e.g. climbed over boulder)
        targets_high = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=target_z + 0.05, core_vz=0.0, target_ride_height=target_z
        )

        # Low core (e.g. dipped into trench)
        targets_low = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=target_z - 0.05, core_vz=0.0, target_ride_height=target_z
        )

        for idx in down_idx:
            self.assertLessEqual(targets_high[idx], targets_nom[idx] + 1e-4)
            self.assertGreaterEqual(targets_low[idx], targets_nom[idx] - 1e-4)

    def test_skyhook_velocity_damping(self):
        """Upward heave velocity vz > 0 should yield downward rods."""
        d_hat = np.array([1.0, 0.0], dtype=np.float32)
        down_idx = [i for i, d in enumerate(self.dirs_body) if d[2] < -0.30 and d[0] <= -0.10]

        targets_still = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=0.28, core_vz=0.0, target_ride_height=0.28
        )
        targets_rising = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=0.28, core_vz=0.5, target_ride_height=0.28
        )
        for idx in down_idx:
            self.assertLessEqual(targets_rising[idx], targets_still[idx] + 1e-4)

    def test_bump_force_compliance(self):
        """High contact force on downward rods should cause proportional retraction."""
        d_hat = np.array([1.0, 0.0], dtype=np.float32)
        down_idx = [i for i, d in enumerate(self.dirs_body) if d[2] < -0.30 and d[0] <= -0.10][0]

        forces_zero = np.zeros(60, dtype=np.float32)
        forces_high = np.zeros(60, dtype=np.float32)
        forces_high[down_idx] = 40.0  # 40 N heavy rock impact

        t_zero = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=0.28, contact_forces=forces_zero
        )
        t_high = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=0.28, contact_forces=forces_high
        )
        self.assertLess(t_high[down_idx], t_zero[down_idx],
                        "High contact force must cause rod compliance retraction")

    def test_stone_opens_lower_and_hole_opens_longer(self):
        """Underneath bars adjust themselves: when there is a stone bars open lower (retract),
        and when there is a hole they open longer (extend)."""
        d_hat = np.array([1.0, 0.0], dtype=np.float32)
        # Nominal flat ground
        tc_flat = np.zeros(60, dtype=np.float32)
        t_flat = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=0.28, terrain_clearances=tc_flat
        )

        # Select an active underbelly rod in ground stance (not saturated at full stroke)
        down_idx = [i for i, d in enumerate(self.dirs_body) if d[2] < -0.30 and d[0] <= -0.05 and 0.04 < t_flat[i] < 0.12][0]

        # Over a STONE (terrain rises -> clearance is negative, e.g. -0.05m)
        tc_stone = np.zeros(60, dtype=np.float32)
        tc_stone[down_idx] = -0.05
        t_stone = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=0.28, terrain_clearances=tc_stone
        )

        # Over a HOLE (terrain drops -> clearance is positive, e.g. +0.05m)
        tc_hole = np.zeros(60, dtype=np.float32)
        tc_hole[down_idx] = +0.05
        t_hole = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=0.28, terrain_clearances=tc_hole
        )

        self.assertLess(t_stone[down_idx], t_flat[down_idx],
                        "When there is a stone, the bar must open lower (shorter stroke)")
        self.assertGreater(t_hole[down_idx], t_flat[down_idx],
                           "When there is a hole, the bar must open longer (longer stroke)")

    def test_terrain_corrections_skip_leading_rods(self):
        """A leading rod must stay locked out, even over a measured hole.

        A leading rod driven into a pit plants against the far wall and stops
        the robot, so terrain feedback only rides on the trailing rods.
        """
        d_hat = np.array([1.0, 0.0], dtype=np.float32)
        lead_idx = [i for i, d in enumerate(self.dirs_body) if d[0] > 0.30 and d[2] < -0.40]
        trail_idx = [i for i, d in enumerate(self.dirs_body) if d[0] < -0.30 and d[2] < -0.40]
        self.assertTrue(lead_idx and trail_idx)

        tc_hole = np.full(60, 0.06, dtype=np.float32)
        flat = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=0.20, terrain_clearances=np.zeros(60, dtype=np.float32),
        )
        hole = traverse_rough_terrain(
            self.quat, self.dirs_body, self.max_extend, d_hat=d_hat,
            core_z=0.20, terrain_clearances=tc_hole,
        )
        for i in lead_idx:
            self.assertAlmostEqual(hole[i], flat[i], places=6,
                                   msg=f"Leading rod {i} must not reach into a hole")
            self.assertAlmostEqual(hole[i], self.min_offset, places=6)
        self.assertTrue(any(hole[i] > flat[i] + 1e-4 for i in trail_idx),
                        "Trailing rods must still reach into a hole")

    def test_rough_gait_climbs_before_pivoting_away(self):
        """A blocked robot on rough ground must try to climb over first.

        The plain rolling gait is stopped by a step of about 4 cm, so the old
        reflex pivoted 65 degrees away from every rock. That looks like the
        robot refusing the terrain.
        """
        from skills.navigation import stay_in_boundary

        common = dict(
            ball_xy=np.array([0.2, 0.1]), lin_vel=np.array([0.0, 0.0]),
            boundary_radius=2.6, safety_margin=0.60, step_count=100,
            core_z=0.20, core_vz=0.0, return_metadata=True,
        )
        _, plain = stay_in_boundary(self.quat, self.dirs_body, self.max_extend, **common)
        self.assertEqual(plain["action_name"], "roam_obstacle_escape")

        _, climbing = stay_in_boundary(
            self.quat, self.dirs_body, self.max_extend,
            rough_terrain_gait=True, obstruction_steps=0, climb_patience=150, **common)
        self.assertEqual(climbing["action_name"], "roam_climb_over")
        self.assertTrue(climbing["is_obstructed"])

        # Give up after climb_patience and pivot away, so it can never
        # grind against the same rock forever.
        _, giving_up = stay_in_boundary(
            self.quat, self.dirs_body, self.max_extend,
            rough_terrain_gait=True, obstruction_steps=150, climb_patience=150, **common)
        self.assertEqual(giving_up["action_name"], "roam_obstacle_escape")

    def test_support_weight_must_match_rod_count(self):
        from skills.suspension import apply_suspension
        with self.assertRaises(ValueError):
            apply_suspension(np.zeros(4), -np.ones(4), 0.16, core_z=0.2, core_vz=0.0,
                             support_weight=np.ones(3))

    def test_execute_skill_alias(self):
        """Verify execute_skill works with rough terrain aliases."""
        t1 = execute_skill("traverse_rough_terrain", self.quat, self.dirs_body, self.max_extend)
        t2 = execute_skill("rough_terrain", self.quat, self.dirs_body, self.max_extend)
        t3 = execute_skill("active_suspension", self.quat, self.dirs_body, self.max_extend)
        np.testing.assert_allclose(t1, t2)
        np.testing.assert_allclose(t1, t3)

    def test_sim_rocky_traversal(self):
        """Run 100 steps on rocky terrain and verify stable traversal."""
        cfg = load_config("configs/rl/rocky_mountain_terrain.yaml")
        env = MujocoRadialSphereEnv(cfg, max_steps=300)
        env.reset(seed=42)

        stats = run_skill(env, "traverse_rough_terrain", steps=100, d_hat=np.array([1.0, 0.0]))
        self.assertGreater(stats["displacement"][0], 0.25, "Robot should advance forward > 0.25m over rocks")
        self.assertGreater(stats["end"][2], 0.15, "Ball core should remain upright")
        self.assertLess(stats["end"][2], 0.45, "Ball core should not fly away")
        env.close()


if __name__ == "__main__":
    unittest.main()
