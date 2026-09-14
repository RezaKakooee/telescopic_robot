"""Every industrial-inspection course builds, loads in MuJoCo and has a usable route."""
import unittest

import numpy as np

from radial_sphere.config import load_config
from radial_sphere.inspection_scenarios import INSPECTION_SCENARIOS
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario


class InspectionScenarioTests(unittest.TestCase):
    def test_all_courses_build_and_load(self):
        cfg = load_config()
        for kind in INSPECTION_SCENARIOS:
            with self.subTest(kind=kind):
                sc = generate_scenario(kind, cfg, seed=0)
                self.assertEqual(sc.kind, kind)
                self.assertGreater(sc.path_length, 10.0)
                np.testing.assert_allclose(sc.path_pts[0], sc.spawn_xy, atol=1e-5)
                np.testing.assert_allclose(sc.path_pts[-1], sc.goal, atol=1e-5)
                env = MujocoRadialSphereEnv(cfg, max_steps=5, scenario=sc)
                env.reset(seed=0)
                env.step(np.zeros(env.n_bars, dtype=np.float32))
                env.close()

    def test_random_courses_change_with_the_seed(self):
        cfg = load_config()
        a = generate_scenario("inspection_warehouse", cfg, seed=1)
        b = generate_scenario("inspection_warehouse", cfg, seed=2)
        self.assertFalse(np.array_equal(a.walls, b.walls))


if __name__ == "__main__":
    unittest.main()
