"""Every industrial-inspection course builds, loads in MuJoCo and has a usable route."""
import unittest

import numpy as np

from radial_sphere.config import load_config
from radial_sphere.inspection_oracle import InspectionOracle
from radial_sphere.inspection_scenarios import INSPECTION_SCENARIOS
from radial_sphere.map_perception import LocalMapPatchExtractor
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv


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

    def test_heightmap_knows_the_box_tops(self):
        """A deck the ball lands on must be in the map, or the jump option never
        sees its landing and coasts for its whole 2.4 s. Beams stay out."""
        cfg = load_config()
        sc = generate_scenario("inspection_doubling_boxes", cfg, seed=0)
        pe = LocalMapPatchExtractor(sc)
        for x, want in ((4.0, 0.0), (6.0, 0.05), (16.0, 0.80)):
            gx, gy = pe._world_to_grid(x, 0.0)
            self.assertAlmostEqual(float(pe.global_elevation[gy, gx]), want, places=3, msg=f"x = {x}")
        sc = generate_scenario("inspection_hurdle_lane", cfg, seed=0)
        pe = LocalMapPatchExtractor(sc)
        gx, gy = pe._world_to_grid(3.0, 0.0)                    # the first beam
        self.assertEqual(float(pe.global_elevation[gy, gx]), 0.0)

    def test_expert_runs_the_platform_routine_on_short_decks(self):
        """Land, brake, flip, back up, flip, settle, then run and jump from the deck's own window.

        There is no reverse decision: the ball always rolls forward, it flips first."""
        cfg = load_config()
        sc = generate_scenario("inspection_doubling_boxes", cfg, seed=0)
        oracle = InspectionOracle(sc)
        kinds = [st.kind for st in oracle.stations]
        self.assertEqual(kinds.count("deck"), 5)
        self.assertNotIn("jump", kinds, "a box behind a pit is jumped with the pit, not on its own")
        deck = [st for st in oracle.stations if st.kind == "deck"][1]
        x_on = float(sc.path_pts[np.searchsorted(oracle.s, deck.s_start + 0.9)][0])
        oracle._last = "jump_forward_while_moving"
        seen = []
        for _ in range(16):
            name, _, why = oracle.select([x_on, 0.0, 0.3])
            seen.append(name)
            if name == "move" and oracle._flipped:
                x_on -= 0.15
        self.assertEqual(seen[:4], ["stop"] * 4)
        self.assertEqual(seen[4], "flip")
        self.assertNotIn("reverse", seen)
        self.assertEqual(seen.count("flip"), 2, seen)              # back up, then face forward again
        self.assertEqual(seen[-1], "move")
        self.assertFalse(oracle._flipped)

    def test_flip_turns_move_into_the_reverse_gait(self):
        """The env runs the reverse gait for "move" while flipped, and info says so."""
        cfg = load_config("configs/rl/playground_parkour_skills.yaml")     # the skills backend
        sc = generate_scenario("inspection_hurdle_lane", cfg, seed=0)
        env = SkillArbitrationEnv(cfg, scenario=sc, seed=0, max_steps=50)
        env.reset(seed=0)
        for _ in range(3):
            env.step(env._one_hot("move"))
        x0 = float(env.env.data.qpos[0])
        self.assertGreater(x0, 0.05)
        _, _, _, _, info = env.step(env._one_hot("flip"))
        self.assertTrue(info["flipped"])
        for _ in range(8):
            _, _, _, _, info = env.step(env._one_hot("move"))
        self.assertLess(float(env.env.data.qpos[0]), x0, "flipped move must roll back along the route")
        _, _, _, _, info = env.step(env._one_hot("flip"))
        self.assertFalse(info["flipped"])
        env.close()


if __name__ == "__main__":
    unittest.main()
