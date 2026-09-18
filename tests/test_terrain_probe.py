"""The terrain probe: edges ahead of the ball, the jump plan, and the self-timed jump option.

Three layers are checked, from the cheapest up:

* ``edges_in`` on synthetic height profiles (no MuJoCo);
* ``plan_jump`` on hand-made :class:`Edge` lists;
* ``TerrainProbe`` on real inspection courses, and the running jump option
  end to end in ``SkillArbitrationEnv``.
"""
import os
import unittest

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from radial_sphere import terrain_probe as P
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from radial_sphere.terrain_probe import Edge, TerrainProbe, edges_in, plan_jump

SKILLS_CFG = "configs/rl/playground_parkour_skills.yaml"
JUMP = "jump_forward_while_moving"


def synthetic(*bands, max_dist=P.PROBE_RANGE, step=P.PROBE_STEP):
    """A flat floor with height bands ``(start, end, height)`` for ``start <= d < end``."""
    dists = np.arange(0.0, max_dist + 1e-9, step)
    heights = np.zeros(len(dists))
    for start, end, h in bands:
        heights[(dists >= start - 1e-9) & (dists < end - 1e-9)] = h
    return dists, heights


def edge(dist, kind="rise", change=0.16, length=0.10, returns=False, level=0.0):
    return Edge(dist=dist, kind=kind, change=change, length=length, returns=returns, level=level)


def place(env, x, y, ground=0.0):
    """Put the ball at rest over (x, y) on ground of the given height."""
    env.data.qpos[:3] = [x, y, ground + env.sphere_radius + env.base_ext + 0.02]
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


class EdgesInTests(unittest.TestCase):
    def test_flat_floor_has_no_edges(self):
        self.assertEqual(edges_in(*synthetic()), [])

    def test_all_nan_profile_has_no_edges(self):
        dists, heights = synthetic()
        heights[:] = np.nan
        self.assertEqual(edges_in(dists, heights), [])

    def test_beam_is_a_short_rise_then_a_drop(self):
        edges = edges_in(*synthetic((2.0, 2.1, 0.16)))
        self.assertEqual(len(edges), 2)
        rise, drop = edges
        self.assertEqual(rise.kind, "rise")
        self.assertAlmostEqual(rise.dist, 2.0, delta=0.05)
        self.assertAlmostEqual(rise.length, 0.1, places=6)
        self.assertAlmostEqual(rise.change, 0.16, places=6)
        self.assertAlmostEqual(rise.level, 0.0, places=6)
        self.assertFalse(rise.returns)
        self.assertEqual(rise.shape, "beam")
        self.assertEqual(drop.kind, "drop")
        self.assertAlmostEqual(drop.change, -0.16, places=6)
        self.assertEqual(drop.length, float("inf"))
        self.assertFalse(drop.returns)
        self.assertIsNone(drop.shape)

    def test_trench_is_a_drop_that_returns(self):
        edges = edges_in(*synthetic((2.0, 2.4, -0.4)))
        drop = edges[0]
        self.assertEqual(drop.kind, "drop")
        self.assertAlmostEqual(drop.dist, 2.0, delta=0.05)
        self.assertAlmostEqual(drop.length, 0.4, places=6)
        self.assertAlmostEqual(drop.change, -0.4, places=6)
        self.assertTrue(drop.returns)
        self.assertEqual(drop.shape, "gap")

    def test_step_up_that_never_comes_back_is_a_platform(self):
        edges = edges_in(*synthetic((2.0, 99.0, 0.3)))
        self.assertEqual(len(edges), 1)
        step = edges[0]
        self.assertEqual(step.kind, "rise")
        self.assertEqual(step.length, float("inf"))
        self.assertFalse(step.returns)
        self.assertEqual(step.shape, "platform")

    def test_tall_rise_is_a_wall_with_no_shape(self):
        edges = edges_in(*synthetic((2.0, 99.0, 1.0)))
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].kind, "rise")
        self.assertGreater(edges[0].change, P.MAX_JUMP_RISE)
        self.assertIsNone(edges[0].shape)

    def test_small_bump_is_ignored(self):
        self.assertEqual(edges_in(*synthetic((2.0, 2.1, 0.05))), [])

    def test_nan_samples_are_skipped_without_an_edge(self):
        dists, heights = synthetic()
        heights[[3, 17, 40, 41, 69]] = np.nan
        self.assertEqual(edges_in(dists, heights), [])
        # Missing rays before, inside and after a beam do not split or hide it.
        dists, heights = synthetic((2.0, 2.1, 0.16))
        heights[[5, 20, 41, 50]] = np.nan
        edges = edges_in(dists, heights)
        self.assertEqual([e.kind for e in edges], ["rise", "drop"])
        self.assertAlmostEqual(edges[0].dist, 2.0, delta=0.05)
        self.assertAlmostEqual(edges[0].length, 0.1, places=6)
        self.assertEqual(edges[0].shape, "beam")

    def test_drop_that_comes_back_late_is_not_a_gap(self):
        edges = edges_in(*synthetic((1.0, 2.5, -0.4)))
        drop = edges[0]
        self.assertEqual(drop.kind, "drop")
        self.assertAlmostEqual(drop.length, 1.5, places=6)
        self.assertGreater(drop.length, P.MAX_GAP)
        self.assertFalse(drop.returns)
        self.assertIsNone(drop.shape)


class PlanJumpTests(unittest.TestCase):
    def test_picks_the_first_target(self):
        beam = edge(1.5)
        gap = edge(2.5, kind="drop", change=-0.4, length=0.4, returns=True)
        plan = plan_jump([beam, gap])
        self.assertIs(plan["edge"], beam)
        self.assertEqual(plan["shape"], "beam")
        self.assertEqual(plan["trigger"], P.TRIGGER["beam"])

    def test_a_rise_below_the_ball_is_not_a_target(self):
        """A step down then a beam on the lower level: the ball rolls down first
        and looks again from there. Told the real floor is that lower level
        (inside a pipe the first ray sample is the roof), the beam is a target."""
        down = edge(1.0, kind="drop", change=-0.3, length=float("inf"))
        beam = edge(2.0, level=-0.3)
        self.assertIsNone(plan_jump([down, beam]))
        plan = plan_jump([down, beam], ground=-0.3)
        self.assertIs(plan["edge"], beam)
        self.assertEqual(plan["shape"], "beam")

    def test_wall_in_front_gives_none(self):
        wall = edge(1.0, change=1.0, length=0.05)
        beam = edge(2.0)
        self.assertIsNone(plan_jump([wall, beam]))

    def test_empty_list_gives_none(self):
        self.assertIsNone(plan_jump([]))

    def test_honours_min_target(self):
        """An edge already under the rods is rolled, not jumped, and not looked past:
        aiming at the next one would drive the approach into this one."""
        near = edge(P.MIN_TARGET - 0.05)
        far = edge(2.0)
        self.assertIsNone(plan_jump([near]))
        self.assertIsNone(plan_jump([near, far]))
        self.assertIsNotNone(plan_jump([edge(P.MIN_TARGET)]), "exactly MIN_TARGET is still a target")

    def test_trigger_matches_the_shape(self):
        cases = {
            "beam": edge(2.0, length=0.1),
            "riser": edge(2.0, change=0.4, length=1.8),
            "platform": edge(2.0, change=0.3, length=float("inf")),
            "gap": edge(2.0, kind="drop", change=-0.4, length=0.4, returns=True),
        }
        for shape, e in cases.items():
            with self.subTest(shape=shape):
                self.assertEqual(e.shape, shape)
                plan = plan_jump([e])
                self.assertEqual(plan["shape"], shape)
                self.assertEqual(plan["trigger"], P.TRIGGER[shape])


class TerrainProbeCourseTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(SKILLS_CFG)
        self.env = None

    def tearDown(self):
        if self.env is not None:
            self.env.close()

    def open(self, kind):
        sc = generate_scenario(kind, self.cfg, seed=0)
        self.env = MujocoRadialSphereEnv(self.cfg, max_steps=5, scenario=sc)
        self.env.reset(seed=0)
        return sc, TerrainProbe(self.env)

    def test_hurdle_lane_sees_the_first_beam(self):
        sc, probe = self.open("inspection_hurdle_lane")
        beam_x, spawn_x = float(sc.steps[0][0]), float(sc.spawn_xy[0])
        edges = probe.edges([1.0, 0.0])
        self.assertTrue(edges)
        first = edges[0]
        self.assertEqual(first.kind, "rise")
        self.assertAlmostEqual(first.dist, beam_x - 0.05 - spawn_x, delta=0.1)
        self.assertEqual(first.shape, "beam")
        plan = probe.plan([1.0, 0.0])
        self.assertEqual(plan["shape"], "beam")
        self.assertEqual(plan["trigger"], P.TRIGGER["beam"])

    def test_doubling_boxes_sees_the_pit_then_the_next_box(self):
        sc, probe = self.open("inspection_doubling_boxes")
        gap = sc.gaps[0]
        pit_start = float(gap[0]) - float(gap[2])
        place(self.env, pit_start - 1.0, 0.0, ground=float(sc.steps[0][4]))
        edges = probe.edges([1.0, 0.0])
        self.assertGreaterEqual(len(edges), 2)
        pit, rise = edges[0], edges[1]
        self.assertEqual(pit.kind, "drop")
        self.assertAlmostEqual(pit.dist, 1.0, delta=0.1)
        self.assertAlmostEqual(pit.length, 0.25, delta=0.05)
        self.assertTrue(pit.returns)
        self.assertEqual(pit.shape, "gap")
        self.assertEqual(rise.kind, "rise")
        plan = probe.plan([1.0, 0.0])
        self.assertEqual(plan["shape"], "gap")
        self.assertEqual(plan["trigger"], P.TRIGGER["gap"])

    def test_boiler_house_sees_the_first_riser_as_a_riser(self):
        """From the run-up after the door (route point x = 5.2, y = 2.5).

        From x = 3.5 the probe cannot see a riser: partition 1 stands at
        x = 4.5 for y < 3.6, and the second riser (x = 8.6) is 5.1 m away,
        past the 3.5 m range, so the first tread has no visible end."""
        sc, probe = self.open("inspection_boiler_house")
        stairs = sc.staircases[0]
        first_riser_x, run = float(stairs[0]), float(stairs[4])
        place(self.env, 5.2, 2.5)
        edges = probe.edges([1.0, 0.0])
        self.assertTrue(edges)
        riser = edges[0]
        self.assertEqual(riser.kind, "rise")
        self.assertAlmostEqual(riser.dist, first_riser_x - 5.2, delta=0.1)
        self.assertAlmostEqual(riser.length, run, delta=0.1)
        self.assertGreater(riser.length, 1.2)
        self.assertLess(riser.length, 2.5)
        self.assertEqual(riser.shape, "riser")
        self.assertEqual(probe.plan([1.0, 0.0])["trigger"], P.TRIGGER["riser"])
        # Behind the partition the wall blocks the view: no plan.
        place(self.env, 3.5, 2.5)
        self.assertIsNone(probe.plan([1.0, 0.0]))

    def test_wall_ahead_gives_no_plan(self):
        sc, probe = self.open("inspection_hurdle_lane")
        wall_x = float(np.max(sc.walls[:, [0, 2]]))            # the far boundary wall
        place(self.env, wall_x - 1.0, 0.0)
        edges = probe.edges([1.0, 0.0])
        self.assertTrue(edges)
        self.assertEqual(edges[0].kind, "rise")
        self.assertAlmostEqual(edges[0].dist, 1.0, delta=0.1)
        self.assertGreater(edges[0].change, P.MAX_JUMP_RISE)
        self.assertIsNone(edges[0].shape)
        self.assertIsNone(probe.plan([1.0, 0.0]))


class SelfTimedJumpOptionTests(unittest.TestCase):
    def setUp(self):
        self.env = None

    def tearDown(self):
        if self.env is not None:
            self.env.close()

    def open(self, kind, self_timed=True):
        cfg = load_config(SKILLS_CFG)
        cfg.rl.self_timed_jumps = self_timed
        sc = generate_scenario(kind, cfg, seed=0)
        self.env = SkillArbitrationEnv(cfg, scenario=sc, seed=0, max_steps=200)
        self.env.reset(seed=0)
        return sc, self.env

    def roll_to(self, env, x, limit=60):
        for _ in range(limit):
            if float(env.env.data.qpos[0]) > x:
                return
            env.step(env._one_hot("move"))
        self.fail(f"the ball did not reach x > {x} in {limit} macro steps")

    def test_jump_approaches_the_beam_and_lands_past_it(self):
        sc, env = self.open("inspection_hurdle_lane")
        beam_x = float(sc.steps[0][0])
        self.roll_to(env, 1.4)
        _, _, _, _, info = env.step(env._one_hot(JUMP))
        self.assertEqual(info["skill_name"], JUMP)
        self.assertIsNotNone(info["skill_plan"])
        self.assertEqual(info["skill_plan"]["shape"], "beam")
        self.assertEqual(info["skill_result"], "success")
        self.assertGreater(info["skill_control_steps"], env.k)
        self.assertGreater(float(env.env.data.qpos[0]), beam_x + 0.2)
        self.assertEqual(info["obstacle_hit"], 0)

    def test_without_self_timing_the_jump_fires_at_once(self):
        sc, env = self.open("inspection_hurdle_lane", self_timed=False)
        beam_x = float(sc.steps[0][0])
        self.roll_to(env, 1.4)
        _, _, _, _, info = env.step(env._one_hot(JUMP))
        self.assertIsNone(info["skill_plan"])
        self.assertNotEqual(info["skill_result"], "no_target")
        self.assertGreater(info["skill_control_steps"], env.k)
        self.assertLess(float(env.env.data.qpos[0]), beam_x)

    def test_no_target_costs_one_macro_step(self):
        sc, env = self.open("inspection_tank_farm")
        _, _, _, _, info = env.step(env._one_hot(JUMP))
        self.assertEqual(info["skill_name"], JUMP)
        self.assertEqual(info["skill_result"], "no_target")
        self.assertIsNone(info["skill_plan"])
        self.assertEqual(info["skill_control_steps"], env.k)


if __name__ == "__main__":
    unittest.main()
