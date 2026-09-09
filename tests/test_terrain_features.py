"""Terrain feature records and the per-feature MJCF builders.

``build_mujoco_scene_mjcf`` was one eleven-hundred-line function, so no test
could build a single feature and look at it. The floor-slab bug lived there
unseen. These checks work on one feature at a time.
"""
import os
import re
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

from radial_sphere import mjcf_features as features
from radial_sphere import terrain as T
from radial_sphere.config import load_config
from radial_sphere.mujoco_mjcf import build_mujoco_scene_mjcf
from radial_sphere.scenario import KINDS, generate_scenario

RECORD_FOR = {
    "gaps": T.Gap,
    "sand_patches": T.SandPatch,
    "stones": T.StoneField,
    "steps": T.Step,
    "ramps": T.Ramp,
    "staircases": T.Staircase,
    "pipes": T.Pipe,
    "vertical_cylinders": T.VerticalCylinder,
    "motordromes": T.Motordrome,
    "cones": T.Cone,
    "yardlines": T.Yardline,
}


class _Bag:
    """Minimal stand-in for a Scenario carrying one feature."""

    def __init__(self, **features_):
        for name, value in features_.items():
            setattr(self, name, value)


class TerrainRecordTests(unittest.TestCase):
    def test_records_are_still_tuples(self):
        """Existing consumers index these rows, so they must stay tuples."""
        gap = T.Gap(1.0, 2.0, 0.2, 0.3, 0.05)
        self.assertEqual(gap[0], 1.0)
        self.assertEqual(gap[4], 0.05)
        self.assertEqual(len(gap), 5)
        self.assertEqual(gap.half_y, 0.3)

    def test_rows_accepts_lists_tuples_and_records(self):
        made = T.rows(T.Gap, [[1, 2, 0.2, 0.3], (4, 5, 0.1, 0.1, 0.06),
                              T.Gap(7, 8, 0.2, 0.2, 0.03)])
        self.assertEqual(len(made), 3)
        self.assertEqual(made[0].depth, 0.10, "missing depth takes the default")
        self.assertEqual(made[1].depth, 0.06)
        self.assertEqual(made[2].x, 7)

    def test_rows_of_none_is_empty(self):
        self.assertEqual(T.rows(T.Gap, None), ())

    def test_computed_defaults_are_resolved_once(self):
        """A pipe's outer radius is derived from its bore, not a constant."""
        pipe = T.rows(T.Pipe, [[1.0, 0.0, 10.5, 0.38]])[0]
        self.assertAlmostEqual(pipe.outer_radius, 0.395)
        kept = T.rows(T.Pipe, [[1.0, 0.0, 10.5, 0.38, 0.42]])[0]
        self.assertAlmostEqual(kept.outer_radius, 0.42)
        cyl = T.rows(T.VerticalCylinder, [[0.0, 0.0, 4.5, 0.30]])[0]
        self.assertAlmostEqual(cyl.outer_radius, 0.32)

    def test_every_scenario_kind_fits_its_records(self):
        """The shipped scenarios must match the shapes declared in terrain.py."""
        cfg = load_config("configs/rl/standing_jump_showcase.yaml")
        checked = 0
        for kind in KINDS:
            try:
                scenario = generate_scenario(kind, cfg, seed=42)
            except Exception:
                continue
            for attr, record in RECORD_FOR.items():
                raw = getattr(scenario, attr, None)
                if raw is None or len(raw) == 0:
                    continue
                T.rows(record, raw)  # raises if the shape drifted
                checked += 1
        self.assertGreater(checked, 5, "expected several kinds to carry features")


class FeatureBuilderTests(unittest.TestCase):
    def test_one_gap_builds_its_curbs_and_pit_floor(self):
        xml = "\n".join(features.gaps_xml(_Bag(gaps=[[0.5, 0.0, 0.2, 0.7, 0.05]])))
        for name in ("gap_curb_left_0", "gap_curb_right_0",
                     "gap_end_a_0", "gap_end_b_0", "gap_pit_floor_0"):
            self.assertIn(name, xml)
        pit = re.search(r'name="gap_pit_floor_0".*?pos="[-\d.]+ [-\d.]+ (-?[\d.]+)"', xml)
        self.assertIsNotNone(pit)
        self.assertAlmostEqual(float(pit.group(1)), -0.05, places=4)

    def test_no_feature_means_no_geoms(self):
        for builder in (features.gaps_xml, features.stone_field_xml,
                        features.steps_xml, features.cone_xml,
                        features.sand_patch_xml, features.ramp_xml):
            self.assertEqual(builder(_Bag()), [])

    def test_stone_count_is_honoured(self):
        xml = "\n".join(features.stone_field_xml(
            _Bag(stones=[[0.0, 0.0, 1.0, 1.0, 7, 0.05]])))
        self.assertEqual(len(re.findall(r'name="rock_0_\d+"', xml)), 7)

    def test_circular_scatter_stays_inside_its_radius(self):
        xml = "\n".join(features.stone_field_xml(
            _Bag(stones=[[0.0, 0.0, 1.0, 1.0, 40, 0.04, True]])))
        for x, y in re.findall(r'name="rock_0_\d+".*?pos="(-?[\d.]+) (-?[\d.]+)', xml):
            self.assertLessEqual((float(x) ** 2 + float(y) ** 2) ** 0.5, 1.0 + 1e-6)

    def test_floor_slabs_around_a_pit_are_solid_blocks(self):
        """A thin slab lets a fast rod tip cross the mid-plane and drop through.

        The rod then rests on the base plane below and the robot stalls. Slabs
        must be thick, and must reach below the deepest pit floor.
        """
        cfg = load_config("configs/rl/standing_jump_showcase.yaml")
        scenario = generate_scenario("goal", cfg, seed=7)
        scenario.gaps = [[0.5, 0.0, 0.2, 0.7, 0.05]]
        xml, _ = build_mujoco_scene_mjcf(scenario=scenario, n_bars=60,
                                         sphere_radius=0.15, max_extend=0.16)
        slabs = re.findall(r'name="floor_slab_\d+".*?size="[\d.]+ [\d.]+ ([\d.]+)"', xml)
        self.assertTrue(slabs, "a scenario with pits must build floor slabs")
        for half_thickness in slabs:
            self.assertGreaterEqual(float(half_thickness), 0.10,
                                    "floor slab is thin enough to be punched through")
        tops = re.findall(r'name="floor_slab_\d+".*?pos="[-\d.]+ [-\d.]+ (-[\d.]+)"', xml)
        for centre, half in zip(tops, slabs):
            self.assertAlmostEqual(float(centre) + float(half), 0.0, places=6,
                                   msg="walkable floor surface must sit at z = 0")


if __name__ == "__main__":
    unittest.main()
