"""One calling convention across the whole skill registry.

Skills return three different shapes. Most hand back a bare target array.
Three take ``return_metadata`` and return a pair. ``wall_of_death`` always
returns a tuple, pairing its targets with an info dict.

A caller should not have to know which is which. ``execute_skill`` answers
``return_metadata`` for every registered name, so these checks walk the whole
registry and insist on it.
"""
import os
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")
import inspect

import numpy as np

from radial_sphere.geometry import fibonacci_sphere
from skills import SKILL_REGISTRY, execute_skill
from skills.low_level.bowl_riding import Bowl

#: Arguments for the skills that need real scene geometry. Anything a skill
#: requires and this map does not cover makes the coverage test fail, which is
#: the point: a new skill has to declare what it needs.
SCENE_ARGS = {
    "wall_normal": np.array([0.0, 1.0, 0.0]),
    "wall_axis": np.array([0.0, 1.0, 0.0]),
    "v_z": -0.4,
    "phase_ratio": 0.5,
    "side": 1,
    "pos": np.array([0.0, 0.0, 0.30]),
    "vel": np.array([0.5, 0.0, 0.0]),
    "normal": np.array([0.0, 0.0, 1.0]),
    "along": np.array([1.0, 0.0, 0.0]),
    "ball_xy": np.array([0.5, 0.2]),
    "ball_pos": np.array([0.5, 0.2, 0.30]),
    "lin_vel": np.array([0.5, 0.1, 0.0]),
    "d_hat": np.array([1.0, 0.0]),
    "cones": np.array([[1.0, 0.0, 0.12], [2.0, 0.5, 0.12], [3.0, -0.5, 0.12]]),
    "path_pts": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5], [3.0, 0.5]]),
    "r_cmd": 1.8,
    # A bowl is a radius-against-height profile, flat floor then banked wall.
    "bowl": Bowl(r=np.array([1.6, 2.0, 2.3, 2.4, 2.4]),
                 z=np.array([0.0, 0.35, 0.75, 1.10, 1.60]),
                 floor_r=1.6, lateral_limit=0.35, ride_limit=1.5),
    "phase": "sprint",
    "wall_dist": 0.35,
    # wall_run projects travel against a 3-D wall normal, so this is 3-D.
    "travel": np.array([1.0, 0.0, 0.0]),
}

MAX_EXTEND = 0.16


def _call_args(fn):
    """Fill every required argument of `fn` from SCENE_ARGS."""
    kwargs, missing = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        if name in ("quat", "dirs_body", "max_extend"):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        if name in SCENE_ARGS:
            kwargs[name] = SCENE_ARGS[name]
        else:
            missing.append(name)
    return kwargs, missing


class SkillInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.dirs = fibonacci_sphere(60).astype(np.float32)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])

    def test_every_registered_name_answers_return_metadata(self):
        uncovered = []
        for name, fn in sorted(SKILL_REGISTRY.items()):
            kwargs, missing = _call_args(fn)
            if missing:
                uncovered.append((name, missing))
                continue
            with self.subTest(skill=name):
                targets, meta = execute_skill(
                    name, self.quat, self.dirs, MAX_EXTEND,
                    return_metadata=True, **kwargs)
                self.assertEqual(np.asarray(targets).shape, (60,))
                self.assertEqual(meta["skill"], fn.__name__)
                self.assertEqual(meta["requested_as"], name)
                self.assertIn("n_extended", meta)
                self.assertIn("stroke_fraction", meta)
        self.assertEqual(uncovered, [],
                         "add these arguments to SCENE_ARGS so the skill is covered")

    def test_aliases_report_the_function_they_reach(self):
        """An alias must say which skill actually ran."""
        targets, meta = execute_skill("rough_terrain", self.quat, self.dirs,
                                      MAX_EXTEND, d_hat=[1.0, 0.0],
                                      return_metadata=True)
        self.assertEqual(meta["skill"], "traverse_rough_terrain")
        self.assertEqual(meta["requested_as"], "rough_terrain")

    def test_skills_with_native_metadata_keep_their_own_keys(self):
        _, meta = execute_skill("stay_in_boundary", self.quat, self.dirs,
                                MAX_EXTEND, ball_xy=np.array([0.5, 0.2]),
                                boundary_radius=2.0, return_metadata=True)
        for key in ("sub_skill", "action_name", "distance_to_edge"):
            self.assertIn(key, meta)
        self.assertEqual(meta["skill"], "stay_in_boundary")

    def test_always_tuple_skills_are_unpacked(self):
        """`wall_of_death` has no return_metadata flag; it always pairs up."""
        targets, meta = execute_skill(
            "wall_of_death", self.quat, self.dirs, MAX_EXTEND,
            ball_pos=SCENE_ARGS["ball_pos"], lin_vel=SCENE_ARGS["lin_vel"],
            r_cmd=SCENE_ARGS["r_cmd"], bowl=SCENE_ARGS["bowl"],
            return_metadata=True)
        self.assertEqual(np.asarray(targets).shape, (60,))
        self.assertEqual(meta["skill"], "wall_of_death")
        self.assertIn("n_extended", meta)

    def test_public_skills_missing_from_the_registry(self):
        """Some public skill functions cannot be reached by name.

        `execute_skill` can only give a uniform interface to what is
        registered, so this records the gap rather than hiding it.
        """
        from skills.low_level import climbing
        unregistered = {fn.__name__ for fn in SKILL_REGISTRY.values()}
        for name in ("chimney_friction_servo", "chimney_step_down",
                     "cylinder_spiral_climb"):
            self.assertTrue(hasattr(climbing, name))
            self.assertNotIn(name, unregistered,
                             f"{name} is registered now; drop it from this list")

    def test_default_call_still_returns_a_bare_array(self):
        out = execute_skill("move", self.quat, self.dirs, MAX_EXTEND,
                            d_hat=[1.0, 0.0])
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (60,))

    def test_every_skill_summarises_itself_in_one_sentence(self):
        for fn in dict.fromkeys(SKILL_REGISTRY.values()):
            doc = inspect.getdoc(fn) or ""
            first = doc.splitlines()[0] if doc else ""
            with self.subTest(skill=fn.__name__):
                self.assertTrue(first, "skill has no docstring")
                self.assertTrue(first[0].isupper(), f"{first!r} must start a sentence")
                self.assertTrue(first.rstrip().endswith((".", "!")),
                                f"{first!r} must end a sentence")

    def test_wide_skills_document_their_knobs(self):
        """A skill with many options must have a structured section.

        `Phases` counts: the phase machines explain each option where it is
        used rather than in a flat list, and that reads better for them.
        """
        import re
        headings = ("Parameters", "Args", "Phases")
        for fn in dict.fromkeys(SKILL_REGISTRY.values()):
            n_kwargs = sum(1 for p in inspect.signature(fn).parameters.values()
                           if p.default is not inspect.Parameter.empty)
            if n_kwargs < 8:
                continue
            doc = inspect.getdoc(fn) or ""
            with self.subTest(skill=fn.__name__, kwargs=n_kwargs):
                self.assertTrue(
                    any(re.search(rf"^{h}\b", doc, re.M) for h in headings),
                    f"{fn.__name__} takes {n_kwargs} options and documents none")

    def test_no_skill_declares_a_parameter_it_never_uses(self):
        """A parameter the body never reads is a promise the skill cannot keep.

        `straddle_gap` documented `speed`, `gap_half_width` and `min_offset`
        and used none of them, so `run_gap.py --speed` did nothing at all.
        `slalom` accepted `lin_vel` and dropped it.
        """
        import ast
        import pathlib

        wanted = {fn.__name__ for fn in SKILL_REGISTRY.values()}
        dead = {}
        for path in pathlib.Path("skills").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.FunctionDef) or node.name not in wanted:
                    continue
                args = node.args
                declared = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
                used = {n.id for n in ast.walk(node)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                used |= {k.arg for k in ast.walk(node)
                         if isinstance(k, ast.keyword) and k.arg}
                unused = [d for d in declared if d not in used]
                if unused:
                    dead[node.name] = unused
        self.assertEqual(dead, {},
                         "either use the parameter or drop it from the signature")

    def test_no_module_is_named_after_a_skill(self):
        """A module and a skill must never share a name.

        When they did, the attribute resolved differently depending on
        whether `__init__` happened to import a symbol of that name.
        `skills.slalom` was the function, so `import skills.slalom` then
        `skills.slalom.slalom` raised AttributeError, while the identical
        `skills.stairs.climb_stairs` worked. Callers could not predict which
        they would get.
        """
        import pkgutil

        import skills
        modules = {m.name for m in pkgutil.iter_modules(skills.__path__)}
        clashes = sorted(modules & set(SKILL_REGISTRY))
        self.assertEqual(clashes, [],
                         "rename the module, not the skill: a skill name is "
                         "part of the public API and a module name is not")

    def test_unknown_name_is_rejected(self):
        with self.assertRaises(ValueError):
            execute_skill("no_such_skill", self.quat, self.dirs, MAX_EXTEND)


if __name__ == "__main__":
    unittest.main()
