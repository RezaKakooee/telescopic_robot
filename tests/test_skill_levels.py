"""The level split only pays for itself if the dependency arrow holds.

    high_level  may import mid and low
    mid_level   may import low
    low_level   may import neither

Without this check the folders are decoration. With it, a low-level skill
cannot quietly start depending on a path follower, which is the drift that
put `bar_targets` and `traverse_rough_terrain` out of sync in the first
place.
"""
import ast
import unittest
from pathlib import Path

SKILLS = Path(__file__).resolve().parent.parent / "skills"
#: What each level is allowed to import from, besides itself.
ALLOWED = {
    "low_level": set(),
    "mid_level": {"low_level"},
    "high_level": {"low_level", "mid_level"},
}
LEVELS = tuple(ALLOWED)


def _imported_levels(path: Path) -> set[str]:
    """Levels this module imports from, by absolute or relative import."""
    tree = ast.parse(path.read_text())
    own = path.parent.name
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:      # from ..low_level.x import y
                head = node.module.split(".")[0]
                if head in LEVELS:
                    found.add(head)
            elif node.module:                   # from skills.low_level.x import y
                parts = node.module.split(".")
                if len(parts) > 1 and parts[0] == "skills" and parts[1] in LEVELS:
                    found.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) > 1 and parts[0] == "skills" and parts[1] in LEVELS:
                    found.add(parts[1])
    return found - {own}


class SkillLevelTests(unittest.TestCase):
    def test_every_level_folder_exists_and_is_documented(self):
        for level in LEVELS:
            folder = SKILLS / level
            with self.subTest(level=level):
                self.assertTrue(folder.is_dir(), f"{folder} is missing")
                init = folder / "__init__.py"
                self.assertTrue(init.exists())
                self.assertTrue(ast.get_docstring(ast.parse(init.read_text())),
                                f"{level}/__init__.py needs a docstring saying "
                                "what belongs there")

    def test_the_dependency_arrow_points_one_way(self):
        for level, allowed in ALLOWED.items():
            for path in sorted((SKILLS / level).glob("*.py")):
                with self.subTest(module=f"{level}/{path.name}"):
                    illegal = _imported_levels(path) - allowed
                    self.assertEqual(
                        illegal, set(),
                        f"skills/{level}/{path.name} imports {sorted(illegal)}; "
                        f"{level} may only import {sorted(allowed) or 'nothing'}")

    def test_no_skill_module_sits_outside_a_level(self):
        """Everything but the package plumbing belongs to a level."""
        plumbing = {"__init__.py", "runner.py"}
        stray = sorted(p.name for p in SKILLS.glob("*.py") if p.name not in plumbing)
        self.assertEqual(stray, [],
                         "put these in low_level, mid_level or high_level")


if __name__ == "__main__":
    unittest.main()
