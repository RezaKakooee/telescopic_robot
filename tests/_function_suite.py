"""Let ``unittest discover`` collect modules written as plain test functions.

Several test modules here are scripts: bare ``test_*`` functions with plain
``assert`` statements and a ``__main__`` block that calls them in order. That
keeps them runnable with ``python tests/<file>.py``, which is how the docs
describe them.

The cost is that ``unittest discover`` walks straight past those modules. It
reports a green run while executing none of them. A module opts back in with
three lines at the bottom::

    def load_tests(loader, tests, pattern):
        from tests._function_suite import suite_from_module
        return suite_from_module(sys.modules[__name__])

Only zero-argument functions are collected. A ``test_*`` function that takes
required arguments is a helper called by the module's own ``main``, not a
standalone case.
"""
import inspect
import unittest


class _FunctionCase(unittest.TestCase):
    """One collected function, reported under its own name."""

    def __init__(self, module_name: str, name: str, fn):
        super().__init__("runTest")
        self._fn = fn
        self._id = f"{module_name}.{name}"

    def runTest(self):  # noqa: N802 - unittest's required method name
        self._fn()

    def id(self):
        return self._id

    def __str__(self):
        return self._id


def suite_from_module(module) -> unittest.TestSuite:
    """Build a suite from every zero-argument ``test_*`` function in `module`."""
    suite = unittest.TestSuite()
    for name, obj in sorted(vars(module).items()):
        if not name.startswith("test_") or not inspect.isfunction(obj):
            continue
        if obj.__module__ != module.__name__:
            continue
        required = [p for p in inspect.signature(obj).parameters.values()
                    if p.default is inspect.Parameter.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        if required:
            continue
        suite.addTest(_FunctionCase(module.__name__, name, obj))
    return suite
