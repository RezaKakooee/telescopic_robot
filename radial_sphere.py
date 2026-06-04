"""Radial-sphere locomotion demo — entry point.

A sphere with N telescoping bars (Fibonacci-distributed on the unit sphere)
rolls along a sinusoidal path on the floor.

Only --sim mujoco is supported: we ship only an MJCF for the custom robot.

Usage:
    python radial_sphere.py --headless
    python radial_sphere.py --headless --n-bars 80 --n-sim-steps 6000
"""
from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from radial_sphere.config import Args
from radial_sphere.radial_sphere import run

if __name__ == "__main__":
    run(tyro.cli(Args))
