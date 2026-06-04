"""Radial-sphere locomotion package.

Public API:
    from radial_sphere import run, Args
    from radial_sphere import fibonacci_sphere, sample_path, quat_to_rotmat
    from radial_sphere import build_robot_mjcf
    from radial_sphere import desired_direction, bar_targets

Modular pieces (see each module's docstring):
    geometry    — Fibonacci sphere, sinusoidal path, quaternion math (pure)
    mjcf        — MJCF XML generation for the radial-sphere robot
    controller  — open-loop locomotion controller (pure)
    config      — Args configclass (tyro-parseable)
    radial_sphere — run() wiring everything into a RoboVerse scenario
"""
from __future__ import annotations

from .config import Args
from .controller import bar_targets, desired_direction
from .geometry import (
    PATH_AMPLITUDE,
    PATH_LENGTH,
    PATH_WAVES,
    fibonacci_sphere,
    path_xy,
    quat_to_rotmat,
    sample_path,
)
from .mjcf import build_robot_mjcf
from .radial_sphere import run

__all__ = [
    "run",
    "Args",
    "fibonacci_sphere",
    "path_xy",
    "sample_path",
    "quat_to_rotmat",
    "PATH_LENGTH",
    "PATH_AMPLITUDE",
    "PATH_WAVES",
    "build_robot_mjcf",
    "desired_direction",
    "bar_targets",
]
