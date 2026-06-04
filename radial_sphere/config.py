"""Args configclass for the radial-sphere demo.

All tuneable parameters live here so callers only need:

    from radial_sphere.config import Args
    import tyro
    args = tyro.cli(Args)
"""
from __future__ import annotations

from typing import Literal

from loguru import logger as log

from metasim.utils import configclass


@configclass
class Args:
    """Arguments for the radial-sphere demo."""

    # Only mujoco for now — we don't have USD/URDF for this custom robot.
    sim: Literal["mujoco"] = "mujoco"
    n_bars: int = 60
    max_extend: float = 0.12
    sphere_radius: float = 0.15
    kp: float = 900.0
    kv: float = 22.0
    # Locomotion gains. Lower back_gain → gentler push → slower ball.
    back_gain: float = 0.5
    down_gain: float = 0.4
    # ~ 4 s of physics @ dt=0.002 by default.  Bump for longer runs.
    n_sim_steps: int = 2000
    # Capture one obs every N sim steps → controls effective video frame rate.
    frame_every: int = 17
    # Initial settle steps where bars are held at base extension.
    n_settle_steps: int = 300
    num_envs: int = 1
    headless: bool = True

    def __post_init__(self):
        log.info(f"Args: {self}")
