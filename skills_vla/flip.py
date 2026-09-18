"""Turn the travel direction around. Mechanics: `skills.stop` for the flip step.

The ball has no front: "forward" is whatever direction the gait pushes. A
flip is a decision, not a motion. It brakes for one step and from then on
`roll` drives the other way, until the next flip. There is no reverse skill:
the ball always rolls forward, it just flips first.
"""
from __future__ import annotations

import numpy as np

from skills.low_level.locomotion import stop
from .base import ParamSpec, RobotState, VLASkill
from .common import call_skill


class FlipSkill(VLASkill):
    """Turn the travel direction around; roll then goes the other way."""

    name: str = "flip"
    summary: str = "turn the travel direction around (brake for this step); roll then goes the other way"
    params: tuple[ParamSpec, ...] = ()
    is_multi_step: bool = False

    def act(self, state: RobotState, camera_heading: float = 0.0, **kwargs) -> np.ndarray:
        lin_vel = None if state.lin_vel is None else np.asarray(state.lin_vel, dtype=np.float64)[:2]
        return call_skill(stop, state, lin_vel=lin_vel)


_INSTANCE = FlipSkill()


def flip(state: RobotState, camera_heading: float = 0.0, **kwargs) -> np.ndarray:
    return _INSTANCE.act(state, camera_heading=camera_heading, **kwargs)
