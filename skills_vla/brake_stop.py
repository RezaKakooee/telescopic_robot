"""Braking to a held stance. Mechanics: `skills.stop`."""
from __future__ import annotations

import numpy as np

from skills.low_level.locomotion import stop
from .base import ParamSpec, RobotState, VLASkill
from .common import call_skill


class BrakeStopSkill(VLASkill):
    """Shed velocity and hold a stationary stance."""

    name: str = "brake_stop"
    summary: str = "active dynamic braking to rapidly shed velocity and hold stationary stance"
    params: tuple[ParamSpec, ...] = (
        ParamSpec("strength", 0.5, 3.2, 1.8, "braking gain: higher brakes harder, lower halts smoothly"),
    )
    is_multi_step: bool = False

    def act(self, state: RobotState, camera_heading: float = 0.0, *,
            strength: float = 1.8, **kwargs) -> np.ndarray:
        lin_vel = None if state.lin_vel is None else np.asarray(state.lin_vel, dtype=np.float64)[:2]
        return call_skill(stop, state, lin_vel=lin_vel, brake_gain=float(strength))


_INSTANCE = BrakeStopSkill()


def brake_stop(state: RobotState, camera_heading: float = 0.0, strength: float = 1.8, **kwargs) -> np.ndarray:
    return _INSTANCE.act(state, camera_heading=camera_heading, strength=strength, **kwargs)
