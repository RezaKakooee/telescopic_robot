"""Rolling with egocentric steering. Mechanics: `skills.move`."""
from __future__ import annotations

import numpy as np

from skills.low_level.locomotion import move
from .base import ParamSpec, RobotState, VLASkill
from .common import call_skill, resolve_heading


class RollSkill(VLASkill):
    """Continuous directional rolling with egocentric steering."""

    name: str = "roll"
    summary: str = "continuous directional rolling with egocentric steering"
    params: tuple[ParamSpec, ...] = (
        ParamSpec("heading_ego", -np.pi, np.pi, 0.0,
                  "egocentric steering angle in radians (0 = straight ahead in camera, +pi/2 = left)"),
        ParamSpec("speed", 0.2, 1.6, 1.1, "target cruising speed in m/s"),
    )
    is_multi_step: bool = False

    def act(self, state: RobotState, camera_heading: float = 0.0, *,
            heading_ego: float = 0.0, speed: float = 1.1, **kwargs) -> np.ndarray:
        d_world = resolve_heading(camera_heading, heading_ego, kwargs)
        lin_vel = None if state.lin_vel is None else np.asarray(state.lin_vel, dtype=np.float64)[:2]
        return call_skill(move, state, d_hat=d_world, speed=float(speed), lin_vel=lin_vel,
                          min_offset=state.min_offset, rod_mechanism=state.rod_mechanism,
                          cross_track_error=float(kwargs.get("cross_track_error", 0.0)))


_INSTANCE = RollSkill()


def roll(state: RobotState, camera_heading: float = 0.0, heading_ego: float = 0.0,
         speed: float = 1.1, **kwargs) -> np.ndarray:
    return _INSTANCE.act(state, camera_heading=camera_heading, heading_ego=heading_ego, speed=speed, **kwargs)
