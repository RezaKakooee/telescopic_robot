"""Compliant gait for cobbles and curbs. Mechanics: `skills.traverse_rough_terrain`."""
from __future__ import annotations

import numpy as np

from skills.low_level.terrain_following import traverse_rough_terrain
from .base import ParamSpec, RobotState, VLASkill
from .common import call_skill, resolve_heading


class TraverseRoughSkill(VLASkill):
    """Active suspension drive across rough terrain."""

    name: str = "traverse_rough"
    summary: str = "active compliant suspension gait for cobblestones, steps, and rough terrain"
    params: tuple[ParamSpec, ...] = (
        ParamSpec("heading_ego", -np.pi, np.pi, 0.0, "egocentric travel heading relative to camera view"),
        ParamSpec("speed", 0.3, 1.3, 0.8, "cruising speed in m/s across rough surface"),
        ParamSpec("curb_boost", 1.5, 4.0, 2.6, "gain boost on rods touching step faces, to vault edges"),
    )
    is_multi_step: bool = False

    def act(self, state: RobotState, camera_heading: float = 0.0, *,
            heading_ego: float = 0.0, speed: float = 0.8, curb_boost: float = 2.6, **kwargs) -> np.ndarray:
        d_world = resolve_heading(camera_heading, heading_ego, kwargs)
        lin_vel = None if state.lin_vel is None else np.asarray(state.lin_vel, dtype=np.float64)[:2]
        return call_skill(
            traverse_rough_terrain, state, d_hat=d_world, speed=float(speed), min_offset=state.min_offset,
            lin_vel=lin_vel, core_z=state.core_z, core_vz=state.core_vz,
            contact_forces=state.contact_forces, terrain_clearances=state.terrain_clearances,
            curb_boost_gain=float(curb_boost),
            suspension_state=kwargs.get("suspension_state"), control_dt=float(kwargs.get("control_dt", 0.01)),
        )


_INSTANCE = TraverseRoughSkill()


def traverse_rough(state: RobotState, camera_heading: float = 0.0, heading_ego: float = 0.0,
                   speed: float = 0.8, curb_boost: float = 2.6, **kwargs) -> np.ndarray:
    return _INSTANCE.act(state, camera_heading=camera_heading, heading_ego=heading_ego,
                         speed=speed, curb_boost=curb_boost, **kwargs)
