"""Centred, wall-free rolling inside a round conduit. Mechanics: `skills.crawl_pipe`."""
from __future__ import annotations

import numpy as np

from skills.low_level.pipe_crawling import crawl_pipe as _crawl
from .base import ParamSpec, RobotState, VLASkill
from .common import call_skill, resolve_heading


class CrawlPipeSkill(VLASkill):
    """Roll along a pipe axis, steering to the centre, rods capped short of the wall."""

    name: str = "crawl_pipe"
    summary: str = "roll through a round conduit without touching its walls"
    params: tuple[ParamSpec, ...] = (
        ParamSpec("heading_ego", -np.pi, np.pi, 0.0, "pipe axis direction relative to camera view"),
        ParamSpec("speed", 0.3, 0.9, 0.6, "cruising speed inside the pipe (m/s)"),
    )
    is_multi_step: bool = False

    def act(self, state: RobotState, camera_heading: float = 0.0, *,
            heading_ego: float = 0.0, speed: float = 0.6, pipe_radius: float = 0.44,
            axis_offset=(0.0, -0.25), **kwargs) -> np.ndarray:
        d_world = resolve_heading(camera_heading, heading_ego, kwargs)
        lin_vel = None if state.lin_vel is None else np.asarray(state.lin_vel, dtype=np.float64)[:2]
        return call_skill(_crawl, state, d_hat=d_world, speed=float(speed), pipe_radius=float(pipe_radius),
                          axis_offset=tuple(axis_offset), min_offset=state.min_offset, lin_vel=lin_vel,
                          rod_mechanism=state.rod_mechanism)


_INSTANCE = CrawlPipeSkill()


def crawl_pipe(state: RobotState, camera_heading: float = 0.0, heading_ego: float = 0.0,
               speed: float = 0.6, **kwargs) -> np.ndarray:
    return _INSTANCE.act(state, camera_heading=camera_heading, heading_ego=heading_ego, speed=speed, **kwargs)
