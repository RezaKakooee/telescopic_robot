"""Aimed standing leap across a gap. Mechanics: `skills.jump_to`.

`jump_to` servos the burn against the measured velocity, so two calls with
the same targets land in the same place. Phases (crouch, takeoff, airborne,
landing) follow the `jump_forward_while_stopped` schedule in `skills.runner`.
"""
from __future__ import annotations

import numpy as np

from skills.low_level.jumping import jump_to
from .base import ParamSpec, RobotState, SkillResult, VLASkill
from radial_sphere.terrain_probe import plan_jump
from .common import call_skill, jump_phase, resolve_heading

SCHEDULE = "jump_forward_while_stopped"
ROLLOUT_STEPS = 20


class JumpGapSkill(VLASkill):
    """Aimed gap and valley jumping skill."""

    name: str = "jump_gap"
    summary: str = "aimed leap across trenches, chasms, and box platforms"
    params: tuple[ParamSpec, ...] = (
        ParamSpec("heading_ego", -np.pi / 2, np.pi / 2, 0.0, "jump azimuth relative to camera view"),
        ParamSpec("vx_target", 0.3, 1.2, 0.6, "target forward velocity at takeoff (m/s)"),
        ParamSpec("vz_target", 1.8, 3.6, 2.6, "target vertical velocity at takeoff (m/s)"),
    )
    is_multi_step: bool = True

    def _targets(self, state, d_world, phase, vx_target, vz_target):
        vel = None
        if state.lin_vel is not None:
            v = np.asarray(state.lin_vel, dtype=np.float64)
            vz = float(state.core_vz) if state.core_vz is not None else (v[2] if v.shape[0] > 2 else 0.0)
            vel = np.array([v[0], v[1], vz])
        return call_skill(jump_to, state, d_hat=d_world, phase=phase, vel=vel,
                          vx_target=float(vx_target), vz_target=float(vz_target), wall_lock=True)

    def can_start(self, terrain) -> bool:
        """A jump needs an edge to aim at: a beam, a tread, a deck or a trench ahead."""
        return terrain is None or plan_jump(terrain) is not None

    def plan(self, terrain, ground: float | None = None) -> dict:
        """The edge to aim at and the distance before it to fire (``terrain_probe.plan_jump``).

        ``ground`` is the floor the ball stands on; pass it when known (inside
        a pipe the first ray sample is the roof, not the floor)."""
        if terrain is None:
            return {}
        return plan_jump(terrain, ground=ground) or {}

    def act(self, state: RobotState, camera_heading: float = 0.0, *,
            heading_ego: float = 0.0, vx_target: float = 0.6, vz_target: float = 2.6, **kwargs) -> np.ndarray:
        d_world = resolve_heading(camera_heading, heading_ego, kwargs)
        return self._targets(state, d_world, "takeoff", vx_target, vz_target)

    def step(self, state: RobotState, substep: int, camera_heading: float = 0.0, *,
             heading_ego: float = 0.0, vx_target: float = 0.6, vz_target: float = 2.6, **kwargs) -> SkillResult:
        d_world = resolve_heading(camera_heading, heading_ego, kwargs)
        phase, budget = jump_phase(SCHEDULE, substep, state)
        targets = self._targets(state, d_world, phase, vx_target, vz_target)
        done = substep >= budget - 1
        return SkillResult(targets=targets, done=done,
                           info={"phase": phase, "substep": substep, "vx_target": vx_target, "vz_target": vz_target})


_INSTANCE = JumpGapSkill()


def jump_gap(state: RobotState, substep: int = 0, camera_heading: float = 0.0, heading_ego: float = 0.0,
             vx_target: float = 0.6, vz_target: float = 2.6, **kwargs) -> SkillResult:
    return _INSTANCE.step(state, substep, camera_heading=camera_heading, heading_ego=heading_ego,
                          vx_target=vx_target, vz_target=vz_target, **kwargs)
