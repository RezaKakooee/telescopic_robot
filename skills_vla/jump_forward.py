"""Running hurdle leap. Mechanics: `skills.jump_forward_while_moving`.

The phase schedule (sprint, dip, launch, airborne, landing) is the verified one
from `skills.runner`. This wrapper only counts substeps and reports `done`.
"""
from __future__ import annotations

import numpy as np

from skills.low_level.jumping import jump_forward_while_moving
from .base import ParamSpec, RobotState, SkillResult, VLASkill
from .common import call_skill, jump_phase, resolve_heading

SCHEDULE = "jump_forward_while_moving"
# Substeps after touchdown before the option is reported complete.
ROLLOUT_STEPS = 20


class JumpForwardSkill(VLASkill):
    """Forward jump over hurdles and raised obstacles."""

    name: str = "jump_forward"
    summary: str = "running leap over hurdles and curbs with sprint-dip-launch-land sequence"
    params: tuple[ParamSpec, ...] = (
        ParamSpec("heading_ego", -np.pi / 2, np.pi / 2, 0.0, "jump azimuth relative to camera forward"),
        ParamSpec("power", 0.40, 1.0, 0.85, "stroke scaling for the launch impulse"),
    )
    is_multi_step: bool = True

    def _targets(self, state, d_world, phase, power):
        return call_skill(jump_forward_while_moving, state, d_hat=d_world, phase=phase, power=float(power))

    def act(self, state: RobotState, camera_heading: float = 0.0, *,
            heading_ego: float = 0.0, power: float = 0.85, **kwargs) -> np.ndarray:
        """Single-step launch targets."""
        d_world = resolve_heading(camera_heading, heading_ego, kwargs)
        return self._targets(state, d_world, "launch", power)

    def step(self, state: RobotState, substep: int, camera_heading: float = 0.0, *,
             heading_ego: float = 0.0, power: float = 0.85, **kwargs) -> SkillResult:
        d_world = resolve_heading(camera_heading, heading_ego, kwargs)
        phase, budget = jump_phase(SCHEDULE, substep, state)
        targets = self._targets(state, d_world, phase, power)
        done = substep >= budget - 1 or (phase == "landing" and substep >= budget // 2 + ROLLOUT_STEPS
                                         and _settled(state))
        return SkillResult(targets=targets, done=done, info={"phase": phase, "substep": substep, "power": power})


def _settled(state: RobotState) -> bool:
    return state.core_vz is None or abs(float(state.core_vz)) < 0.15


_INSTANCE = JumpForwardSkill()


def jump_forward(state: RobotState, substep: int = 0, camera_heading: float = 0.0,
                 heading_ego: float = 0.0, power: float = 0.85, **kwargs) -> SkillResult:
    return _INSTANCE.step(state, substep, camera_heading=camera_heading, heading_ego=heading_ego, power=power, **kwargs)
