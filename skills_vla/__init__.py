"""Modular VLA skills package for RoboBall.

Each skill is an independent module with its own parameter specification,
designed for vision-language-action policies using egocentric camera steering.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .base import ParamSpec, RobotState, SkillResult, VLASkill
from .common import ego_to_world_heading, finish_targets, rods_world, state_from_env, surface_frame

from .roll import RollSkill, roll
from .jump_forward import JumpForwardSkill, jump_forward
from .jump_gap import JumpGapSkill, jump_gap
from .traverse_rough import TraverseRoughSkill, traverse_rough
from .brake_stop import BrakeStopSkill, brake_stop
from .crawl_pipe import CrawlPipeSkill, crawl_pipe
from .flip import FlipSkill, flip

# Registry of independent skill instances
SKILL_REGISTRY: dict[str, VLASkill] = {
    "roll": RollSkill(),
    "jump_forward": JumpForwardSkill(),
    "jump_gap": JumpGapSkill(),
    "traverse_rough": TraverseRoughSkill(),
    "brake_stop": BrakeStopSkill(),
    "crawl_pipe": CrawlPipeSkill(),
    "flip": FlipSkill(),          # appended: earlier indices stay valid
}

# Synonyms and aliases for natural language compatibility
_ALIASES: dict[str, str] = {
    "move": "roll",
    "steer": "roll",
    "drive": "roll",
    "jump": "jump_forward",
    "jump_hurdle": "jump_forward",
    "leap": "jump_forward",
    "leap_gap": "jump_gap",
    "jump_chasm": "jump_gap",
    "turn_around": "flip",
    "reverse": "flip",
    "rough_terrain": "traverse_rough",
    "cobblestones": "traverse_rough",
    "stop": "brake_stop",
    "brake": "brake_stop",
    "halt": "brake_stop",
    "pipe": "crawl_pipe",
    "tunnel": "crawl_pipe",
    "conduit": "crawl_pipe",
}

SKILL_NAMES = tuple(SKILL_REGISTRY.keys())

#: How SkillArbitrationEnv (skill_backend="skills") executes each VLA skill.
#: jump_forward is the running leap for barriers (hurdle, wall, stairs);
#: jump_gap is the velocity-aimed hop for platforms and gaps (jump_to).
ENV_SKILL_MAP: dict[str, str] = {
    "roll": "move",
    "jump_forward": "jump_forward_while_moving",
    "jump_gap": "jump_forward_while_moving",
    "traverse_rough": "traverse_rough_terrain",
    "brake_stop": "stop",
    "crawl_pipe": "crawl_pipe",
    "flip": "flip",
}


def get_skill(name: str) -> VLASkill:
    """Retrieve skill instance by canonical name or alias."""
    canonical = _ALIASES.get(name.lower(), name.lower())
    if canonical not in SKILL_REGISTRY:
        raise KeyError(
            f"Unknown VLA skill '{name}'. Available skills: {list(SKILL_REGISTRY.keys())} "
            f"or aliases: {list(_ALIASES.keys())}"
        )
    return SKILL_REGISTRY[canonical]


def dispatch_vla_action(
    skill_name: str,
    action_params: np.ndarray | list[float] | dict[str, Any],
    state: RobotState,
    camera_heading: float = 0.0,
    substep: int = 0,
) -> SkillResult:
    """Unified VLA dispatch entry point.

    Parameters
    ----------
    skill_name : str
        Name or alias of skill to execute (e.g. 'roll', 'jump_forward', 'brake_stop').
    action_params : np.ndarray, list, or dict
        Either a normalized continuous parameter vector in [-1, 1] predicted by a VLA,
        or a dictionary of physical keyword arguments.
    state : RobotState
        Current proprioceptive robot state.
    camera_heading : float
        Current yaw of active camera in radians.
    substep : int
        Current substep index for multi-step sequenced maneuvers (jumps).

    Returns
    -------
    SkillResult
        Contains `.targets` (60-element rod target vector) and `.done` boolean flag.
    """
    skill = get_skill(skill_name)

    if isinstance(action_params, dict):
        kwargs = dict(action_params)
    else:
        kwargs = skill.scale_params(action_params)

    if skill.is_multi_step:
        return skill.step(state, substep=substep, camera_heading=camera_heading, **kwargs)
    else:
        targets = skill.act(state, camera_heading=camera_heading, **kwargs)
        return SkillResult(targets=targets, done=True, info={"skill": skill.name})


__all__ = [
    "RobotState",
    "ParamSpec",
    "SkillResult",
    "VLASkill",
    "ego_to_world_heading",
    "rods_world",
    "surface_frame",
    "finish_targets",
    "state_from_env",
    "RollSkill",
    "roll",
    "JumpForwardSkill",
    "jump_forward",
    "JumpGapSkill",
    "jump_gap",
    "TraverseRoughSkill",
    "traverse_rough",
    "BrakeStopSkill",
    "brake_stop",
    "CrawlPipeSkill",
    "crawl_pipe",
    "FlipSkill",
    "flip",
    "SKILL_REGISTRY",
    "SKILL_NAMES",
    "ENV_SKILL_MAP",
    "get_skill",
    "dispatch_vla_action",
]
