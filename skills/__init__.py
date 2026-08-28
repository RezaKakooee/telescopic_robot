"""Low-level locomotion skill primitives for the 60-bar radial sphere robot.

Each skill is a **pure function** that takes the robot's current state
(quaternion, rod body-frame directions, max extension) and returns a
(n_bars,) array of rod extension targets.

Sub-modules
-----------
locomotion : Forward/reverse drive, strafe, speed control, stop.
interaction : Push against wall, brace against surfaces.
jumping : Vertical jump, forward jump from standstill, running hurdle leap.
falling : Stepping off a ledge and absorbing the landing.
runner : Drive a skill against a live env (arg routing + jump phase timing).
"""
from __future__ import annotations

from .locomotion import (
    move,
    move_forward,
    move_right,
    move_left,
    stop,
    go_fast,
    go_slow,
    reverse,
)
from .falling import fall_down
from .interaction import push_against_wall
from .jumping import (
    jump_to,
    jump_up,
    jump_forward_while_stopped,
    jump_forward_while_moving,
)

# ---------------------------------------------------------------------------
# Skill Registry — string name → callable
# ---------------------------------------------------------------------------
# Every registered skill can be dispatched by name:
#     targets = SKILL_REGISTRY["move_forward"](quat, dirs_body, max_extend, ...)
#
# To add a new skill in the future:
#   1. Write the function in the appropriate sub-module.
#   2. Add one entry to this dict.
# ---------------------------------------------------------------------------
SKILL_REGISTRY: dict[str, callable] = {
    # Locomotion. `move` is the gait; the rest are presets of it.
    "move": move,
    "move_forward": move_forward,
    "move_right": move_right,
    "move_left": move_left,
    "stop": stop,
    "go_fast": go_fast,
    "go_slow": go_slow,
    "reverse": reverse,
    # Interaction
    "push_against_wall": push_against_wall,
    # Jumping
    "jump_up": jump_up,
    "jump_forward_while_stopped": jump_forward_while_stopped,
    "jump_forward_while_moving": jump_forward_while_moving,
    "jump_to": jump_to,
    # Falling
    "fall_down": fall_down,
}

SKILL_NAMES = list(SKILL_REGISTRY.keys())


def execute_skill(name: str, *args, **kwargs):
    """Dispatch a skill by string name.

    Parameters
    ----------
    name : str
        One of :data:`SKILL_NAMES`.
    *args, **kwargs
        Forwarded to the skill function.

    Returns
    -------
    np.ndarray
        (n_bars,) rod extension targets in [0, max_extend].
    """
    if name not in SKILL_REGISTRY:
        raise ValueError(
            f"Unknown skill {name!r}; available: {SKILL_NAMES}"
        )
    return SKILL_REGISTRY[name](*args, **kwargs)
