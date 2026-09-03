"""Low-level locomotion skill primitives for the 60-bar radial sphere robot.

Each skill is a **pure function** that takes the robot's current state
(quaternion, rod body-frame directions, max extension) and returns a
(n_bars,) array of rod extension targets.

Sub-modules
-----------
locomotion : Forward/reverse drive, strafe, speed control, stop, and
    ``surface_drive`` -- the same gait aimed at any surface, not just the floor.
interaction : Push against wall, brace against surfaces.
jumping : Vertical jump, forward jump from standstill, running hurdle leap.
falling : Stepping off a ledge and absorbing the landing.
wall_of_death : Spiralling up the inside of a banked drome bowl.
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
    circle,
    straddle_gap,
    surface_drive,
)
from .wall_of_death import (
    wall_of_death,
    Bowl,
    advance_radius,
    descend_radius,
    surface_frame,
    surface_normal,
    reach_caps,
    grip_margin,
)
from .wall_run import wall_run, wall_reach, next_phase as wall_run_next_phase
from .falling import fall_down
from .interaction import push_against_wall, chimney_climb
from .jumping import (
    jump_to,
    jump_up,
    jump_forward_while_stopped,
    jump_forward_while_moving,
)
from .slalom import slalom, training_cones, curved_slalom

from .stairs import climb_stairs

# ---------------------------------------------------------------------------
# Skill Registry — string name → callable
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
    "circle": circle,
    "straddle_gap": straddle_gap,
    "straddle": straddle_gap,
    "surface_drive": surface_drive,
    "wall_ride": surface_drive,
    "wall_of_death": wall_of_death,
    "motordrome": wall_of_death,
    "wall_run": wall_run,
    "horizontal_wall_run": wall_run,
    "slalom": slalom,
    "training_cones": training_cones,
    "curved_slalom": curved_slalom,
    "curved_training_cones": curved_slalom,
    # Interaction
    "push_against_wall": push_against_wall,
    "chimney_climb": chimney_climb,
    "chimney": chimney_climb,
    "vertical_climb": chimney_climb,
    # Jumping
    "jump_up": jump_up,
    "jump_forward_while_stopped": jump_forward_while_stopped,
    "jump_forward_while_moving": jump_forward_while_moving,
    "jump_to": jump_to,
    # Stairs & Obstacle Traversal
    "stairs": climb_stairs,
    "climb_stairs": climb_stairs,
    "step_vault": climb_stairs,
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
