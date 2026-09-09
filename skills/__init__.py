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

import inspect
from functools import lru_cache

import numpy as np

from .low_level.suspension import SuspensionGains, SuspensionState, apply_suspension
from .low_level.locomotion import (
    move,
    turn,
    move_forward,
    move_right,
    move_left,
    stop,
    go_fast,
    go_slow,
    reverse,
    circle,
    curve,
    straddle_gap,
    surface_drive,
)
from .low_level.terrain_following import traverse_rough_terrain
from .mid_level.navigation import follow_path, stay_in_boundary
from .low_level.bowl_riding import (
    wall_of_death,
    Bowl,
    advance_radius,
    descend_radius,
    surface_frame,
    surface_normal,
    reach_caps,
    grip_margin,
)
from .low_level.wall_running import wall_run, wall_reach, next_phase as wall_run_next_phase
from .low_level.falling import fall_down
from .low_level.climbing import push_against_wall, chimney_climb
from .low_level.jumping import (
    jump_to,
    jump_up,
    jump_forward_while_stopped,
    jump_forward_while_moving,
    power_for_jump_height,
    JUMP_HEIGHT_CURVES,
)
from .low_level.cone_courses import slalom, training_cones, curved_slalom

from .mid_level.stair_climbing import climb_stairs

# ---------------------------------------------------------------------------
# Skill Registry — string name → callable
# ---------------------------------------------------------------------------
SKILL_REGISTRY: dict[str, callable] = {
    # Locomotion. `move` is the gait; the rest are presets of it.
    "move": move,
    "turn": turn,
    "move_forward": move_forward,
    "move_right": move_right,
    "move_left": move_left,
    "stop": stop,
    "go_fast": go_fast,
    "go_slow": go_slow,
    "reverse": reverse,
    "circle": circle,
    "curve": curve,
    "curved_movement": curve,
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
    "follow_path": follow_path,
    "track_path": follow_path,
    "traverse_rough_terrain": traverse_rough_terrain,
    "rough_terrain": traverse_rough_terrain,
    "active_suspension": traverse_rough_terrain,
    "stay_in_boundary": stay_in_boundary,
    "stay_within_boundary": stay_in_boundary,
    "boundary_containment": stay_in_boundary,
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


def _split_result(result):
    """Return ``(targets, meta)`` from whatever shape a skill handed back.

    Skills return three different things. Most return a bare target array.
    Three take `return_metadata` and return a `(targets, dict)` pair. Two
    always return a tuple: `wall_of_death` pairs targets with an info dict,
    and `chimney_friction_servo` follows them with two servo readings.
    """
    if isinstance(result, tuple):
        if len(result) == 2 and isinstance(result[1], dict):
            return result[0], dict(result[1])
        return result[0], {"extra": tuple(result[1:])}
    return result, {}


def _summarise(targets, max_extend, min_offset):
    """Telemetry any skill can report, derived from its targets alone."""
    array = np.asarray(targets, dtype=float)
    if array.ndim != 1 or array.size == 0:
        return {}
    summary = {
        "n_rods": int(array.size),
        "n_extended": int(np.count_nonzero(array > min_offset + 1e-6)),
        "max_extension_m": float(array.max()),
        "mean_extension_m": float(array.mean()),
    }
    if max_extend:
        summary["stroke_fraction"] = float(array.mean() / max_extend)
    return summary


@lru_cache(maxsize=None)
def _accepts_metadata(fn) -> bool:
    return "return_metadata" in inspect.signature(fn).parameters


def execute_skill(name: str, *args, return_metadata: bool = False, **kwargs):
    """Dispatch a skill by string name.

    Every skill answers `return_metadata`, whether or not the skill function
    takes the argument itself. Skills that support it natively supply their own
    telemetry; the rest get a summary derived from the targets they returned.
    That way a caller can ask any of the names in :data:`SKILL_NAMES` for
    telemetry without knowing which ones implement it.

    Parameters
    ----------
    name : str
        One of :data:`SKILL_NAMES`.
    return_metadata : bool
        If True, return ``(targets, meta)`` instead of just the targets.
    *args, **kwargs
        Forwarded to the skill function.

    Returns
    -------
    np.ndarray
        (n_bars,) rod extension targets in [0, max_extend].
    meta : dict, only when `return_metadata` is set
        Always carries ``skill`` (the function's own name) and
        ``requested_as`` (the registry name used, which may be an alias),
        plus a summary of the targets and anything the skill itself reported.
    """
    if name not in SKILL_REGISTRY:
        raise ValueError(
            f"Unknown skill {name!r}; available: {SKILL_NAMES}"
        )
    fn = SKILL_REGISTRY[name]
    if not return_metadata:
        return fn(*args, **kwargs)

    if _accepts_metadata(fn):
        kwargs = {**kwargs, "return_metadata": True}
    targets, meta = _split_result(fn(*args, **kwargs))

    max_extend = kwargs.get("max_extend")
    if max_extend is None and len(args) >= 3 and isinstance(args[2], (int, float)):
        max_extend = float(args[2])
    min_offset = float(kwargs.get("min_offset", 0.025))
    return targets, {
        **_summarise(targets, max_extend, min_offset),
        **meta,
        "skill": fn.__name__,
        "requested_as": name,
    }
