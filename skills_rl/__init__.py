"""Primitive skills shaped for a reinforcement learning policy to choose from.

`skills/` is a library for a person: rich signatures, physical keyword names,
per-skill defaults. `skills_rl/` is the same mechanics re-cut for a policy,
which needs every skill to look identical and every argument to be a bounded
number.

**What counts as primitive here.** A basis: the smallest set of rod-activation
patterns that mid- and high-level behaviour can be built from. The 27 functions
in `skills/low_level/` are not 27 patterns. `turn` is `move` with a rotated
reference. `move_forward`, `move_right`, `move_left`, `go_fast`, `go_slow` and
`reverse` are all presets of `move`. `surface_drive` is `move` with the ground
replaced by an arbitrary surface. `circle` and `curve` are `move` with the
heading turned a little each step. Stripping the aliases leaves seven:

======== =========================================================
drive    roll along the surface
brake    plant a leading kickstand to shed speed
stance   hold a base and make no drive
thrust   fire a sector at full stroke to leave the ground
tuck     pull every rod in
brace    hold a sector out against a surface
conform  adjust the support rods to the ground under them
======== =========================================================

**What is deliberately absent.** `follow_path`, `slalom`, `stay_in_boundary`
and `climb_stairs` are choosers: they pick among primitives from live state.
That is the job of the layer above, whether written by hand in
`skills/mid_level/` or learned. `wall_of_death` and `chimney_climb` are
sequences of these seven with a phase counter. None of them belong here.

**What each primitive is not.** It has no memory, no phase, and no context the
policy cannot see. One call is one control step. Committing to a multi-step
manoeuvre -- the crouch, burst, flight and landing of a jump -- is the caller's
business, because a primitive that remembered its own phase could not be
chosen freely at each step.

    from skills_rl import PRIMITIVES, RobotState, action_space, decode

    state = RobotState(quat=q, dirs_body=dirs, max_extend=0.16, lin_vel=v)
    targets = PRIMITIVES["drive"].act(state, azimuth=0.0, speed=1.2)
"""

from __future__ import annotations

import numpy as np

from . import brace, brake, conform, drive, stance, thrust, tuck
from .base import Param, Primitive, RobotState, SkillSpec, direction_3d, heading

#: Insertion order is the action index. Appending is safe; reordering silently
#: invalidates every trained policy, so do not.
PRIMITIVES: dict[str, Primitive] = {
    "drive": drive,
    "brake": brake,
    "stance": stance,
    "thrust": thrust,
    "tuck": tuck,
    "brace": brace,
    "conform": conform,
}

SKILL_NAMES: tuple[str, ...] = tuple(PRIMITIVES)

#: Width of the shared parameter block. Every primitive reads the first
#: `SPEC.n_params` entries and ignores the rest, so one fixed action vector
#: serves them all and the action space does not change when a primitive gains
#: a parameter.
MAX_PARAMS: int = max(m.SPEC.n_params for m in PRIMITIVES.values())


def use_profile(name: str) -> None:
    """Re-advertise `drive`'s speed range for one calibration profile.

    The parameter range a policy explores has to match what the robot can do.
    On the ideal actuator that is 0.33 to 2.8 m/s; with the motors modelled it
    is 0.32 to 0.58, and everything above is the same constant. Leaving the
    wide range in place would hand a policy two thirds of an action dimension
    that does nothing.

    This only moves the advertised bounds. Which curve is used per call comes
    from `RobotState.profile`, so two envs with different settings can run in
    the same process.
    """
    from . import calibration
    if name not in calibration.PROFILES:
        raise ValueError(f"unknown profile {name!r}; have {calibration.PROFILES}")
    lo, hi = calibration.speed_range(name)
    ps = list(drive.SPEC.params)
    ps[1] = Param("speed", float(lo), float(hi), ps[1].doc)
    drive.SPEC = SkillSpec(drive.SPEC.name, drive.SPEC.summary, tuple(ps))


def act(name: str, state: RobotState, **params) -> np.ndarray:
    """Run one primitive by name."""
    if name not in PRIMITIVES:
        raise KeyError(f"unknown primitive {name!r}; have {list(PRIMITIVES)}")
    return PRIMITIVES[name].act(state, **params)


def action_space():
    """A single `Box` covering the skill choice and its parameters.

    Laid out as ``[skill scores (n_skills), parameters (MAX_PARAMS)]``. The
    argmax of the scores picks the skill.

    A `Dict` or `Tuple` space would say this more plainly, but Stable-Baselines3
    PPO does not accept either for actions. Scores in one flat `Box` is the
    usual way round it, and it keeps the whole action continuous, which suits
    the Gaussian policy PPO already uses.
    """
    from gymnasium import spaces
    n = len(PRIMITIVES) + MAX_PARAMS
    return spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)


def decode(action) -> tuple[str, dict]:
    """Turn one action vector into a primitive name and physical arguments."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    n = len(PRIMITIVES)
    name = SKILL_NAMES[int(np.argmax(a[:n]))]
    return name, PRIMITIVES[name].SPEC.scale(a[n:n + MAX_PARAMS])


def describe() -> str:
    """Every primitive and its parameter ranges, for a run log."""
    return "\n".join(m.SPEC.describe() for m in PRIMITIVES.values())


__all__ = ["PRIMITIVES", "SKILL_NAMES", "MAX_PARAMS", "RobotState", "SkillSpec",
           "use_profile",
           "Param", "Primitive", "act", "action_space", "decode", "describe",
           "heading", "direction_3d"]
