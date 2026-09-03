"""Run a named skill against a live environment.

The skills themselves are pure functions: state in, rod targets out. This
module adds the two things a caller needs to actually *command* a skill:

1. **Argument routing.** Each skill wants different state. ``move_forward``
   needs a heading, ``push_against_wall`` needs a wall normal, ``stop`` wants
   the current velocity. :func:`skill_targets` reads what is needed off the
   env and passes it through.
2. **Phase sequencing.** The three jump skills are phase machines. The caller
   is supposed to drive them through crouch → takeoff → airborne → landing.
   :data:`PHASE_SCHEDULES` holds the verified timings so callers do not have
   to rediscover them.

Typical use::

    from skills.runner import run_skill
    stats = run_skill(env, "go_fast", steps=200, d_hat=np.array([1.0, 0.0]))
"""
from __future__ import annotations

import numpy as np

from . import execute_skill

# Skills that steer relative to a heading vector.
NEEDS_HEADING = {
    "move_forward", "move_right", "move_left", "go_fast", "go_slow", "reverse",
    "jump_forward_while_stopped", "jump_forward_while_moving", "fall_down",
    "jump_to", "straddle_gap", "straddle",
}


# Skills that read the robot's own position.
NEEDS_POSITION = {"circle", "slalom", "training_cones", "curved_slalom", "curved_training_cones"}

#: Skills that carry their own state and arena model, so the generic runner
#: cannot drive them. They need arena geometry, contacts, and course state;
#: use their dedicated runner under ``scripts/skills/``.
SELF_DRIVEN = {
    "wall_of_death", "motordrome", "wall_run", "horizontal_wall_run",
    "stairs", "climb_stairs", "step_vault",
}

# Skills that read the robot's own velocity.
NEEDS_VELOCITY = {"stop"}

# Skills that need to know where the wall is.
NEEDS_WALL_NORMAL = {"push_against_wall"}

# Phase-based skills.
PHASE_SKILLS = {"jump_up", "jump_forward_while_stopped",
                "jump_forward_while_moving", "fall_down"}


def _phase_jump_up(step, core_z):
    if step < 20:
        return "crouch"
    if step < 32:
        return "takeoff"
    return "airborne"


def _phase_jump_forward_stopped(step, core_z):
    if step < 20:
        return "crouch"
    if step < 32:
        return "takeoff"
    return "airborne" if core_z > 0.28 else "landing"


def _phase_fall_down(step, core_z):
    # Height-driven: creep to the lip, tuck once the core starts dropping,
    # spread the landing gear just above the floor, then settle.
    if core_z > 0.30:
        return "edge"
    if core_z > 0.24:
        return "freefall"
    if core_z > 0.20:
        return "absorb"
    return "settle"


def _phase_jump_forward_moving(step, core_z):
    if step < 55:
        return "sprint"
    if step < 62:
        return "dip"
    if step < 75:
        return "launch"
    return "airborne" if core_z > 0.28 else "landing"


# Verified phase timings, and the step budget each maneuver needs to finish.
PHASE_SCHEDULES = {
    "jump_up": (_phase_jump_up, 60),
    "jump_forward_while_stopped": (_phase_jump_forward_stopped, 80),
    "jump_forward_while_moving": (_phase_jump_forward_moving, 200),
    "fall_down": (_phase_fall_down, 120),
}


def skill_targets(env, name, step=0, *, d_hat=None, wall_normal=None, **kwargs):
    """Rod targets for one step of skill *name*, with env state routed in."""
    if name in SELF_DRIVEN:
        raise ValueError(
            f"skill {name!r} keeps its own state and needs an arena model; "
            "drive it with its dedicated runner under scripts/skills/")

    quat = env.data.qpos[3:7].copy()
    call = dict(kwargs)

    if name in NEEDS_POSITION and "ball_xy" not in call:
        call["ball_xy"] = env.data.qpos[0:2].copy()

    if name in NEEDS_HEADING:
        if d_hat is None:
            raise ValueError(f"skill {name!r} needs d_hat (a 2-vector heading)")
        call["d_hat"] = np.asarray(d_hat, dtype=np.float64)

    if name in NEEDS_VELOCITY and "lin_vel" not in call:
        call["lin_vel"] = env.data.qvel[0:2].copy()

    # jump_to servos its burn against the live velocity.
    if name == "jump_to" and "vel" not in call:
        call["vel"] = env.data.qvel[0:3].copy()


    if name in NEEDS_WALL_NORMAL:
        if wall_normal is None:
            raise ValueError(f"skill {name!r} needs wall_normal (a 2-vector)")
        call["wall_normal"] = np.asarray(wall_normal, dtype=np.float64)

    if name in PHASE_SKILLS and "phase" not in call:
        phase_fn, _ = PHASE_SCHEDULES[name]
        call["phase"] = phase_fn(step, float(env.data.qpos[2]))

    return execute_skill(name, quat, env.dirs_body, env.max_extend, **call)


def default_steps(name, fallback=120):
    """Step budget a skill needs to complete. Jumps have a fixed schedule."""
    if name in PHASE_SCHEDULES:
        return PHASE_SCHEDULES[name][1]
    return fallback


def run_skill(env, name, steps=None, *, d_hat=None, wall_normal=None,
              on_frame=None, **kwargs):
    """Drive *env* with skill *name* for *steps* steps.

    Parameters
    ----------
    on_frame : callable(env, step) or None
        Called after every env step. Use it to record video frames.

    Returns
    -------
    dict with start/end position, displacement, peak height and final speed.
    """
    if steps is None:
        steps = default_steps(name)

    start = env.data.qpos[0:3].copy()
    peak_z = float(start[2])
    max_speed = 0.0

    for step in range(steps):
        targets = skill_targets(env, name, step, d_hat=d_hat,
                                wall_normal=wall_normal, **kwargs)
        env.step(targets)
        peak_z = max(peak_z, float(env.data.qpos[2]))
        max_speed = max(max_speed, float(np.linalg.norm(env.data.qvel[0:2])))
        if on_frame is not None:
            on_frame(env, step)

    end = env.data.qpos[0:3].copy()
    return {
        "skill": name,
        "steps": steps,
        "start": start,
        "end": end,
        "displacement": end - start,
        "peak_z": peak_z,
        "net_lift": peak_z - float(start[2]),
        "max_speed": max_speed,
        "final_speed": float(np.linalg.norm(env.data.qvel[0:2])),
    }


def run_program(env, program, *, on_frame=None):
    """Run a sequence of skills back to back.

    ``program`` is a list of ``(skill_name, steps, kwargs)`` tuples; ``steps``
    may be None to use the skill's default budget.

    Returns the list of per-skill stats dicts.
    """
    out = []
    for entry in program:
        name, steps, kw = (list(entry) + [None, {}])[:3]
        out.append(run_skill(env, name, steps, on_frame=on_frame, **(kw or {})))
    return out
