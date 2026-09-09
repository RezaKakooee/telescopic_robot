"""Stair traversal dispatcher composed from the existing motion primitives.

The verified ascent uses geometry-planned ``jump_to`` hops, not fixed running
vault timers. Braking delegates to ``stop``, plateau travel to ``move``, and
each descent to the complete ``fall_down`` state machine. The course runner is
responsible for geometry, contact checks, phase transitions, and retries.
"""
from __future__ import annotations

import numpy as np
from .locomotion import move, stop
from .jumping import jump_forward_while_moving, jump_to
from .falling import fall_down


def climb_stairs(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray = np.array([1.0, 0.0], dtype=np.float32),
    *,
    phase: str = "approach",
    speed: float = 1.2,
    lin_vel: np.ndarray | None = None,
    drop_height: float = 0.25,
    stance_height: float = 0.08,
    vx_target: float = 0.6,
    vz_target: float = 3.0,
    wall_lock: bool = True,
) -> np.ndarray:
    """Execute stair climbing and descent by delegating to jumping, falling, and locomotion skills.

    Parameters
    ----------
    quat : (4,) body orientation quaternion [w, x, y, z].
    dirs_body : (60, 3) rod direction vectors in body frame.
    max_extend : maximum rod extension (m).
    d_hat : (2,) 2D commanded heading unit vector.
    phase :
        - Locomotion: ``'approach'``, ``'runup'``, ``'plateau'``, ``'finish'`` (via `move`)
        - Stance Poise: ``'stand'``, ``'poise'``, ``'brake'``, ``'standup'`` (via `stop`)
        - Planned tread hops: ``'hop_crouch'``, ``'hop_takeoff'``, ``'hop_airborne'``, ``'hop_landing'`` (via `jump_to`)
        - Legacy running vault: ``'sprint'``, ``'dip'``, ``'launch'``, ``'vault_airborne'``, ``'vault_landing'``
        - Step Drops: ``'edge'``, ``'freefall'``, ``'absorb'``, ``'settle'``, ``'step_down'`` (via `fall_down`)
    speed : commanded cruising / sprint speed (m/s).
    lin_vel : (3,) current body linear velocity for active braking and stabilization.
    drop_height : expected step drop height for compliant descent (m).
    stance_height : suspension standoff height (m).
    vx_target, vz_target : planned takeoff velocity passed to ``jump_to``.
    wall_lock : keep the leading rod sector closed during a tread hop.
    """
    d_norm = float(np.linalg.norm(d_hat))
    dh = d_hat / max(d_norm, 1e-6)

    # 1. Stance Poise & Active Braking (via stop skill)
    if phase in ("stand", "poise", "brake", "standup"):
        return stop(quat, dirs_body, max_extend, lin_vel=lin_vel, stance_height=stance_height)

    # 2. Geometry-planned tread hops (via the velocity-servo jump skill).
    elif phase.startswith("hop_"):
        sub_phase = phase.removeprefix("hop_")
        return jump_to(
            quat, dirs_body, max_extend, d_hat=dh, phase=sub_phase,
            vel=lin_vel, vx_target=vx_target, vz_target=vz_target,
            wall_lock=wall_lock, drop_height=drop_height,
        )

    # 3. Legacy running vault aliases, kept for callers that need a hurdle leap.
    elif phase in ("sprint", "dip", "launch", "vault_airborne", "vault_landing", "vault_jump"):
        sub_phase = {
            "vault_jump": "launch", "vault_airborne": "airborne",
            "vault_landing": "landing",
        }.get(phase, phase)
        return jump_forward_while_moving(quat, dirs_body, max_extend, d_hat=dh, phase=sub_phase)

    # 4. Controlled Step Drops & Landing Absorption (via fall_down skill)
    elif phase in ("edge", "freefall", "absorb", "settle", "step_down"):
        sub_phase = "edge" if phase == "step_down" else phase
        return fall_down(
            quat,
            dirs_body,
            max_extend,
            d_hat=dh,
            phase=sub_phase,
            drop_height=drop_height,
            stance_height=stance_height,
            absorb=0.08,
            gear=0.5,
            edge_speed=0.60,
        )

    # 5. Forward Locomotion & Plateau Cruising (via move skill)
    elif phase in ("approach", "runup", "plateau", "finish", "cruise"):
        return move(quat, dirs_body, max_extend, d_hat=dh, speed=speed)

    else:
        raise ValueError(
            f"Unknown stairs phase {phase!r}; available: 'stand', 'poise', 'hop_crouch', "
            f"'hop_takeoff', 'hop_airborne', 'hop_landing', 'edge', 'freefall', 'absorb', 'settle', "
            f"'approach', 'plateau', 'finish'"
        )
