"""Plant a kickstand ahead of the roll to stop it.

Split out of `stop` in `skills/low_level/locomotion.py`, which does two jobs:
brake while moving, then hold a stance at rest. Those are separate intents for
a policy -- one sheds speed, the other resists being moved -- so they are
separate primitives here, and `stance.py` holds the other half.

The mechanism is a lever, not friction. Rods in the *leading* lower sector
extend into the ground ahead of the contact point, and the reaction torque
opposes the roll.
"""

from __future__ import annotations

import numpy as np

from .base import Param, RobotState, SkillSpec, finish
from .common import surface_frame

#: Speed at which the brake reaches full stroke, from `stop`'s calibration.
BRAKE_SPEED_REF = 0.6

SPEC = SkillSpec(
    name="brake",
    summary="extend a leading kickstand to shed speed",
    params=(
        Param("strength", 0.0, 3.5,
              "brake gain; `stop` uses 1.0 for a gentle halt and 3.2 for an "
              "emergency stop, and above about 3.5 the ball tips rather than stops"),
    ),
)


def act(state: RobotState, *, strength: float = 1.6) -> np.ndarray:
    """Rod targets for one step of braking.

    Needs ``state.lin_vel``. Without it there is no travel direction, so there
    is no 'ahead' to plant against and the call returns a bare stance.
    """
    targets = np.zeros(state.n_bars, dtype=np.float64)
    if state.lin_vel is None:
        return finish(targets, state)

    v = np.asarray(state.lin_vel, dtype=np.float64)[:2]
    speed = float(np.linalg.norm(v))
    if speed < 1e-6:
        return finish(targets, state)

    u_long, u_lat, u_into = surface_frame(state, v / speed)
    u_z = -u_into

    # Ahead of the contact point, below the core, and not scrubbing sideways.
    front = np.clip((u_long - 0.05) / 0.75, 0.0, 1.0)
    down = np.clip(1.0 - np.abs(u_z + 0.45) / 0.75, 0.0, 1.0)
    lat_tuck = np.clip(1.0 - 1.8 * (u_lat ** 2), 0.0, 1.0)

    # Stroke grows with speed so a slow ball is not slammed to a halt.
    scale = min(1.0, speed / BRAKE_SPEED_REF) * float(max(strength, 0.0))
    wave = np.clip(front * down * lat_tuck * scale, 0.0, 1.0)
    wave[u_z > 0.05] = 0.0
    return finish(wave * state.max_extend, state)
