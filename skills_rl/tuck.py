"""Pull every rod in.

The airborne and neutral pose. `chimney_climb` uses it for the flight between
walls, `fall_down` for the drop, and every jump for the arc after take-off. A
rod left out in flight is mass on a lever arm that spins the ball, and in a
corridor it is the thing that catches a wall.

One parameter, because there is only one decision: how far in.
"""

from __future__ import annotations

from .base import Param, RobotState, SkillSpec, finish

#: What `chimney_climb` tucks to between walls.
CHIMNEY_TUCK = 0.010

SPEC = SkillSpec(
    name="tuck",
    summary="retract every rod to a uniform standoff",
    params=(
        Param("extension", 0.0, 0.06,
              f"metres left out; {CHIMNEY_TUCK} is the chimney flight pose, "
              "and a little stroke kept in hand absorbs a landing better than "
              "fully shut"),
    ),
)


def act(state: RobotState, *, extension: float = CHIMNEY_TUCK):
    """Rod targets for one step of tucking."""
    import numpy as np
    return finish(np.full(state.n_bars, float(extension)), state, floor=0.0)
