"""Stand still and resist being moved.

The other half of `stop`. Bottom rods hold a low standoff and everything else
shuts, which gives a wide base and no drive in any direction. Used at rest, as
the landing pose after a fall, and as the pause between other primitives.

Held apart from `brake` on purpose: braking is a transient that needs a travel
direction, standing is a steady state that does not.
"""

from __future__ import annotations

import numpy as np

from radial_sphere.gait import STANCE_HEIGHT

from .base import Param, RobotState, SkillSpec, finish
from .common import surface_frame

SPEC = SkillSpec(
    name="stance",
    summary="hold a stable base and make no drive",
    params=(
        Param("height", 0.02, 0.10,
              f"standoff of the supporting rods in metres; `stop` rests at "
              f"{STANCE_HEIGHT}, taller is more stable on rough ground and "
              f"easier to tip on a slope"),
    ),
)


def act(state: RobotState, *, height: float = STANCE_HEIGHT) -> np.ndarray:
    """Rod targets for one step of standing."""
    # No travel direction matters here; any in-surface axis gives the same
    # `u_into`, which is the only component this primitive scores on.
    _, _, u_into = surface_frame(state, np.array([1.0, 0.0]))
    targets = np.zeros(state.n_bars, dtype=np.float64)
    targets[u_into > 0.30] = float(np.clip(height, 0.0, state.max_extend))
    return finish(targets, state)
