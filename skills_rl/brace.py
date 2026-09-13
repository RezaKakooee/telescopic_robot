"""Press a sector against a surface, in place.

The static-contact primitive, as against `thrust`, which is the impulsive one.
Same rods, different intent: brace holds a commanded extension so the load
stays on, thrust fires to full stroke and lets go.

It covers `push_against_wall`, the clamp and the controlled descent of
`chimney_climb` (the friction servo is this with the caller moving `extension`
on measured vertical speed), the ride phase of `wall_run`, and the press term
of `surface_drive`.

Bracing on two opposite sides at once is what makes a chimney hold: set
``both_sides`` and the mirrored sector presses too, so the ball wedges instead
of being shoved away.
"""

from __future__ import annotations

import numpy as np

from .base import Param, RobotState, SkillSpec, direction_3d, finish
from .common import rods_world

SPEC = SkillSpec(
    name="brace",
    summary="hold a sector out against a surface",
    params=(
        Param("azimuth", -np.pi, np.pi, "horizontal aim of the press"),
        Param("elevation", -np.pi / 2, np.pi / 2,
              "vertical aim: 0 presses level into a wall, negative into the "
              "ground, positive into a ceiling"),
        Param("extension", 0.0, 1.0,
              "fraction of the stroke to hold. This is the friction knob: less "
              "extension is less normal force, so a clamped ball slides faster"),
        Param("spread", 0.2, 0.9, "how wide the pressing sector is"),
        Param("both_sides", 0.0, 1.0,
              "above 0.5 the mirrored sector presses as well, which wedges the "
              "ball between two facing walls instead of pushing it off one"),
    ),
)


def act(state: RobotState, *, azimuth: float = 0.0, elevation: float = 0.0,
        extension: float = 1.0, spread: float = 0.45,
        both_sides: float = 0.0) -> np.ndarray:
    """Rod targets for one step of bracing."""
    aim = direction_3d(azimuth, elevation)
    u = rods_world(state) @ aim
    if float(both_sides) > 0.5:
        u = np.abs(u)                    # both the sector and its mirror

    w = float(np.clip(spread, 1e-3, 1.0))
    wave = np.clip((u - (1.0 - w)) / w, 0.0, 1.0)
    wave = wave * wave * (3.0 - 2.0 * wave)
    return finish(wave * float(np.clip(extension, 0.0, 1.0)) * state.max_extend, state)
