"""Fire the loaded sector all at once to leave the ground.

The one airborne primitive. Every jump in `skills/low_level/jumping.py` --
`jump_up`, `jump_to`, `jump_forward_while_stopped`, `jump_forward_while_moving`
-- is this burst aimed differently, wrapped in a phase machine that crouches
first and tucks after. The crouch is `stance` with a low height, the tuck is
`tuck`, so only the burst itself is primitive.

It is also the launch of `chimney_climb` off the floor and of `wall_run`, and
the push off a wall mid-shaft is `brace` at full stroke, not this.

Aiming is by azimuth and elevation rather than a 3-vector, so a policy emits
two bounded angles instead of three numbers it must keep normalised.
"""

from __future__ import annotations

import numpy as np

from .base import Param, RobotState, SkillSpec, direction_3d, finish
from .common import rods_world

SPEC = SkillSpec(
    name="thrust",
    summary="fire one sector at full stroke to launch the ball",
    params=(
        Param("azimuth", -np.pi, np.pi, "horizontal aim in radians"),
        Param("elevation", -np.pi / 2, 0.0,
              "aim below the horizon: -pi/2 fires straight down for a vertical "
              "hop, shallower angles trade height for distance"),
        Param("power", 0.0, 1.0,
              "fraction of the stroke. `calibration.thrust_height` turns it "
              "into a peak rise: 0.49 m at full power on an ideal actuator, "
              "0.13 m with the motors modelled"),
        Param("spread", 0.25, 0.95,
              "how much of the sphere joins in. Narrow is a sharp, directional "
              "kick; wide recruits more rods for height and is more forgiving "
              "of the pose the ball happens to be in"),
    ),
)


def act(state: RobotState, *, azimuth: float = 0.0, elevation: float = -np.pi / 2,
        power: float = 1.0, spread: float = 0.55) -> np.ndarray:
    """Rod targets for one step of the launch burst.

    One step only. Holding the burst is the caller's job, and is what the
    `phase` argument does in the hand-written jump skills.

    How long to hold it is not a free choice once the motors are modelled.
    A rod capped at 0.28 m/s needs 0.48 s to cross its stroke, so a burst
    shorter than about 50 control steps never delivers the power it was asked
    for: measured, 10 steps at full power rises 0.030 m and 50 steps rises
    0.128 m. On an ideal actuator the rod arrives in one step and the burn
    length stops mattering after about 10. `calibration.full_stroke_steps`
    returns the figure for the active profile.
    """
    aim = direction_3d(azimuth, elevation)      # where the push goes
    u = rods_world(state) @ aim                 # rods aligned with it

    w = float(np.clip(spread, 1e-3, 1.0))
    wave = np.clip((u - (1.0 - w)) / w, 0.0, 1.0)
    wave = wave * wave * (3.0 - 2.0 * wave)     # smooth, so no rod snaps in
    return finish(wave * float(np.clip(power, 0.0, 1.0)) * state.max_extend, state)
