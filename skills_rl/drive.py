"""Roll along the surface. The one locomotion primitive.

Every travelling skill in `skills/low_level/locomotion.py` is this pattern:
`move`, `turn`, `move_forward`, `move_right`, `move_left`, `go_fast`,
`go_slow` and `reverse` all resolve to one traveling wave with a different
heading or speed. `turn` is literally `move` with a rotated reference, and
`surface_drive` is `move` with the ground replaced by an arbitrary surface.
So there is one primitive here, not eight, and it takes the surface from the
state rather than assuming a floor.

`curve` and `circle` are not primitives either: they are this skill with the
heading turned a little each step, which a policy does by emitting a new
azimuth.
"""

from __future__ import annotations

import numpy as np

from radial_sphere.gait import curb_vault

from .base import Param, RobotState, SkillSpec, finish, heading
from .calibration import gain_for_speed, speed_range
from .common import push_wave, sector, surface_frame

# The spec advertises the ideal range, which is what an env without the
# actuator model can hold. `skills_rl.use_profile("hardware")` narrows it to
# the 0.32-0.58 m/s this build reaches with its own motors, so a policy does
# not spend two thirds of its action range on commands nothing can follow.
_LO, _HI = speed_range("ideal")

SPEC = SkillSpec(
    name="drive",
    summary="roll along the surface at a chosen heading and speed",
    params=(
        Param("azimuth", -np.pi, np.pi,
              "world heading in radians; the policy picks a direction, not a turn"),
        Param("speed", float(_LO), float(_HI),
              "metres per second, converted to push strength by the measured "
              "speed calibration rather than by a raw gain"),
        Param("vault", 1.0, 4.0,
              "boost on the rods bearing against the face of a step. Without "
              "it the rolling wave is stopped by a 2 cm kerb; at 2.6 it clears "
              "4 cm and at 4.0 it clears 6 cm. Taller than that needs `thrust`, "
              "because no amount of boost gets the wave over it"),
        Param("flank", 0.0, 1.0,
              "0 drives on the underbelly, 1 lifts the centre and carries the "
              "load on two side rails, which is how `straddle_gap` crosses a "
              "trench narrower than the ball"),
    ),
)


def act(state: RobotState, *, azimuth: float = 0.0, speed: float = 1.2,
        vault: float = 1.0, flank: float = 0.0) -> np.ndarray:
    """Rod targets for one step of rolling travel.

    The speed is honoured through whichever calibration `state.profile` names.
    On the hardware profile a request above about 0.58 m/s is clamped rather
    than turned into more push, because measurement says extra amplitude there
    changes nothing.
    """
    d = heading(azimuth)
    u_long, u_lat, u_into = surface_frame(state, d)

    gain = gain_for_speed(abs(float(speed)), state.profile, state.max_extend)
    wave = push_wave(u_long, u_lat, u_into, gain)
    if float(vault) > 1.0:
        # The rods already bearing on the face of a step push harder. This is
        # `traverse_rough_terrain`'s mechanism, and without it the plain
        # rolling wave stalls against a 2 cm kerb: measured, the ball stopped
        # short at every step height tested until the boost was switched on.
        wave = curb_vault(wave, u_long, -u_into, float(vault))
    if float(speed) < 0.0:
        # Reverse is the same wave measured against the opposite heading, which
        # is what `move` does with a pi turn. Recomputing keeps one code path.
        u_long, u_lat, u_into = surface_frame(state, -d)
        wave = push_wave(u_long, u_lat, u_into, gain)
        if float(vault) > 1.0:
            wave = curb_vault(wave, u_long, -u_into, float(vault))

    f = float(np.clip(flank, 0.0, 1.0))
    if f > 1e-3:
        # Straddle: tuck the rods directly under the core so nothing rests in
        # the trench, and let the two lateral rails carry the weight instead.
        centre = sector(u_lat, centre=0.0, width=0.35) * sector(-u_into, centre=1.0, width=0.5)
        rails = sector(np.abs(u_lat), centre=0.75, width=0.45)
        wave = wave * (1.0 - f * centre) + f * rails * np.clip(wave.max(), 0.2, 1.0)

    return finish(state.min_offset + wave * (state.max_extend - state.min_offset), state)
