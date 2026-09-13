"""Follow the ground instead of ignoring it.

The terrain primitive. It is the only one that reads the rod sensors, and the
only one that is normally applied *on top of* another primitive rather than
alone: `traverse_rough_terrain` is `drive` plus this, and the active
suspension in `stay_in_boundary` is the same correction over whichever
sub-skill is running.

It is a primitive rather than a flag because the decision is real: how high to
try to ride, and how hard to insist. On a stone field those two numbers are
the difference between conforming and being thrown.

One warning is worth repeating from `SuspensionGains`. A ride height the build
cannot reach pins the height term at its clip forever, and because the
correction only extends the trailing support rods, a constant clip is a
constant forward push. The upper bound below is set to what the 15 cm core
actually holds, so a policy cannot ask for a height that becomes a throttle.
"""

from __future__ import annotations

import numpy as np

from skills.low_level.suspension import (SuspensionGains, SuspensionState,
                                         apply_suspension)

from .base import Param, RobotState, SkillSpec, finish
from .common import surface_frame

SPEC = SkillSpec(
    name="conform",
    summary="adjust the support rods to the ground under them",
    params=(
        Param("ride_height", 0.16, 0.30,
              "world core height to hold, metres. Asking for more than the "
              "build can reach is safe here, unlike in `apply_suspension`: the "
              "lift is spread over every supporting rod, so a saturated "
              "request stops lifting instead of turning into a push"),
        Param("terrain_gain", 0.0, 1.2,
              "how strongly measured ground height feeds into the rods"),
        Param("stiffness", 0.0, 1.2, "proportional gain on the height error"),
    ),
)


def act(state: RobotState, *, ride_height: float = 0.20,
        terrain_gain: float = 0.85, stiffness: float = 0.75,
        base: np.ndarray | None = None,
        susp_state: SuspensionState | None = None,
        control_dt: float = 0.01) -> np.ndarray:
    """Rod targets for one step of terrain conformance.

    ``base`` is the targets to correct, so this composes: pass the output of
    `drive` to get `traverse_rough_terrain`. Alone it conforms a flat stance.

    ``susp_state`` carries the filter and the slew limit between steps. Leaving
    it None is legal and stateless, but the rods then jump to the full
    correction every step, which pumps the ball along. An RL env should hold
    one per robot and reset it with the episode.
    """
    heading = (np.asarray(state.lin_vel, dtype=np.float64)[:2]
               if state.lin_vel is not None else np.array([1.0, 0.0]))
    if float(np.linalg.norm(heading)) < 1e-6:
        heading = np.array([1.0, 0.0])

    u_long, _, u_into = surface_frame(state, heading)
    if base is None:
        base = np.full(state.n_bars, state.min_offset, dtype=np.float64)

    # Terrain conformance rides on the rods already carrying the robot. A
    # leading rod driven into a pit plants against the far wall and stops the
    # robot dead, so the trailing weight below is deliberate.
    support = np.clip((-u_long - 0.05) / 0.35, 0.0, 1.0)
    support = support * support * (3.0 - 2.0 * support)

    # Ride height is handled here rather than by `apply_suspension`, and
    # symmetrically. That function multiplies every term by the same trailing
    # weight, which is right for terrain but wrong for lift: extending only the
    # trailing rods is how the rolling gait pushes, so a height error that
    # cannot be satisfied becomes a permanent throttle. Measured on the stone
    # field, this build holds 0.18-0.19 m whatever is asked, and the old
    # arrangement turned a 0.24 m request into 2.4 m of unwanted travel in 250
    # steps. Spreading the lift over every downward rod pushes the core up
    # without pushing it along.
    down = np.clip((u_into - 0.30) / 0.50, 0.0, 1.0)
    down = down * down * (3.0 - 2.0 * down)
    z_err = float(state.core_z - ride_height) if state.core_z is not None else 0.0
    lift = float(np.clip(-float(stiffness) * z_err
                         - 0.15 * float(state.core_vz or 0.0), -0.025, 0.025))
    base = np.asarray(base, dtype=np.float64) + down * lift

    gains = SuspensionGains(target_ride_height=float(ride_height),
                            kp=0.0, kd=0.0,
                            terrain_adaptation=float(terrain_gain))
    out, _meta = apply_suspension(
        base, -u_into, state.max_extend,
        core_z=state.core_z, core_vz=state.core_vz or 0.0,
        min_offset=state.min_offset,
        contact_forces=state.contact_forces,
        terrain_clearances=state.terrain_clearances,
        support_weight=support, gains=gains, state=susp_state, dt=control_dt,
    )
    return finish(out, state)
