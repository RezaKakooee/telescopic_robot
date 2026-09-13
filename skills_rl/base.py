"""The contract every RL primitive keeps, and the pieces to build an action.

`skills/` grew as a library for a person to call: rich signatures, keyword
arguments with physical names, defaults tuned per skill. That is right for
hand-written control and wrong for a policy, which needs every skill to look
the same and every argument to be a bounded number.

So a primitive here is three things:

* a :class:`SkillSpec` naming its parameters and the real range of each;
* an ``act(state, **params) -> (n_bars,) targets`` function;
* nothing else. No phase counter, no memory, no context the policy cannot see.

The policy emits a unit vector in ``[-1, 1]``. :meth:`SkillSpec.scale` turns
that into physical arguments, so the network never has to learn that a speed
is metres per second and an angle is radians.

Why bounded parameters rather than the raw keyword lists in `skills/`:
a policy exploring an unbounded ``back_gain`` will find values that fling the
robot or stall it, and the resulting gradient says nothing. The ranges here
are the ones the calibrations in `skills/low_level/locomotion.py` were
measured over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from radial_sphere.gait import MIN_OFFSET


@dataclass(frozen=True)
class RobotState:
    """Everything a primitive may read.

    One object rather than a long argument list, because every primitive takes
    the same one and an RL env builds it once per step. Fields past
    ``max_extend`` are optional: a primitive that needs one says so in its
    docstring and falls back sensibly when it is absent, exactly as the
    hand-written skills do.
    """

    quat: np.ndarray                        # (4,) body orientation, wxyz
    dirs_body: np.ndarray                   # (n_bars, 3) rod directions, body frame
    max_extend: float                       # stroke ceiling, metres
    lin_vel: np.ndarray | None = None       # (2,) or (3,) world velocity
    core_z: float | None = None             # world height of the core
    core_vz: float | None = None            # vertical speed of the core
    contact_forces: np.ndarray | None = None       # (n_bars,) newtons
    terrain_clearances: np.ndarray | None = None   # (n_bars,) signed metres
    surface_normal: np.ndarray | None = None       # (3,) core -> surface; None means floor
    min_offset: float = MIN_OFFSET          # retracted baseline
    #: Which speed calibration to believe, "ideal" or "hardware". Set it from
    #: `skills_rl.calibration.profile_for_config(cfg)` so it always matches how
    #: the env was built, rather than being chosen twice and disagreeing.
    profile: str = "ideal"

    @property
    def n_bars(self) -> int:
        return len(self.dirs_body)

    def floor_normal(self) -> np.ndarray:
        """The surface the robot is riding, defaulting to the ground."""
        if self.surface_normal is None:
            return np.array([0.0, 0.0, -1.0])
        n = np.asarray(self.surface_normal, dtype=np.float64)
        mag = float(np.linalg.norm(n))
        return n / mag if mag > 1e-9 else np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class Param:
    """One tunable number, and the range a policy is allowed to pick from."""

    name: str
    low: float
    high: float
    doc: str = ""

    def scale(self, unit: float) -> float:
        """Map ``[-1, 1]`` onto ``[low, high]``, clipping outside."""
        u = float(np.clip(unit, -1.0, 1.0))
        return self.low + 0.5 * (u + 1.0) * (self.high - self.low)

    def unit(self, value: float) -> float:
        """The inverse, for writing a known command as an action."""
        if self.high - self.low < 1e-12:
            return 0.0
        return float(np.clip(2.0 * (value - self.low) / (self.high - self.low) - 1.0,
                             -1.0, 1.0))


@dataclass(frozen=True)
class SkillSpec:
    """What one primitive is called and what it can be asked for."""

    name: str
    summary: str
    params: tuple[Param, ...] = field(default_factory=tuple)

    @property
    def n_params(self) -> int:
        return len(self.params)

    def scale(self, unit_params) -> dict:
        """Turn a unit vector into physical keyword arguments.

        Extra entries are ignored, so one fixed-width action vector can serve
        every primitive regardless of how many parameters it actually uses.
        Missing entries take the middle of the range, which is the least
        opinionated default available.
        """
        u = np.asarray(unit_params, dtype=np.float64).reshape(-1)
        out = {}
        for i, p in enumerate(self.params):
            out[p.name] = p.scale(u[i]) if i < len(u) else p.scale(0.0)
        return out

    def describe(self) -> str:
        lines = [f"{self.name}: {self.summary}"]
        for p in self.params:
            lines.append(f"    {p.name:<12} [{p.low:g}, {p.high:g}]  {p.doc}")
        return "\n".join(lines)


class Primitive(Protocol):
    """What each module in this package exposes."""

    SPEC: SkillSpec

    def act(self, state: RobotState, **params) -> np.ndarray:  # pragma: no cover
        ...


def finish(targets: np.ndarray, state: RobotState, *,
           floor: float | None = None) -> np.ndarray:
    """Clip to the physical stroke and return float32.

    Every primitive ends here so none of them can hand the simulator a target
    outside the actuator range. ``floor`` is the retracted baseline; pass 0.0
    for a primitive that genuinely wants rods shut, such as a brake that
    unloads everything but the kickstand.
    """
    lo = state.min_offset if floor is None else float(floor)
    return np.clip(np.asarray(targets, dtype=np.float64),
                   lo, state.max_extend).astype(np.float32)


def heading(azimuth_rad: float) -> np.ndarray:
    """A world-xy unit vector from an angle, so a policy emits one number."""
    a = float(azimuth_rad)
    return np.array([np.cos(a), np.sin(a)], dtype=np.float64)


def direction_3d(azimuth_rad: float, elevation_rad: float) -> np.ndarray:
    """A world unit vector from azimuth and elevation.

    Elevation is measured from the horizon: 0 is level, +pi/2 is straight up.
    Two angles rather than three components, because a policy emitting three
    numbers has to learn to keep them normalised and mostly does not.
    """
    a, e = float(azimuth_rad), float(elevation_rad)
    ce = np.cos(e)
    return np.array([ce * np.cos(a), ce * np.sin(a), np.sin(e)], dtype=np.float64)
