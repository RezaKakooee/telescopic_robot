"""Which speed curve `drive` believes, and why there are two.

`skills/low_level/locomotion.py` carries a calibration measured on an ideal
actuator: amplitude 4.0 cruises at 2.8 m/s. Switch the motors on and the same
amplitude cruises at 0.54 m/s. The robot is not underperforming; a rod that
tops out at 0.28 m/s cannot complete its stroke inside the time the gait gives
it, and above amplitude 2.0 extra push buys nothing at all.

So a policy trained against the ideal curve spends most of its speed range on
commands the robot cannot follow. Asking for 2.4 m/s and getting 0.55 is not a
control signal, it is a constant.

Two profiles, chosen by config rather than by edit:

``ideal``
    The library curve. What every existing demo and skill was tuned against,
    and the default, so nothing changes unless it is asked for.

``hardware``
    Measured with ``sim2real.actuator_in_env`` on. Range 0.32 to 0.58 m/s,
    which is what this build can hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from skills.low_level.locomotion import (gain_for_speed as _ideal_gain,
                                         speed_range as _ideal_range)

_PATH = Path(__file__).with_name("speed_calibration.json")
_DATA = json.load(open(_PATH))
_HW = np.asarray(_DATA["profiles"]["hardware"]["rows"], dtype=float)
_HW_GAINS, _HW_SPEEDS = _HW[:, 0], _HW[:, 1]

PROFILES = ("ideal", "hardware")


def speed_range(profile: str = "ideal", max_extend=None) -> tuple[float, float]:
    """Slowest and fastest cruise this profile can hold, in m/s."""
    if profile == "hardware":
        return float(_HW_SPEEDS[0]), float(_HW_SPEEDS[-1])
    return _ideal_range(max_extend)


def gain_for_speed(speed: float, profile: str = "ideal", max_extend=None) -> float:
    """Push amplitude that cruises at `speed`, under this profile.

    Clamped at both ends. Past the top of the hardware curve the amplitude is
    held at its saturation value rather than extrapolated, because more push
    measurably does nothing there.
    """
    if profile != "hardware":
        return _ideal_gain(speed, max_extend)
    v = float(np.clip(abs(speed), _HW_SPEEDS[0], _HW_SPEEDS[-1]))
    return float(np.interp(v, _HW_SPEEDS, _HW_GAINS))


def speed_for_gain(gain: float, profile: str = "ideal", max_extend=None) -> float:
    """The inverse: what this amplitude actually cruises at."""
    if profile != "hardware":
        from skills.low_level.locomotion import speed_for_gain as _sfg
        return _sfg(gain, max_extend)
    return float(np.interp(float(gain), _HW_GAINS, _HW_SPEEDS))


def profile_for_config(cfg) -> str:
    """Pick the profile that matches how the env is set up.

    If the env applies the actuator model, the hardware curve is the one that
    describes it. Reading the same switch means the two cannot disagree.
    """
    s2r = getattr(cfg, "sim2real", None)
    if s2r is None:
        return "ideal"
    on = bool(getattr(s2r, "enabled", False)) and bool(getattr(s2r, "actuator_in_env", False))
    return "hardware" if on else "ideal"


__all__ = ["PROFILES", "speed_range", "gain_for_speed", "speed_for_gain",
           "profile_for_config"]


# ---------------------------------------------------------------------------
# Thrust
# ---------------------------------------------------------------------------
_TH = _DATA["thrust"]["profiles"]


def full_stroke_steps(profile: str = "ideal") -> int:
    """Control steps a burst must be held for to deliver its commanded power.

    On an ideal actuator a rod reaches its target in one step, so this is small
    and a caller can ignore it. With the motors modelled the rod crosses its
    0.135 m stroke at 0.28 m/s, which takes 0.48 s: about 50 steps. Measured,
    a 10-step burst at full power rises 0.030 m and a 50-step burst rises
    0.128 m, so the difference is most of the jump.
    """
    p = _TH.get(profile, _TH["ideal"])
    return int(p.get("full_stroke_steps", p.get("burn_steps", 10)))


def thrust_height(power: float, profile: str = "ideal") -> float:
    """Peak rise in metres from a full-length burst at this power.

    Interpolated from measurement, clamped at both ends. The hardware profile
    tops out near 0.13 m against 0.49 m ideal; that is the build, not a tuning
    failure, and a planner should size its jumps against this number rather
    than the ideal one.
    """
    rows = np.asarray(_TH.get(profile, _TH["ideal"])["rows"], dtype=float)
    return float(np.interp(float(np.clip(power, 0.0, 1.0)), rows[:, 0], rows[:, 1]))


def max_thrust_height(profile: str = "ideal") -> float:
    """The tallest hop this profile can produce."""
    return thrust_height(1.0, profile)


__all__ += ["full_stroke_steps", "thrust_height", "max_thrust_height"]
