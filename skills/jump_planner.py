"""Work out the jump parameters an obstacle needs, instead of hand-tuning them.

The jump has three free numbers: how fast to run up, how long to hold the
crouch, and how far from the obstacle to start it. This module derives all
three from the obstacle's own geometry.

It works in two halves.

**The measured half.** `scripts/skills/calibrate_jump.py` records what each
(run-up gain, crouch length) pair actually produces: the height it takes off
at, the peak it reaches, how far past take-off that peak falls, and how far
the ball travels between starting the crouch and leaving the ground.

Identical commands do not give identical jumps — the peak lands anywhere
between 0.35 m and 0.61 m depending where the rolling gait is when the crouch
starts. The calibrator therefore samples a whole gait cycle and stores the
*worst* peak of the set. Planning against that number turns a lottery into a
guarantee.

**The computed half.** Flight is ballistic, so the arc is a parabola. Three
measured numbers pin it down completely: it starts at ``z_takeoff``, its
vertex is ``peak_guaranteed`` high, and that vertex sits ``dx_to_apex``
downrange. From there the height at any distance is arithmetic, and so is
everything the caller wants to know:

* how high the ball must get   -> obstacle height + ball underside + margin
* which crouch delivers that   -> the cheapest calibration row that does
* where to start the crouch     -> put the apex where it does the most good,
                                   then walk back by the measured travel

If no calibrated jump can clear the obstacle, :func:`plan_jump` returns
``None`` rather than guessing. Knowing it cannot make the jump is the point.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Underside of the ball while airborne: core radius plus the tucked rod.
BALL_UNDERSIDE = 0.165

# Never take off closer than this to the obstacle face, or the rods clip it.
MIN_STANDOFF = 0.12

# Braking distance of the `stop` skill, measured: 0.36 m of coast per m/s
# of entry speed. Landing ON a box is only useful if the ball can pull up
# before the far edge, so the plan has to buy room for this.
COAST_PER_SPEED = 0.36

_TABLE_PATH = Path(__file__).resolve().parent / "jump_calibration.json"
_table_cache: dict | None = None


def load_calibration(path=None) -> dict:
    """Read the calibration table produced by `calibrate_jump.py`."""
    global _table_cache
    if path is None and _table_cache is not None:
        return _table_cache
    p = Path(path) if path is not None else _TABLE_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"no jump calibration at {p}. Run:\n"
            f"    python scripts/skills/calibrate_jump.py")
    data = json.loads(p.read_text())
    if path is None:
        _table_cache = data
    return data


@dataclass
class JumpPlan:
    """Everything the caller needs to execute one jump."""

    approach_gain: float      # back_gain for the run-up
    crouch_steps: int         # how long to hold the dip
    launch_steps: int         # how long to hold the launch
    trigger_distance: float   # start the crouch this far from the near face
    predicted_peak: float     # guaranteed core height at the apex (m)
    clearance: float          # guaranteed gap under the ball over the obstacle
    landing_offset: float | None  # for "onto": where it lands, from box centre
    mode: str

    def describe(self) -> str:
        return (f"gain {self.approach_gain:.1f}, crouch {self.crouch_steps}, "
                f"trigger {self.trigger_distance:.2f} m, "
                f"peak {self.predicted_peak:.3f} m, "
                f"clearance {self.clearance:+.3f} m")


def _height_at(row: dict, dx: float) -> float:
    """Core height ``dx`` metres downrange of take-off, from the measured arc.

    The flight is a parabola with its vertex at the measured apex, so it is
    fully determined by where it starts and where it peaks.
    """
    z0 = row["z_takeoff"]
    peak = row["peak_guaranteed"]
    dxa = max(row["dx_to_apex"], 1e-3)
    return peak - (peak - z0) * ((dx - dxa) / dxa) ** 2


def _descent_distance(row: dict, height: float) -> float | None:
    """Downrange distance at which the arc falls back to ``height``.

    ``None`` if the arc never gets that high.
    """
    z0 = row["z_takeoff"]
    peak = row["peak_guaranteed"]
    dxa = max(row["dx_to_apex"], 1e-3)
    if peak <= height or peak <= z0:
        return None
    return dxa * (1.0 + float(np.sqrt((peak - height) / (peak - z0))))


def plan_jump(height: float, half_depth: float, *, mode: str = "over",
              margin: float = 0.03, calibration=None,
              underside: float = BALL_UNDERSIDE,
              from_height: float = 0.0, gap: float = 0.0) -> JumpPlan | None:
    """Plan a jump over, or onto, a box of the given size.

    Parameters
    ----------
    height : obstacle height above the floor, metres.
    half_depth : half its depth along the direction of travel, metres.
    mode : ``"over"`` to clear it, ``"onto"`` to land on top.
    margin : how much air to insist on under the ball, metres.
    from_height : height of the surface being launched from. Non-zero when
        hopping from one box to the next, in which case only the difference
        matters -- the arc does not care how high both of them are.
    gap : clear air between the launch surface's far edge and the target's
        near face. The ball cannot take off over the gap, so the stand-off
        can never be smaller than this.

    Returns the best plan, or ``None`` if nothing in the calibration clears it.
    """
    data = calibration if calibration is not None else load_calibration()
    rows = data["rows"]
    launch_steps = data["launch_steps"]

    # Distances measured from the box centre, along the travel direction.
    near, far = -half_depth, +half_depth
    # Only the height DIFFERENCE matters: hopping between two decks is the
    # same problem as jumping that difference off the floor.
    height = height - from_height
    # Take-off has to happen on solid ground, so never out over the gap.
    min_standoff = max(MIN_STANDOFF, gap + 0.02)
    best = None
    for row in rows:
        travel = row["travel_dip_to_takeoff"]

        # The ball must leave the ground BEFORE the box, never inside its
        # footprint, so the stand-off is searched rather than assumed.
        for standoff in np.arange(min_standoff, 1.60, 0.02):
            x_takeoff = near - float(standoff)

            if mode == "over":
                # Worst point of the crossing is one of the two edges: the arc
                # is concave, so it cannot dip in between.
                clear = min(_height_at(row, near - x_takeoff),
                            _height_at(row, far - x_takeoff)) - underside - height
                landing_offset = None

            else:   # "onto"
                dx_land = _descent_distance(row, height + underside)
                if dx_land is None:
                    break                      # this arc never gets high enough
                landing = x_takeoff + dx_land
                # It has to come down ON the deck, with room for its own width.
                if not (near + underside < landing < far - underside):
                    continue
                # It must also be able to stop before running off the far end.
                braking = abs(row["vx_takeoff"]) * COAST_PER_SPEED
                if (far - landing) < braking + underside:
                    continue
                clear = _height_at(row, near - x_takeoff) - underside - height
                landing_offset = float(landing)

            if clear < margin:
                continue

            trigger = float(standoff) + travel
            plan = JumpPlan(
                approach_gain=row["gain"],
                crouch_steps=row["crouch_steps"],
                launch_steps=launch_steps,
                trigger_distance=trigger,
                predicted_peak=row["peak_guaranteed"],
                clearance=float(clear),
                landing_offset=landing_offset,
                mode=mode,
            )
            # Prefer the biggest guaranteed clearance; break ties on a calmer
            # approach, which is easier on the landing.
            key = (round(plan.clearance, 3), -plan.approach_gain)
            if best is None or key > best[0]:
                best = (key, plan)

    return None if best is None else best[1]


def max_clearable(mode: str = "over", margin: float = 0.03,
                  calibration=None, underside: float = BALL_UNDERSIDE) -> float:
    """Tallest obstacle of negligible depth this robot can guarantee.

    Useful for saying "that box is too tall" before driving at it.
    """
    data = calibration if calibration is not None else load_calibration()
    # Landing on a box needs a deck the ball fits on; clearing one does not.
    half_depth = 0.10 if mode == "over" else 0.45
    best = 0.0
    for h in np.arange(0.05, 0.80, 0.01):
        if plan_jump(float(h), half_depth, mode=mode, margin=margin,
                     calibration=data, underside=underside) is not None:
            best = float(h)
    return best
