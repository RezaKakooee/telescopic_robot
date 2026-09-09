"""Plan a standing hop between pads from the measured take-off envelope.

`jump_to` gives a take-off commanded by velocity; this module chooses the
command. For each calibrated cell it knows the take-off velocity actually
delivered on a bad day (`vz_eff_min`..`vz_eff_max`, `vx_eff` +- spread), so a
hop is planned as a bracket of parabolas rather than a single ideal one:

    every trajectory in the bracket must land inside the target pad's safe
    band, and clear the target's lip on the way in.

The planner also chooses WHERE TO STAND, since the launch point is the one
thing the robot controls exactly. It returns the stand position `x0`, the
velocity command, and the predicted landing bracket -- or None, honestly,
when no calibrated cell can make the hop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

G = 9.81
ROLL_RADIUS = 0.19       # core height above whatever it stands on
UNDERSIDE = 0.165        # core to underside of the tucked ball
LIP_MARGIN = 0.12        # underside must clear the target lip by this;
                         # thin margins get eaten by the orientation tails
LAND_NEAR = 0.06         # landing band: this far inside the near edge --
                         # small on purpose: the touchdown brake coasts the
                         # ball forward, deeper INTO the pad, never out of it
LAND_FAR = 0.22          # ...and this far short of the far edge (brake room)
STAND_EDGE = 0.13        # how close to a pad edge the ball can stand; the
                         # contact patch sits directly under the core
FACE_STANDOFF = 0.40     # never launch closer than this to a face
PROBE_TRUST = 1.0        # how far toward the cell's MEAN vx the planner may
                         # assume, given that the probe aborts anything below
                         # what it assumed. 0 = plan on the raw worst case and
                         # stand close; 1 = plan on the mean and stand well back

# Early-burn probe. Measured on 48 random orientations: the velocity a few
# steps into the burn predicts the lift-off state almost exactly --
#     vz_eff  = 1.002 * vz(step 8) + 0.107   (residual 0.06 m/s, r = 0.97)
#     vx_lift = 0.833 * vx(step 5) + 0.190   (residual 0.08 m/s, r = 0.95)
# -- while nothing measurable BEFORE the burn predicts it at all (r = 0.28).
# So the first steps of the real jump are the probe: a launch that is
# already below the plan's bracket is aborted while the ball is still only
# centimetres up, and retried from a slightly different footing.
PROBE_VZ_STEP, PROBE_VZ_A, PROBE_VZ_B = 8, 1.002, 0.107
PROBE_VX_STEP, PROBE_VX_A, PROBE_VX_B = 5, 0.833, 0.190
PROBE_GUARD = 0.05

_PATH = Path(__file__).resolve().parent / "hop_calibration.json"
_cache = None


def load_hop_calibration(path=None):
    global _cache
    if path is None and _cache is not None:
        return _cache
    data = json.loads(Path(path or _PATH).read_text())
    if path is None:
        _cache = data
    return data


def _t_land(vz, dz):
    """Flight time until the core has risen by dz (negative = dropped)."""
    disc = vz * vz - 2 * G * dz
    if disc < 0:
        return None
    return (vz + float(np.sqrt(disc))) / G


def _z_at(vz, z0, t):
    return z0 + vz * t - 0.5 * G * t * t


@dataclass
class HopPlan:
    x0: float                 # stand here before the hop
    vx_cmd: float
    vz_cmd: float
    land_lo: float            # predicted landing bracket (x)
    land_hi: float
    apex_rise_min: float
    impact_vz: float          # worst touchdown speed
    cell: dict
    standoff: float = 0.0     # distance from x0 to the target's near face
    vz_gate: float = 0.0      # abort if vz(step 8) is below this
    vx_gate: float = 0.0      # abort if vx(step 5) is below this
    vx_gate_hi: float = 9.0   # ...or above this (it would overshoot the pad)

    def describe(self):
        return (f"stand at x={self.x0:.2f} ({self.standoff:.2f} m from the face), "
                f"command vz={self.vz_cmd:.1f} "
                f"vx={self.vx_cmd:.1f} -> lands {self.land_lo:.2f}..{self.land_hi:.2f}"
                f" (apex rise >= {self.apex_rise_min:.2f} m,"
                f" impact <= {self.impact_vz:.1f} m/s)")


def plan_standing_hop(from_height, from_range, target, *, calibration=None,
                      probe_trust=PROBE_TRUST):
    """Choose stand point and velocity command for one pad-to-pad hop.

    from_height : height of the surface launched from (0.0 = floor).
    from_range : (lo, hi) of x where the ball may stand.
    target : dict with near, far, height (a `pillar_course_columns` entry).

    Works from the measured LIFT-OFF state, not from the command: each
    calibration cell says where the flight really starts (a few cm above the
    stance, essentially zero ground run) and with what velocity bracket, over
    random orientations. Every parabola in the bracket must land inside the
    target's safe band and clear its lip.
    """
    cal = calibration or load_hop_calibration()
    band_lo = target["near"] + LAND_NEAR
    band_hi = target["far"] - LAND_FAR
    stand_lo, stand_hi = from_range
    if from_height < target["height"]:
        stand_hi = min(stand_hi, target["near"] - FACE_STANDOFF)

    best = None
    for row in cal["rows"]:
        vz_lo, vz_hi = row["vz_eff_min"], row["vz_eff_max"]
        vx_raw_lo, vx_hi = row["vx_lift_min"], row["vx_lift_max"]
        # The probe gate (below) refuses any launch weaker than the bracket's
        # lower edge, so that edge can sit above the raw worst case. Raising
        # it buys reach, and reach is what lets the ball stand further back.
        vx_mid = 0.5 * (vx_raw_lo + vx_hi)
        vx_lo = vx_raw_lo + probe_trust * (vx_mid - vx_raw_lo)
        trv_lo, trv_hi = row["travel_min"], row["travel_max"]
        lift_up = row["z_lift_mean"] - row["z0"]        # stance -> lift-off

        z_launch = from_height + ROLL_RADIUS + lift_up
        dz = (target["height"] + ROLL_RADIUS) - z_launch
        rise_min = vz_lo * vz_lo / (2 * G)
        if rise_min < dz + 0.10:
            continue
        t_lo, t_hi = _t_land(vz_lo, dz), _t_land(vz_hi, dz)
        if t_lo is None or t_hi is None:
            continue
        dx_lo = trv_lo + vx_lo * t_lo
        dx_hi = trv_hi + vx_hi * t_hi
        if dx_hi - dx_lo > band_hi - band_lo:
            continue

        x0 = float(np.clip(band_lo - dx_lo, stand_lo, stand_hi))
        land_lo, land_hi = x0 + dx_lo, x0 + dx_hi
        if land_lo < band_lo - 1e-9 or land_hi > band_hi + 1e-9:
            continue

        # Lift-off must still be on the launch surface (burn ground run).
        if from_height > 0.0 and x0 + trv_hi > from_range[1] + STAND_EDGE - 0.06:
            continue

        # Lip clearance for the worst (lowest, slowest) parabola.
        t_cross = (target["near"] - (x0 + trv_lo)) / max(vx_lo, 1e-6)
        z_cross = _z_at(vz_lo, z_launch, t_cross)
        if z_cross - UNDERSIDE < target["height"] + LIP_MARGIN:
            continue

        impact = float(np.sqrt(max(vz_hi * vz_hi - 2 * G * dz, 0.0)))
        margin = min(land_lo - band_lo, band_hi - land_hi)
        # Prefer landing margin, then standing further from the face.
        score = margin + 0.5 * (target["near"] - x0) - 0.03 * vz_lo
        plan = HopPlan(
            x0=x0, standoff=target["near"] - x0,
            vx_cmd=row["vx_cmd"], vz_cmd=row["vz_cmd"],
            land_lo=land_lo, land_hi=land_hi,
            apex_rise_min=rise_min, impact_vz=impact, cell=row,
            # The plan assumed the cell's worst lift-off; the probe insists
            # the launch in progress will at least reach it.
            vz_gate=(vz_lo + PROBE_GUARD - PROBE_VZ_B) / PROBE_VZ_A,
            vx_gate=(vx_lo + PROBE_GUARD - PROBE_VX_B) / PROBE_VX_A,
            vx_gate_hi=(vx_hi - PROBE_GUARD - PROBE_VX_B) / PROBE_VX_A,
        )
        if best is None or score > best[0]:
            best = (score, plan)
    return None if best is None else best[1]
