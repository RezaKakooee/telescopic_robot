"""Dimensions of the playground parkour course, in one place.

Every obstacle sits at what the macro running jump can clear cleanly, measured
on the course with the demo oracle (scratch/measure_course_limits.py and the
oracle trace). Power 0.9, landing rollout 0.04.

=========  =========  ===========  ==================================
obstacle   value      clean limit  note
=========  =========  ===========  ==================================
hurdle     0.18       0.18         crossbar height (unchanged)
boxes      0.40 step  0.40 step    box 1 = 0.40, each next box +0.40
gap        0.60       0.65         valley width between boxes
stairs     0.40 step  0.40 step    treads 1.8 m: land ~1.1 m past a riser, then run up
wall       0.55       0.60         solid wall height
=========  =========  ===========  ==================================

Landing on a platform needs more than clearing a wall of the same height: the
ball must still be above the edge while it comes down and its rods extend for
the landing. That is why a 0.55 m wall works but a 0.55 m box does not.

mjcf_features.playground_xml builds the geometry from these numbers, the
scenario carries them for the height map, and the demo oracle and evaluation
derive obstacle edges from them. Call `configure(...)` to change values and
recompute everything derived (used by the measurement scripts).
"""
from __future__ import annotations

import sys

import numpy as np

# --- Station 2: hurdle on Leg 1 (x axis) -----------------------------------
HURDLE_X = 4.5
HURDLE_H = 0.18          # crossbar centre height
HURDLE_BAR_R = 0.035

# --- Station 3: three boxes on Leg 2 (y axis, x = 9.0) ----------------------
LEG2_X = 9.0
BOX_HEIGHTS = (0.40, 0.80, 1.20)   # box 1, 2, 3
BOX_LEN = 1.50           # along y
BOX_HALF_W = 1.20        # across the corridor
GAP_W = 0.60             # valley width between boxes
GAP_DEPTH = 0.50
BOX1_Y0 = 2.20           # front face of box 1

# --- Station 4: stairs, deck and ramp on Leg 3 (x axis, y = 9.0, heading -x) -
LEG3_Y = 9.0
STAIR_X0 = 7.2           # first riser
STAIR_N = 2
STAIR_RISE = 0.40
STAIR_RUN = 1.80         # tread depth: long enough to land and run up again
DECK_LEN = 0.80
RAMP_LEN = 1.60

# --- Station 6: jump wall on Leg 4 (y axis, x = 0.0) ------------------------
WALL_Y = 15.0
WALL_H = 0.55
WALL_HALF_T = 0.03


def _derive() -> None:
    """Recompute every value that follows from the settings above."""
    g = globals()
    g["BOX_H"] = BOX_HEIGHTS[0]
    g["BOX_Y0"] = tuple(BOX1_Y0 + i * (BOX_LEN + GAP_W) for i in range(3))      # front faces
    g["BOX_Y1"] = tuple(y0 + BOX_LEN for y0 in g["BOX_Y0"])                    # rear faces
    g["BOX_CENTER_Y"] = tuple((y0 + y1) / 2 for y0, y1 in zip(g["BOX_Y0"], g["BOX_Y1"]))
    g["VALLEY_CENTER_Y"] = tuple((g["BOX_Y1"][i] + g["BOX_Y0"][i + 1]) / 2 for i in range(2))
    g["STAIR_TOP"] = STAIR_N * STAIR_RISE
    g["STAIR_RISER_X"] = tuple(STAIR_X0 - i * STAIR_RUN for i in range(STAIR_N))   # x of each riser
    g["DECK_X1"] = STAIR_X0 - STAIR_N * STAIR_RUN      # deck starts where the stairs end
    g["DECK_X0"] = g["DECK_X1"] - DECK_LEN
    g["RAMP_X1"] = g["DECK_X0"]
    g["RAMP_X0"] = g["RAMP_X1"] - RAMP_LEN
    g["RAMP_PITCH_DEG"] = float(np.degrees(np.arctan2(g["STAIR_TOP"], RAMP_LEN)))


def configure(**values) -> None:
    """Override settings (e.g. BOX_HEIGHTS=(0.4, 0.8, 1.2)) and re-derive."""
    for k, v in values.items():
        if k not in globals():
            raise KeyError(k)
        globals()[k] = v
    _derive()


_derive()
