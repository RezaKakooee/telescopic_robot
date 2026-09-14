"""A scripted expert for any route-based course, built from the scenario's features.

The playground oracle is a list of hand-placed windows. This one derives the
same kind of windows from the scenario itself, so it works on every
inspection course (and on new ones) without editing:

* every ``step`` slab the route crosses (beam, curb, floor pipe, bund wall)
  and every ``gap`` (trench, crack) gets a running-jump window before its
  near edge;
* every ascending stair riser gets a jump window;
* the stretch inside (and 1 m before) a ``pipe`` uses ``crawl_pipe``;
* the stretch across a stone field uses ``traverse_rough_terrain``;
* everything else is ``move`` along the waypoints, and ``stop`` at the goal.

Positions are measured as arc length along the route, so the windows are the
same distances that were measured on the playground: 0.55-0.85 m for the
running jump at cruise speed, 0.38-0.55 m for a stair riser.

    oracle = InspectionOracle(scenario)
    name, params, why = oracle.select(env.env.data.qpos[:3])
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .map_perception import MonotonicWaypointTracker, WaypointTracker
from .terrain import Gap, Pipe, Staircase, Step, StoneField, rows

# Trigger windows (metres before the obstacle edge along the route).
JUMP_WINDOW = (0.55, 0.85)
GAP_WINDOW = (0.25, 0.45)        # closer than a beam: the far edge must be under the flight
STAIR_WINDOW = (0.38, 0.55)
MIN_JUMP_HEIGHT = 0.08          # lower slabs are curbs the ball simply rolls over
PIPE_APPROACH = 2.0
GOAL_STOP_DIST = 0.45


@dataclass
class Station:
    kind: str           # "jump", "stair", "pipe", "rough"
    s_start: float      # arc length where the station begins (near edge / entry)
    s_end: float        # arc length where it ends
    label: str


class InspectionOracle:
    def __init__(self, scenario):
        self.sc = scenario
        self.path = np.asarray(scenario.path_pts, dtype=np.float64).reshape(-1, 2)
        tracker_cls = MonotonicWaypointTracker if getattr(scenario, "monotonic_path", False) else WaypointTracker
        self.tracker = tracker_cls(self.path, scenario.goal)
        self.total = float(self.tracker._arc_lengths[-1])
        self.s = self.tracker._arc_lengths[: len(self.path)]
        self.stations = self._build_stations()
        self._recent = []          # recent arc positions, to notice a stall in front of a station
        self._backing = 0          # remaining "reverse" steps of a retry (back up, then jump)

    # ------------------------------------------------------------------ #
    # Building the station list
    # ------------------------------------------------------------------ #
    def _inside_box(self, cx, cy, hx, hy, yaw_deg=0.0, pad=0.0):
        rel = self.path - np.array([cx, cy])
        if abs(yaw_deg) > 1e-9:
            c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
            rel = rel @ np.array([[c, -s], [s, c]])          # world -> box frame
        return (np.abs(rel[:, 0]) <= hx + pad) & (np.abs(rel[:, 1]) <= hy + pad)

    def _spans(self, mask):
        """Arc-length spans of every contiguous run of route points inside a footprint.

        A tour may cross the same obstacle several times: each crossing is
        its own station."""
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return []
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate([[idx[0]], idx[breaks + 1]])
        ends = np.concatenate([idx[breaks], [idx[-1]]])
        return [(float(self.s[a]), float(self.s[b])) for a, b in zip(starts, ends)]

    def _build_stations(self):
        st = []
        sc = self.sc
        for k, step in enumerate(rows(Step, getattr(sc, "steps", None))):
            if step.height < MIN_JUMP_HEIGHT:
                continue
            for span in self._spans(self._inside_box(step.x, step.y, step.half_x, step.half_y, pad=0.02)):
                st.append(Station("jump", *span, f"slab {k} ({step.height:.2f} m)"))
        for k, gap in enumerate(rows(Gap, getattr(sc, "gaps", None))):
            # A pit just beside the line is still a pit: fall back to a footprint
            # padded by the ball's reach when the route itself does not cross it.
            spans = (self._spans(self._inside_box(gap.x, gap.y, gap.half_x, gap.half_y, pad=0.02))
                     or self._spans(self._inside_box(gap.x, gap.y, gap.half_x, gap.half_y, pad=0.25)))
            for span in spans:
                st.append(Station("gap", *span, f"gap {k} ({2 * min(gap.half_x, gap.half_y):.2f} m)"))
        for k, flight in enumerate(rows(Staircase, getattr(sc, "staircases", None))):
            if flight.descending:
                continue
            yaw = np.radians(float(flight.yaw_deg))
            d = np.array([np.cos(yaw), np.sin(yaw)])
            n_ = np.array([-d[1], d[0]])
            for i in range(int(flight.n_steps)):
                riser = np.array([flight.x, flight.y]) + d * (i * float(flight.run))
                # every pass of the route over the riser line, in the climbing direction
                rel = self.path - riser
                along, across = rel @ d, rel @ n_
                on_line = (np.abs(along) <= 0.06) & (np.abs(across) <= float(flight.width) / 2)
                for span in self._spans(on_line):
                    j = int(np.searchsorted(self.s, span[0]))
                    j2 = min(j + 3, len(self.path) - 1)
                    if float((self.path[j2] - self.path[max(j - 3, 0)]) @ d) > 0:      # climbing, not descending
                        st.append(Station("stair", span[0], span[0] + float(flight.run), f"stair {k} riser {i}"))
        for k, pipe in enumerate(rows(Pipe, getattr(sc, "pipes", None))):
            yaw = float(pipe.yaw_deg or 0.0)
            c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
            cx, cy = pipe.x + c * pipe.length / 2, pipe.y + s_ * pipe.length / 2
            for span in self._spans(self._inside_box(cx, cy, pipe.length / 2, pipe.inner_radius, yaw)):
                st.append(Station("pipe", span[0] - PIPE_APPROACH, span[1] + 0.3, f"pipe {k} (r {pipe.inner_radius:.2f})"))
        for k, field in enumerate(rows(StoneField, getattr(sc, "stones", None))):
            for span in self._spans(self._inside_box(field.x, field.y, field.half_x, field.half_y, pad=0.3)):
                st.append(Station("rough", *span, f"stones {k}"))
        st.sort(key=lambda z: z.s_start)
        return st

    # ------------------------------------------------------------------ #
    # Runtime
    # ------------------------------------------------------------------ #
    def progress(self, pos) -> float:
        return self.total - self.tracker.get_path_progress(np.asarray(pos)[:2])[1]

    def select(self, pos) -> tuple[str, dict, str]:
        """Skill name (env option name), params, and a short reason."""
        pos = np.asarray(pos, dtype=np.float64)
        s_here = self.progress(pos)
        if (np.linalg.norm(pos[:2] - np.asarray(self.sc.goal)[:2]) < GOAL_STOP_DIST
                and self.total - s_here < 1.0):
            return "stop", {}, "at the goal"
        if self._backing > 0:
            # Retry, part 2: after backing up, jump again.
            self._backing -= 1
            if self._backing == 0:
                self._recent = []
                return "jump_forward_while_moving", {}, "retry: jump"
            return "reverse", {}, "retry: back up"
        # 6 macro steps without 5 cm of progress: the env itself gives up after 10.
        self._recent = (self._recent + [s_here])[-6:]
        stalled = len(self._recent) == 6 and (max(self._recent) - min(self._recent)) < 0.05
        for st in self.stations:
            # Stalled right in front of a jump station (a failed jump, a landing
            # short of the next riser): back up ~1 m for a run-up, then jump.
            if stalled and st.kind in ("jump", "gap", "stair") and -1.2 <= st.s_start - s_here <= 1.0:
                self._recent = []
                self._backing = 9
                return "reverse", {}, f"retry {st.label}: back up"
        for st in self.stations:
            ahead = st.s_start - s_here
            if st.kind == "jump" and JUMP_WINDOW[0] <= ahead <= JUMP_WINDOW[1]:
                return "jump_forward_while_moving", {}, f"jump {st.label}"
            if st.kind == "gap" and GAP_WINDOW[0] <= ahead <= GAP_WINDOW[1]:
                return "jump_forward_while_moving", {}, f"jump {st.label}"
            if st.kind == "stair" and STAIR_WINDOW[0] <= ahead <= STAIR_WINDOW[1]:
                return "jump_forward_while_moving", {}, f"jump {st.label}"
        for st in self.stations:
            if st.kind == "pipe" and st.s_start <= s_here <= st.s_end:
                return "crawl_pipe", {}, f"crawl {st.label}"
        for st in self.stations:
            if st.kind == "rough" and st.s_start <= s_here <= st.s_end:
                return "traverse_rough_terrain", {}, f"rough {st.label}"
        return "move", {}, "follow the route"
