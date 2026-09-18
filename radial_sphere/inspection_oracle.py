"""A scripted expert for any route-based course, built from the scenario's features.

The playground oracle is a list of hand-placed windows. This one derives the
same kind of windows from the scenario itself, so it works on every
inspection course (and on new ones) without editing:

* every ``step`` slab the route crosses (beam, curb, floor pipe, bund wall)
  and every ``gap`` (trench, crack) gets a running-jump window before its
  near edge;
* every ascending stair riser gets a jump window;
* a short deck (a box top the ball lands on, under 2.6 m long) gets the
  platform routine: brake after the landing, back up to the deck's rear,
  settle, then run and jump from a nearer window;
* there is no reverse decision: to back up the expert says ``flip`` (the
  env turns the travel direction around and brakes for that step), then
  ``move``, then ``flip`` again to face forward;
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

from .map_perception import DECK_MIN_HALF_SIZE, MonotonicWaypointTracker, WaypointTracker
from .terrain import Gap, Pipe, Staircase, Step, StoneField, rows

# Trigger windows (metres before the obstacle edge along the route).
JUMP_WINDOW = (0.55, 0.85)
GAP_WINDOW = (0.25, 0.45)        # closer than a beam: the far edge must be under the flight
STAIR_WINDOW = (0.38, 0.55)
MIN_JUMP_HEIGHT = 0.08          # lower slabs are curbs the ball simply rolls over
# Short decks (box tops under this length along the route): land, brake, back
# up, run, jump. A longer deck gives the running jump its run-up by itself.
SHORT_DECK = 2.6
DECK_REAR = 0.7                 # back up to this far past the deck's near edge
DECK_STOP_STEPS = 4             # macro steps of braking after the landing
DECK_SETTLE_STEPS = 3           # macro steps of standing still at the rear
DECK_GAP_WINDOW = (0.30, 0.50)  # the run-up is short, so launch nearer the edge
PIPE_APPROACH = 2.0
GOAL_STOP_DIST = 0.45


@dataclass
class Station:
    kind: str           # "jump", "gap", "stair", "deck", "pipe", "rough"
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
        self._script = []          # queued decisions of a retry (flip, back up, flip, jump)
        self._flipped = False      # what the env's travel direction is, as far as this expert has commanded
        self._deck = None          # the short deck the ball is on, or None
        self._deck_phase = None    # "stop", "back", "settle", "run" on that deck
        self._deck_count = 0       # macro steps left in the stop / settle phases
        self._last = "move"        # the skill returned last time

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
        for k, step in enumerate(rows(Step, getattr(sc, "steps", None))):
            if min(step.half_x, step.half_y) < DECK_MIN_HALF_SIZE:
                continue                          # a beam or a curb, not something to land on
            for span in self._spans(self._inside_box(step.x, step.y, step.half_x, step.half_y)):
                if span[1] - span[0] < SHORT_DECK:
                    st.append(Station("deck", *span, f"deck {k} ({step.height:.2f} m)"))
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
        # A box right behind a pit is jumped with the pit: its own window would
        # fire first, half a metre too early, and put the ball into the face.
        gap_ends = [z.s_end for z in st if z.kind == "gap"]
        st = [z for z in st if not (z.kind == "jump" and any(0.0 <= z.s_start - e <= 0.4 for e in gap_ends))]
        st.sort(key=lambda z: z.s_start)
        return st

    # ------------------------------------------------------------------ #
    # Runtime
    # ------------------------------------------------------------------ #
    def progress(self, pos) -> float:
        return self.total - self.tracker.get_path_progress(np.asarray(pos)[:2])[1]

    def select(self, pos) -> tuple[str, dict, str]:
        """Skill name (env option name), params, and a short reason."""
        choice = self._select(np.asarray(pos, dtype=np.float64))
        if choice[0] == "flip":
            self._flipped = not self._flipped
        self._last = choice[0]
        return choice

    def _go(self, backwards: bool, why: str) -> tuple[str, dict, str]:
        """Drive along the route, or back along it. The ball always rolls
        "forward": to go the other way it flips first, then rolls; there is
        no reverse decision. One flip per change of direction."""
        if backwards != self._flipped:
            return "flip", {}, why + " (flip)"
        return "move", {}, why

    def _deck_routine(self, s_here):
        """The platform routine on a short deck, or None to carry on as usual.

        The running jump needs a run-up and its landing rolls on, so on a
        deck shorter than 2.6 m the ball would launch from a standstill or
        coast off the far edge. Instead: brake after the landing, back up to
        the deck's rear, stand still, then run and jump from the nearer
        `DECK_GAP_WINDOW`. Measured on the doubling boxes: 2.2 m decks pass
        with this, 1.5 m decks only sometimes, 2.0 m decks never without it.
        """
        deck = next((st for st in self.stations
                     if st.kind == "deck" and st.s_start - 0.05 <= s_here <= st.s_end + 0.05), None)
        if deck is None:
            self._deck, self._deck_phase = None, None
            return None
        if deck is not self._deck:
            self._deck = deck
            follows = any(st.kind in ("gap", "jump") and 0.0 <= st.s_start - deck.s_end <= 0.6
                          for st in self.stations)
            landed = self._last.startswith("jump")
            self._deck_phase = "stop" if (landed and follows) else "run"
            self._deck_count = DECK_STOP_STEPS
        if self._deck_phase == "stop":
            self._deck_count -= 1
            if self._deck_count <= 0:
                self._deck_phase = "back"
            return "stop", {}, f"{deck.label}: brake"
        if self._deck_phase == "back":
            if s_here - deck.s_start > DECK_REAR:
                return self._go(True, f"{deck.label}: back up")
            self._deck_phase, self._deck_count = "settle", DECK_SETTLE_STEPS
        if self._deck_phase == "settle":
            if self._flipped:
                return self._go(False, f"{deck.label}: face forward")
            self._deck_count -= 1
            if self._deck_count <= 0:
                self._deck_phase = "run"
                self._recent = []
            return "stop", {}, f"{deck.label}: settle"
        return None

    def _select(self, pos) -> tuple[str, dict, str]:
        s_here = self.progress(pos)
        if (np.linalg.norm(pos[:2] - np.asarray(self.sc.goal)[:2]) < GOAL_STOP_DIST
                and self.total - s_here < 1.0):
            return "stop", {}, "at the goal"
        if self._script:
            # Retry, part 2: flip, back up, flip, and jump again.
            name, why = self._script.pop(0)
            if not self._script:
                self._recent = []
            return name, {}, why
        routine = self._deck_routine(s_here)
        if routine is not None:
            return routine
        on_deck = self._deck is not None
        # 6 macro steps without 5 cm of progress: the env itself gives up after 10.
        self._recent = (self._recent + [s_here])[-6:]
        stalled = len(self._recent) == 6 and (max(self._recent) - min(self._recent)) < 0.05
        if stalled and on_deck:
            # A blind 1 m reverse would back off the deck's rear edge: run the
            # deck routine again instead.
            self._recent = []
            self._deck_phase, self._deck_count = "stop", 1
            return "stop", {}, f"{self._deck.label}: stalled, again"
        for st in self.stations:
            # Stalled right in front of a jump station (a failed jump, a landing
            # short of the next riser): back up ~1 m for a run-up, then jump.
            if stalled and st.kind in ("jump", "gap", "stair") and -1.2 <= st.s_start - s_here <= 1.0:
                self._recent = []
                steps = [("flip", f"retry {st.label}: back up (flip)")] if not self._flipped else []
                steps += [("move", f"retry {st.label}: back up")] * 7
                steps += [("flip", "retry: face forward (flip)"), ("jump_forward_while_moving", "retry: jump")]
                self._script = steps[1:]
                return steps[0][0], {}, steps[0][1]
        gap_window = DECK_GAP_WINDOW if on_deck else GAP_WINDOW
        for st in self.stations:
            ahead = st.s_start - s_here
            if st.kind == "jump" and JUMP_WINDOW[0] <= ahead <= JUMP_WINDOW[1]:
                return "jump_forward_while_moving", {}, f"jump {st.label}"
            if st.kind == "gap" and gap_window[0] <= ahead <= gap_window[1]:
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
