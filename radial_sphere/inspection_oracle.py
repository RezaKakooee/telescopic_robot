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

Positions are measured as arc length along the route. The expert arms a
jump once its station is within ``ARM_DIST``; the jump skill itself measures
the edge ahead and fires at the calibrated distance (``terrain_probe``).

    oracle = InspectionOracle(scenario)
    name, params, why = oracle.select(env.env.data.qpos[:3])
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .map_perception import DECK_MIN_HALF_SIZE, MonotonicWaypointTracker, WaypointTracker
from .terrain import Gap, Pipe, Staircase, Step, StoneField, rows

# The jump skill times itself: it rolls up to the next edge and fires at the
# calibrated distance (radial_sphere/terrain_probe.py, TRIGGER). The expert
# only has to say "jump" once the station is within reach of the probe,
# which makes any decision in the last ARM_DIST metres a right one. It used
# to fire the jump itself from 0.2 m wide windows (slabs 0.55-0.85 m, gaps
# 0.25-0.45 m, risers 0.38-0.55 m): at 1.1 m/s and 10 Hz that was 2 or 3
# decisions, and the policy learning from it kept missing the moment.
ARM_DIST = 1.6                  # say "jump" once the station's edge is this near (metres along the route)
ARM_MIN = -0.1                  # ...and not once the ball is past it
ARM_VIEW_DEG = 35.0             # ...and only when the station lies within this bearing of the ball's
                                # heading: the skill probes along the heading, and a station round a
                                # corner is not on it yet (arming it there only costs roll steps, but
                                # a late arm is what the data should show)
ARM_ALWAYS = 0.5                # ...except this near, where the bearing is noise: arm regardless
#: A stall is 6 macro steps with under 5 cm of arc progress AND under 15 cm
#: of travel: arc length alone read a ball rolling past a corner as stalled
#: (the tracker clamps the segment fraction) and fired the retry at 1.5 m/s.
STALL_XY = 0.15
#: A station already under the rods (nearer than the skill's MIN_TARGET) cannot
#: be jumped from here; when the ball then stops moving for 2 steps, back up
#: and jump at once rather than after 6 steps of pushing into it.
TOO_CLOSE = 0.35
MIN_JUMP_HEIGHT = 0.08          # lower slabs are curbs the ball simply rolls over
# Short decks (box tops under this length along the route): land, brake, back
# up, run, jump. A longer deck gives the running jump its run-up by itself.
SHORT_DECK = 2.6
#: A deck shorter than this cannot give the running jump its run-up even
#: after a back-up (1.2 m and 1.0 m decks: 0 of 8 either way): the routine
#: is brake, settle, jump, and the jump option hops from a standstill where
#: the build can (the long-stroke build). 1.6 m decks still need the back-up.
HOP_DECK = 1.2
DECK_REAR = 0.7                 # back up to this far past the deck's near edge
DECK_STOP_STEPS = 4             # macro steps of braking after the landing
DECK_SETTLE_STEPS = 3           # macro steps of standing still at the rear
PIPE_APPROACH = 2.0
GOAL_STOP_DIST = 0.45


@dataclass
class Station:
    kind: str           # "jump", "gap", "stair", "deck", "pipe", "rough"
    s_start: float      # arc length where the station begins (near edge / entry)
    s_end: float        # arc length where it ends
    label: str
    heading: np.ndarray = None   # the route's direction at s_start (set after building)
    xy: np.ndarray = None        # the route point at s_start


class InspectionOracle:
    def __init__(self, scenario, arm_dist: float = ARM_DIST):
        self.sc = scenario
        self.arm_dist = float(arm_dist)
        self.path = np.asarray(scenario.path_pts, dtype=np.float64).reshape(-1, 2)
        tracker_cls = MonotonicWaypointTracker if getattr(scenario, "monotonic_path", False) else WaypointTracker
        self.tracker = tracker_cls(self.path, scenario.goal)
        self.total = float(self.tracker._arc_lengths[-1])
        self.s = self.tracker._arc_lengths[: len(self.path)]
        self.stations = self._build_stations()
        self._recent = []          # recent arc positions, to notice a stall in front of a station
        self._recent_xy = []       # ...and recent positions: a stall is no progress AND no travel
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
        for z in st:
            z.heading = self._route_heading_at(z.s_start)
            z.xy = self.path[int(np.clip(np.searchsorted(self.s, z.s_start), 0, len(self.path) - 1))]
        return st

    def _route_heading_at(self, s_at: float) -> np.ndarray:
        """Unit direction of the route at arc length ``s_at``."""
        i = int(np.clip(np.searchsorted(self.s, s_at), 1, len(self.path) - 1))
        a, b = self.path[max(i - 3, 0)], self.path[min(i + 3, len(self.path) - 1)]
        d = b - a
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-9 else np.array([1.0, 0.0])

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
        self-timed jump. Measured on the doubling boxes: 2.2 m decks pass
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
                short = (deck.s_end - deck.s_start) < HOP_DECK
                self._deck_phase = "settle" if short else "back"
                self._deck_count = DECK_SETTLE_STEPS
            return "stop", {}, f"{deck.label}: brake"
        if self._deck_phase == "back":
            rear = min(DECK_REAR, 0.3 * (deck.s_end - deck.s_start))     # short decks: back up less
            if s_here - deck.s_start > rear:
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
        # 6 macro steps without 5 cm of progress and without 15 cm of travel:
        # the env itself gives up after 10.
        self._recent = (self._recent + [s_here])[-6:]
        self._recent_xy = (self._recent_xy + [pos[:2].copy()])[-6:]
        travel = float(np.max(np.linalg.norm(np.asarray(self._recent_xy) - self._recent_xy[0], axis=1))) if self._recent_xy else 0.0
        stalled = (len(self._recent) == 6 and (max(self._recent) - min(self._recent)) < 0.05
                   and travel < STALL_XY)
        if not stalled and len(self._recent_xy) >= 2:
            # Under the rods of a jump station and not moving: do not wait 6 steps.
            crept = float(np.linalg.norm(self._recent_xy[-1] - self._recent_xy[-2])) < 0.03
            under = any(st.kind in ("jump", "gap", "stair") and -0.1 <= st.s_start - s_here < TOO_CLOSE
                        for st in self.stations)
            stalled = crept and under and self._last.startswith("jump")
        if stalled and on_deck:
            # A blind 1 m reverse would back off the deck's rear edge: run the
            # deck routine again instead.
            self._recent, self._recent_xy = [], []
            self._deck_phase, self._deck_count = "stop", 1
            return "stop", {}, f"{self._deck.label}: stalled, again"
        for st in self.stations:
            # Stalled right in front of a jump station (a failed jump, a landing
            # short of the next riser): back up ~1 m for a run-up, then jump.
            # ...but only when the station is still ahead or under the ball: a
            # ball that cleared it and stalled past a corner must not back into it.
            if stalled and st.kind in ("jump", "gap", "stair") and s_here <= st.s_end + 0.3 and st.s_start - s_here <= 1.0:
                self._recent, self._recent_xy = [], []
                steps = [("flip", f"retry {st.label}: back up (flip)")] if not self._flipped else []
                steps += [("move", f"retry {st.label}: back up")] * 7
                steps += [("flip", "retry: face forward (flip)"), ("jump_forward_while_moving", f"retry: jump {st.label}")]
                self._script = steps[1:]
                return steps[0][0], {}, steps[0][1]
        # The nearest jump station ahead: say "jump" as soon as it is in reach.
        # The skill finds the edge itself and fires at the right distance.
        here = self._route_heading_at(s_here)
        for st in self.stations:
            ahead = st.s_start - s_here
            if st.kind in ("jump", "gap", "stair") and ARM_MIN <= ahead <= self.arm_dist:
                to_station = st.xy - pos[:2]
                dist = float(np.linalg.norm(to_station))
                if ahead > ARM_ALWAYS and dist > 1e-6 and float(to_station @ here) / dist < np.cos(np.radians(ARM_VIEW_DEG)):
                    continue                     # round a corner: the probe cannot see it yet
                return "jump_forward_while_moving", {}, f"jump {st.label}"
        for st in self.stations:
            if st.kind == "pipe" and st.s_start <= s_here <= st.s_end:
                return "crawl_pipe", {}, f"crawl {st.label}"
        for st in self.stations:
            if st.kind == "rough" and st.s_start <= s_here <= st.s_end:
                return "traverse_rough_terrain", {}, f"rough {st.label}"
        return "move", {}, "follow the route"
