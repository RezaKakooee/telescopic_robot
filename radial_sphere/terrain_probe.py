"""What the ground does ahead of the ball: the edges a skill can aim at.

Rays go down onto the model along the travel heading, 5 cm apart, out to
3.5 m, the same way the terrain rays under the ball work. The first place
the ground steps up or down by more than a curb is an :class:`Edge`: how
far away it is, whether it rises or drops, by how much, and how long the
new level lasts before it changes again. A skill reads this to time itself.

* a beam is a short rise, a stair tread a rise about one tread long, a
  platform a long rise;
* a trench is a drop that comes back within a ball's length, a cliff or a
  descending stair a drop that does not;
* a wall is a rise taller than any jump.

``plan_jump`` turns the edges into the one number the running jump needs:
how far before the edge to fire. Those distances are the windows the expert
used to fire the jump by hand (``inspection_oracle``: slabs 0.55-0.85 m,
gaps 0.25-0.45 m, risers 0.38-0.55 m), moved into the skill so that the
policy's decision can come a metre early and still be right.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

#: Ground steps smaller than this are curbs the ball simply rolls over
#: (``inspection_oracle.MIN_JUMP_HEIGHT`` is the same number).
MIN_EDGE = 0.08
#: A rise taller than this is a wall, not a jump target.
MAX_JUMP_RISE = 0.55
#: A drop that comes back up within this distance is a trench to jump; a
#: longer one is a lower level the ball rolls down to. The far side may be
#: lower than the near side by up to MAX_DROP (a pit before a lower deck):
#: the jump still crosses it and lands lower.
MAX_GAP = 0.80
MAX_DROP = 0.55
#: How far ahead the probe looks, and how finely. 4 m so that a 1.8 m stair
#: tread seen from 1.6 m still shows its far end (else it reads as a platform).
PROBE_RANGE = 4.0
PROBE_STEP = 0.05
#: A rise or drop must happen within one sample to be an edge; a slower
#: change is a ramp or a curved surface, which the ball rolls.
SHARP_FRACTION = 0.6
#: Nearer than this an edge is too late to aim at: the launch travels about
#: 0.3 m, so the ball would leave the ground on top of it. Such an edge is
#: rolled over, not jumped (measured: jumps fired at 0.2 m into a cable
#: cover hit it every time).
MIN_TARGET = 0.35

#: Fire the running jump this far before the edge. One number per edge
#: shape, measured on the playground and the inspection courses.
TRIGGER = {
    "beam": 0.75,        # a rise shorter than a tread: hurdle, floor pipe, crate
    "riser": 0.50,       # a rise about one tread long: stairs
    "platform": 0.75,    # a long rise: a deck to land on and roll across
    "gap": 0.40,         # a drop that comes back: trench, crack, pit before a deck (0.42 fell into a 0.39 m gap once in 24)
}
#: A rise whose top is shorter than this is a beam (a hurdle, a floor pipe,
#: a crate, a pipe saddle: something to fly over), longer than the second
#: number a platform; in between it is a stair tread (the courses' treads
#: are 1.5 to 1.8 m).
BEAM_MAX_TOP = 1.2
TREAD_MAX_TOP = 2.5


@dataclass
class Edge:
    dist: float          # from the ball's centre to the edge, along the heading (m)
    kind: str            # "rise" or "drop"
    change: float        # height change at the edge (m, signed)
    length: float        # how far the new level runs before it changes again; inf if not seen
    returns: bool        # a drop that comes back up within MAX_GAP (to a level at most MAX_DROP lower): a gap
    level: float         # ground height before the edge (m)

    @property
    def shape(self) -> str | None:
        """What a jump would be aiming at, or None when it is not a target."""
        if self.kind == "drop":
            return "gap" if self.returns else None
        if self.change > MAX_JUMP_RISE:
            return None                                  # a wall
        if self.length < BEAM_MAX_TOP:
            return "beam"
        if self.length < TREAD_MAX_TOP:
            return "riser"
        return "platform"


def profile_ahead(model, data, ray_groups, body_exclude, pos, heading, *,
                  max_dist: float = PROBE_RANGE, step: float = PROBE_STEP):
    """Ground height along the heading: (distances, heights), the first sample under the ball."""
    heading = np.asarray(heading, dtype=np.float64)[:2]
    heading = heading / max(float(np.linalg.norm(heading)), 1e-9)
    dists = np.arange(0.0, max_dist + 1e-9, step)
    heights = np.full(len(dists), np.nan)
    top = float(pos[2]) + 3.0                            # above every wall and deck
    origin = np.zeros(3)
    direction = np.array([0.0, 0.0, -1.0])
    geomid = np.zeros(1, dtype=np.int32)
    for i, d in enumerate(dists):
        origin[:] = (float(pos[0]) + d * heading[0], float(pos[1]) + d * heading[1], top)
        hit = float(mujoco.mj_ray(model, data, origin, direction, ray_groups, 1, int(body_exclude), geomid))
        if hit >= 0.0:
            heights[i] = top - hit
    return dists, heights


def edges_in(dists, heights, *, min_change: float = MIN_EDGE) -> list[Edge]:
    """Every place the ground steps by at least ``min_change``, in order."""
    out: list[Edge] = []
    valid = np.isfinite(heights)
    if not valid.any():
        return out
    level = float(heights[valid][0])
    i = 1
    n = len(dists)
    prev = level
    while i < n:
        h = heights[i]
        if not np.isfinite(h):
            i += 1
            continue
        if abs(h - level) < min_change:
            prev = float(h)
            i += 1
            continue
        if abs(h - prev) < SHARP_FRACTION * min_change:
            # The ground got here gradually: a ramp, not an edge. Follow it.
            level, prev = float(h), float(h)
            i += 1
            continue
        # The new level runs from here until the ground steps again.
        new_level = float(h)
        j = i + 1
        while j < n and (not np.isfinite(heights[j]) or abs(heights[j] - new_level) < min_change):
            j += 1
        edge_dist = 0.5 * (dists[i - 1] + dists[i])
        length = (0.5 * (dists[j - 1] + dists[j]) - edge_dist) if j < n else float("inf")
        change = new_level - level
        returns = (change < 0 and j < n and length <= MAX_GAP
                   and float(heights[j]) >= level - MAX_DROP
                   and float(heights[j]) > new_level + min_change)
        out.append(Edge(dist=float(edge_dist), kind="rise" if change > 0 else "drop",
                        change=float(change), length=float(length), returns=bool(returns), level=level))
        level, prev, i = new_level, new_level, j
    return out


def plan_jump(edges: list[Edge], ground: float | None = None) -> dict | None:
    """The first edge a running jump can aim at, and how far before it to fire.

    Returns ``{"edge": Edge, "shape": str, "trigger": float}`` or None when
    nothing ahead is a jump target: open floor, a wall, a cliff, an edge
    already under the rods (nearer than MIN_TARGET: it is rolled, not
    jumped), or a rise that stands below the ball's own level (the ball
    would first drop into a pit or off a ledge: it looks again from there).

    ``ground`` is the height of the floor the ball stands on. Pass it when
    known: the first ray sample is not it inside a pipe or under a deck
    (the ray hits the roof).
    """
    if not edges:
        return None
    if ground is None:
        ground = edges[0].level                          # the level under the ball now
    for e in edges:
        if e.dist < MIN_TARGET:
            return None                                  # too late to aim at; roll it
        shape = e.shape
        if shape is None:
            if e.kind == "rise" and e.change > MAX_JUMP_RISE:
                return None                              # a wall blocks the view
            continue                                     # a step down: roll it, look past it
        if e.kind == "rise" and e.level < ground - MIN_EDGE:
            return None                                  # the far wall of a pit or a lower level
        return {"edge": e, "shape": shape, "trigger": TRIGGER[shape], "edges": edges, "ground": ground}
    return None


def hop_target(plan: dict) -> dict | None:
    """The surface a standing hop must land on, for ``hop_planner.plan_standing_hop``.

    In the plan's own frame: the ball at 0, distances along the heading,
    heights relative to the floor the ball stands on. A beam, tread or
    platform is landed on its top; a trench is crossed onto the level after
    it. ``far`` is the end of that surface, or 2 m on when it is not seen.
    """
    e, edges, ground = plan["edge"], plan["edges"], plan["ground"]
    if e.kind == "rise":
        top = e.level + e.change
        length = e.length if np.isfinite(e.length) else 2.0
        return {"near": e.dist, "far": e.dist + length, "height": top - ground}
    # a gap: the level after it begins where the ground comes back up
    after = [x for x in edges if x.dist > e.dist + e.length - 1e-6 and x.kind == "rise"]
    if not after:
        return None
    a = after[0]
    length = a.length if np.isfinite(a.length) else 2.0
    return {"near": a.dist, "far": a.dist + length, "height": a.level + a.change - ground}


class TerrainProbe:
    """Bound to one env: ``edges(heading)`` and ``plan(heading)`` from the live model.

    Three ray lines: the centre and one at each shoulder (the core radius to
    the side). A rise seen first by a shoulder line is what the ball's side
    would hit (a crate the centre line passes the corner of), so the plan
    takes the nearest rise of the three; drops are measured on the centre
    line, where the ball's weight goes.
    """

    def __init__(self, env):
        self.env = env

    def profile(self, heading, offset: float = 0.0, **kw):
        e = self.env
        heading = np.asarray(heading, dtype=np.float64)[:2]
        heading = heading / max(float(np.linalg.norm(heading)), 1e-9)
        pos = np.array(e.data.qpos[:3], dtype=np.float64)
        pos[:2] += offset * np.array([-heading[1], heading[0]])
        return profile_ahead(e.model, e.data, e._terrain_ray_groups, e.core_body_id, pos, heading, **kw)

    def edges(self, heading, offset: float = 0.0, **kw) -> list[Edge]:
        return edges_in(*self.profile(heading, offset=offset, **kw))

    def plan(self, heading) -> dict | None:
        r = float(getattr(self.env, "sphere_radius", 0.15))
        ground = float(self.env.data.qpos[2]) - r - 0.03           # the floor the ball stands on
        centre = self.edges(heading)
        plan = plan_jump(centre, ground=ground)
        if plan is None and any(e.kind == "rise" and e.change > MAX_JUMP_RISE for e in centre):
            return None          # a wall ahead: a shoulder line seeing past it is not a target
        for offset in (r, -r):
            side = plan_jump(self.edges(heading, offset=offset), ground=ground)
            if side is not None and side["edge"].kind == "rise" and (
                    plan is None or side["edge"].dist < plan["edge"].dist - 0.10):
                plan = side
        return plan
