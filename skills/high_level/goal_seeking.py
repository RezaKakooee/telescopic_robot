"""Get to a point on the floor, deciding what to do about what is in the way.

`go_to_goal` is the first high-level skill. It returns a skill *name* and its
arguments, never rod extensions, which is what separates this level from the
one below it. The caller runs the command:

    cmd = go_to_goal(ball_xy=p, goal_xy=g, obstacles=rocks, lin_vel=v)
    targets = execute_skill(cmd.skill, quat, dirs_body, max_extend, **cmd.kwargs)

Two decisions live here, and neither belongs lower down.

**Route.** The straight line to the goal is used when it is clear. When a
declared obstacle sits on it, a waypoint is placed beside that obstacle and
the check repeats. The result is a polyline for `follow_path`.

**Over or around.** A box on the route is measured against
`jump_planner.max_clearable`, which is the tallest thing this build can
guarantee to clear. Low enough, and the route stays straight and the command
becomes `climb_stairs`. Too tall, and the box becomes one more circle to
route around. That question needs the jump calibration and the map at the
same time, so it can only be answered here.

What this module does **not** do is time the phases of a hop. `climb_stairs`
takes a `phase`, and stepping through crouch, take-off, airborne and landing
needs a step counter. These functions are pure, so they have no counter to
keep. The command carries the approach phase and the `JumpPlan` beside it in
`meta`; the caller drives the phases, the same way `demos/stairs/runner.py`
already does.

`go_to_goal` is deliberately absent from `SKILL_REGISTRY`. Every name in that
registry returns a `(n_bars,)` array of rod targets, and `execute_skill`
summarises those targets as telemetry. A planner returns a command instead,
so putting it there would break the one promise the registry makes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..mid_level.jump_planner import JumpPlan, max_clearable, plan_jump

#: How far to the side of an obstacle a detour waypoint is placed, on top of
#: the obstacle's own radius. The ball is 0.15 m across the shell, so this
#: leaves roughly a ball's width of air at the closest point.
DEFAULT_CLEARANCE = 0.35

#: Give up adding detours after this many. One detour can push the route into
#: a second obstacle, so a map needs more waypoints than it has obstacles. A
#: run-away loop is worse than an honest `route_clear=False`.
MAX_DETOURS = 24


@dataclass(frozen=True)
class SkillCommand:
    """One step of a plan: which skill to run, and with what.

    `kwargs` is ready to splat into `execute_skill` after the three
    positional arguments every skill takes (`quat`, `dirs_body`,
    `max_extend`). `meta` explains the choice; nothing reads it to drive the
    robot, so a HUD or a test can use it freely.
    """

    skill: str
    kwargs: dict = field(default_factory=dict)
    reason: str = ""
    meta: dict = field(default_factory=dict)


def _as_circles(obstacles) -> np.ndarray:
    """(N, 3) array of [x, y, radius]. Empty input gives a (0, 3) array."""
    if obstacles is None:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.asarray(list(obstacles), dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] < 3:
        raise ValueError("each obstacle is (x, y, radius)")
    return arr[:, :3].copy()


def _as_boxes(steps) -> np.ndarray:
    """(N, 5) array of [x, y, half_x, half_y, height], as `Scenario.steps`."""
    if steps is None:
        return np.zeros((0, 5), dtype=np.float64)
    arr = np.asarray(list(steps), dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] < 5:
        raise ValueError("each step is (x, y, half_x, half_y, height)")
    return arr[:, :5].copy()


def _segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, np.ndarray]:
    """Distance from point `c` to segment `a`-`b`, and the closest point on it."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(c - a)), a.copy()
    t = float(np.clip(((c - a) @ ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(c - closest)), closest


#: Slack on the clearance test, so a waypoint placed exactly on the inflated
#: circle is not judged to be inside it by a rounding error.
_EPS = 1e-9

#: How far out to try pushing a detour waypoint, as multiples of the inflated
#: radius. A waypoint dropped exactly on the circle does not help: the two
#: segments that meet there are chords, and a chord cuts inside. Stepping the
#: multiplier out until both segments clear is the cheapest fix that always
#: makes progress, so the same obstacle can never be handled twice.
_PUSH_LADDER = (1.0, 1.15, 1.35, 1.6, 2.0, 2.6, 3.4)


def _too_close(a: np.ndarray, b: np.ndarray, centre: np.ndarray, need: float) -> bool:
    return _segment_distance(a, b, centre)[0] < need - _EPS


def _inside_any(point: np.ndarray, circles: np.ndarray, clearance: float,
                skip: int) -> bool:
    """Is `point` within clearance of any circle other than `skip`?"""
    for j, (cx, cy, r) in enumerate(circles):
        if j == skip:
            continue
        if float(np.linalg.norm(point - np.array([cx, cy]))) < float(r) + clearance - _EPS:
            return True
    return False


def _blocking(route: list[np.ndarray], circles: np.ndarray, clearance: float,
              ignore: set[int]):
    """The first (segment index, circle index) the route runs too close to."""
    for i in range(len(route) - 1):
        worst = None
        for j, (cx, cy, r) in enumerate(circles):
            if j in ignore:
                continue
            need = float(r) + clearance
            centre = np.array([cx, cy])
            d, _ = _segment_distance(route[i], route[i + 1], centre)
            if d < need - _EPS and (worst is None or (need - d) > worst[1]):
                worst = (j, need - d)
        if worst is not None:
            return i, worst[0]
    return None


def plan_route(start, goal, obstacles=(), *, clearance: float = DEFAULT_CLEARANCE,
               max_detours: int = MAX_DETOURS) -> tuple[np.ndarray, bool]:
    """Waypoints from `start` to `goal` that keep `clearance` from every circle.

    The straight line is tried first. When a segment cuts inside an obstacle's
    inflated radius, the obstacle's centre is projected onto that segment and
    one waypoint goes out along that direction, on the side the route already
    passes. Taking the near side keeps the detour short.

    The waypoint is pushed out until *both* new segments clear, stepping
    through `_PUSH_LADDER`. Placing it exactly on the inflated circle is not
    enough, because the two segments meeting there are chords and a chord cuts
    back inside. Insisting that both segments clear is what makes each pass
    finish with one obstacle fewer, instead of re-placing the same waypoint.

    A circle that swallows the start or the goal is left out of the verdict.
    No polyline between those two points can clear it, so counting it would
    make `clear` permanently False and say nothing useful. A robot standing in
    an obstacle has to drive out first, and the straight line does that.

    Returns `(route, clear)`. `clear` is False when `max_detours` ran out with
    an obstacle still on the line, which heavily overlapping obstacles can
    cause. The route is still returned, because a caller that is already
    moving is better off with the best line found than with nothing.
    """
    circles = _as_circles(obstacles)
    route = [np.asarray(start, dtype=np.float64)[:2].copy(),
             np.asarray(goal, dtype=np.float64)[:2].copy()]
    if len(circles) == 0:
        return np.asarray(route), True

    # Circles that swallow the start or the goal are excluded from the verdict.
    # No polyline between those two points can clear them, so counting them
    # would make `clear` permanently False and say nothing.
    endpoint_blocked = set()
    for j, (cx, cy, r) in enumerate(circles):
        need = float(r) + clearance
        centre = np.array([cx, cy])
        if (float(np.linalg.norm(route[0] - centre)) < need
                or float(np.linalg.norm(route[-1] - centre)) < need):
            endpoint_blocked.add(j)

    for _ in range(max_detours):
        hit = _blocking(route, circles, clearance, endpoint_blocked)
        if hit is None:
            break
        i, j = hit
        cx, cy, r = circles[j]
        centre = np.array([cx, cy], dtype=np.float64)
        need = float(r) + clearance
        a, b = route[i], route[i + 1]
        _, closest = _segment_distance(a, b, centre)

        away = closest - centre
        norm = float(np.linalg.norm(away))
        if norm < 1e-6:
            # The line runs through the centre, so there is no near side to
            # pick. Step off along the segment's left normal.
            seg = b - a
            seg_norm = float(np.linalg.norm(seg))
            away = (np.array([-seg[1], seg[0]]) / seg_norm if seg_norm > 1e-9
                    else np.array([0.0, 1.0]))
        else:
            away = away / norm

        # A candidate has to do two things: clear *this* circle on both new
        # segments, and not land inside a different one. Skipping the second
        # test is what made the planner report a clear route through a rock:
        # the waypoint sat inside its neighbour, no push could ever free a
        # segment ending there, and that neighbour was quietly written off.
        # The near side is tried first because it is the shorter detour; the
        # far side is the fallback when the near side is crowded.
        fallback = None
        placed = None
        for k in _PUSH_LADDER:
            for side in (away, -away):
                w = centre + (k * need) * side
                if _too_close(a, w, centre, need) or _too_close(w, b, centre, need):
                    continue
                if fallback is None:
                    fallback = w
                if not _inside_any(w, circles, clearance, skip=j):
                    placed = w
                    break
            if placed is not None:
                break
        placed = placed if placed is not None else fallback
        if placed is None:
            break       # nothing helps this one; the verdict below says so
        route.insert(i + 1, placed)

    clear = _blocking(route, circles, clearance, endpoint_blocked) is None
    return np.asarray(route), clear


@lru_cache(maxsize=4)
def _climbable_height_cached(mode: str) -> float:
    return max_clearable(mode=mode)


@lru_cache(maxsize=256)
def _plan_jump_cached(height: float, half_depth: float, mode: str):
    """`plan_jump` for the stock calibration, keyed on its float inputs.

    The box list does not change between control steps, so the same height and
    depth are asked for over and over. One uncached call is about 0.8 ms.
    """
    return plan_jump(height, half_depth, mode=mode)


def _jump_plan(height: float, half_depth: float, mode: str, calibration):
    if calibration is None:
        return _plan_jump_cached(float(height), float(half_depth), mode)
    return plan_jump(float(height), float(half_depth), mode=mode,
                     calibration=calibration)


def climbable_height(mode: str = "over", calibration=None) -> float:
    """Tallest box this build can guarantee to clear, in metres.

    A thin wrapper so callers do not have to reach into `mid_level` for the
    one number that decides over-or-around.

    The answer is cached for the stock calibration, because it is not cheap:
    `max_clearable` re-plans a jump at every height from 5 cm to 80 cm, which
    measured 57 ms. `go_to_goal` asks for it on every control step, so
    without the cache one planner call costs more than a 100 Hz step budget.
    An explicit `calibration` skips the cache, since a dict is not hashable
    and a test that passes one wants that exact answer.
    """
    if calibration is None:
        return _climbable_height_cached(mode)
    return max_clearable(mode=mode, calibration=calibration)


def _step_on_route(boxes: np.ndarray, route: np.ndarray, ball_xy: np.ndarray,
                   lookahead: float):
    """The nearest box ahead whose footprint the route enters, if any.

    Returns `(index, distance)`, the straight-line distance from the ball to
    where the route meets the box, or None. A box counts as on the route when
    the route passes within its footprint radius, and as ahead when that
    distance is no more than `lookahead`.
    """
    best = None
    for j, (bx, by, hx, hy, _h) in enumerate(boxes):
        centre = np.array([bx, by], dtype=np.float64)
        reach = float(np.hypot(hx, hy))
        for i in range(len(route) - 1):
            d, closest = _segment_distance(route[i], route[i + 1], centre)
            if d > reach:
                continue
            along = float(np.linalg.norm(closest - ball_xy))
            if along <= lookahead and (best is None or along < best[1]):
                best = (j, along)
    return best


def go_to_goal(
    *,
    ball_xy,
    goal_xy,
    lin_vel=None,
    obstacles=(),
    steps=(),
    speed: float = 1.2,
    goal_tolerance: float = 0.35,
    clearance: float = DEFAULT_CLEARANCE,
    lookahead: float = 0.85,
    step_lookahead: float = 2.0,
    calibration=None,
) -> SkillCommand:
    """Choose the skill that gets the robot closer to `goal_xy` right now.

    Parameters
    ----------
    ball_xy : (2,) world position of the robot.
    goal_xy : (2,) world position of the goal.
    lin_vel : (2,) or (3,) world velocity, forwarded to whichever skill runs.
    obstacles : iterable of `(x, y, radius)` to route around.
    steps : iterable of `(x, y, half_x, half_y, height)`, the same shape as
        `Scenario.steps`. Each one is measured against the jump calibration:
        low enough to clear becomes a `climb_stairs` leg, too tall becomes
        another circle to route around.
    speed : cruising speed in m/s, passed through to the chosen skill.
    goal_tolerance : how close counts as arrived, in metres.
    clearance : air to leave beside an obstacle, on top of its radius.
    lookahead : pursuit distance handed to `follow_path`.
    step_lookahead : how far ahead a box has to be before the command
        switches to `climb_stairs`. Keep it longer than the jump's own
        trigger distance so the approach has room to build speed.
    calibration : jump calibration override, for tests.

    Returns
    -------
    SkillCommand
        `skill` is one of ``"stop"``, ``"climb_stairs"`` or ``"follow_path"``.
    """
    p = np.asarray(ball_xy, dtype=np.float64)[:2]
    g = np.asarray(goal_xy, dtype=np.float64)[:2]
    dist_to_goal = float(np.linalg.norm(g - p))

    if dist_to_goal <= goal_tolerance:
        return SkillCommand(
            skill="stop",
            kwargs={"lin_vel": lin_vel},
            reason="within goal tolerance",
            meta={"dist_to_goal": dist_to_goal, "route_clear": True},
        )

    boxes = _as_boxes(steps)
    ceiling = climbable_height("over", calibration=calibration)

    # Split the declared boxes by the one number that decides it. A box the
    # robot can clear stays on the route; one it cannot becomes an obstacle,
    # with a radius that covers its whole footprint.
    low = boxes[boxes[:, 4] <= ceiling] if len(boxes) else boxes
    tall = boxes[boxes[:, 4] > ceiling] if len(boxes) else boxes
    circles = list(_as_circles(obstacles))
    for bx, by, hx, hy, _h in tall:
        circles.append(np.array([bx, by, float(np.hypot(hx, hy))]))

    route, route_clear = plan_route(p, g, circles, clearance=clearance)

    meta = {
        "dist_to_goal": dist_to_goal,
        "route": route,
        "route_clear": route_clear,
        "n_detours": max(0, len(route) - 2),
        "climbable_height": ceiling,
        "n_obstacles": len(circles),
        "n_climbable_steps": int(len(low)),
    }

    hit = _step_on_route(low, route, p, step_lookahead) if len(low) else None
    if hit is not None:
        j, along = hit
        bx, by, hx, hy, h = low[j]
        # Aim the approach at the box, not at the goal, so the ball meets the
        # face square. A skewed hit lands on a corner. Standing on the box
        # centre is the only case with no direction to take, and then the
        # goal serves.
        aim = np.array([bx, by]) - p
        if float(np.linalg.norm(aim)) <= 1e-6:
            aim = g - p
        aim_norm = float(np.linalg.norm(aim))
        d_hat = aim / aim_norm if aim_norm > 1e-9 else np.array([1.0, 0.0])
        plan = _jump_plan(float(h), float(np.hypot(hx, hy)), "over", calibration)
        meta.update({"step_index": j, "step_distance": along,
                     "step_height": float(h), "jump_plan": plan})
        return SkillCommand(
            skill="climb_stairs",
            kwargs={"d_hat": d_hat, "speed": speed, "lin_vel": lin_vel,
                    "phase": "approach"},
            reason=f"a {float(h):.2f} m step is {along:.2f} m ahead and clearable",
            meta=meta,
        )

    return SkillCommand(
        skill="follow_path",
        kwargs={"ball_xy": p, "path_pts": route, "lin_vel": lin_vel,
                "speed": speed, "lookahead": lookahead,
                "goal_tolerance": goal_tolerance},
        reason=("straight line to the goal" if len(route) == 2
                else f"{len(route) - 2} detour waypoint(s) around obstacles"),
        meta=meta,
    )


__all__ = ["SkillCommand", "go_to_goal", "plan_route", "climbable_height",
           "JumpPlan", "DEFAULT_CLEARANCE", "MAX_DETOURS"]
