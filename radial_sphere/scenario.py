"""Scenarios: what task the agent is asked to do.

A :class:`Scenario` is a small, serialisable spec describing one task instance:
where the sphere starts, where the goal is, the waypoints the controller/obs
track, and the breadcrumb markers to render.  Two kinds are supported:

* ``path``      — path navigation: follow a sinusoidal path to its end.
* ``goal``      — goal finding: reach a single goal point (random within an
                  arena); the waypoints are just a straight line spawn → goal.
* ``roundtrip`` — out-and-back: sine out, turnaround, return lane back beside
                  the spawn — the ball comes back toward the chase camera.

Generators:

    from radial_sphere import generate_scenario
    sc = generate_scenario("goal", cfg, seed=0)
    sc.save("scenario.json"); Scenario.load("scenario.json")
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import sample_path, sample_roundtrip

KINDS = ("path", "goal", "roundtrip", "obstacle", "maze")


def _arc_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


@dataclass
class Scenario:
    """One task instance for the agent (see module docstring)."""

    kind: str                # one of KINDS
    name: str
    spawn_xy: np.ndarray     # (2,) start position
    goal: np.ndarray         # (2,) target (== path_pts[-1])
    path_pts: np.ndarray     # (N, 2) waypoints the controller/obs track
    markers: np.ndarray      # (M, 2) breadcrumb markers to render (may be empty)
    path_length: float       # arc length, for normalising goal-distance in obs
    obstacles: np.ndarray = None   # (K, 3) pillars as (x, y, radius); may be empty
    walls: np.ndarray = None       # (M, 4) thin wall segments (x1, y1, x2, y2)
    geo_field: np.ndarray = None   # (H, W) geodesic distance to goal; -1 = blocked
    geo_origin: np.ndarray = None  # (2,) world xy of field cell (0, 0) centre
    geo_res: float = 0.0           # field cell size (m); 0 = no field

    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = np.empty((0, 3), dtype=np.float32)
        self.obstacles = np.asarray(self.obstacles, dtype=np.float32).reshape(-1, 3)
        if self.walls is None:
            self.walls = np.empty((0, 4), dtype=np.float32)
        self.walls = np.asarray(self.walls, dtype=np.float32).reshape(-1, 4)
        if self.geo_field is not None:
            self.geo_field = np.asarray(self.geo_field, dtype=np.float32)
            self.geo_origin = np.asarray(self.geo_origin, dtype=np.float32)

    # ------------------------------------------------------------------
    def nav_distance(self, xy) -> float | None:
        """Geodesic distance (m) from ``xy`` to the goal, through the maze.

        Continuous: min over nearby valid field cells of (cell value +
        straight distance to that cell's centre). None if no field.
        """
        if self.geo_field is None or self.geo_res <= 0:
            return None
        f, res = self.geo_field, float(self.geo_res)
        ny, nx = f.shape
        cx = (float(xy[0]) - float(self.geo_origin[0])) / res
        cy = (float(xy[1]) - float(self.geo_origin[1])) / res
        i0, j0 = int(round(cy)), int(round(cx))
        best = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i, j = i0 + di, j0 + dj
                if 0 <= i < ny and 0 <= j < nx and f[i, j] >= 0:
                    px = self.geo_origin[0] + j * res
                    py = self.geo_origin[1] + i * res
                    d = float(f[i, j]) + float(np.hypot(xy[0] - px, xy[1] - py))
                    best = d if best is None or d < best else best
        return best

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "spawn_xy": np.asarray(self.spawn_xy, dtype=float).tolist(),
            "goal": np.asarray(self.goal, dtype=float).tolist(),
            "path_pts": np.asarray(self.path_pts, dtype=float).tolist(),
            "markers": np.asarray(self.markers, dtype=float).reshape(-1, 2).tolist(),
            "path_length": float(self.path_length),
            "obstacles": np.asarray(self.obstacles, dtype=float).reshape(-1, 3).tolist(),
            "walls": np.asarray(self.walls, dtype=float).reshape(-1, 4).tolist(),
            "geo_field": self.geo_field.tolist() if self.geo_field is not None else None,
            "geo_origin": self.geo_origin.tolist() if self.geo_origin is not None else None,
            "geo_res": float(self.geo_res),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            kind=d["kind"],
            name=d["name"],
            spawn_xy=np.asarray(d["spawn_xy"], dtype=np.float32),
            goal=np.asarray(d["goal"], dtype=np.float32),
            path_pts=np.asarray(d["path_pts"], dtype=np.float32),
            markers=np.asarray(d["markers"], dtype=np.float32).reshape(-1, 2),
            path_length=float(d["path_length"]),
            obstacles=np.asarray(d.get("obstacles", []), dtype=np.float32).reshape(-1, 3),
            walls=np.asarray(d.get("walls", []), dtype=np.float32).reshape(-1, 4),
            geo_field=d.get("geo_field"),
            geo_origin=d.get("geo_origin"),
            geo_res=float(d.get("geo_res", 0.0)),
        )

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path) -> "Scenario":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def path_scenario(cfg, *, rng=None, name: str = "path") -> Scenario:
    """Path navigation: the sinusoidal path from the config ``path`` section."""
    p = cfg.path
    pts = sample_path(240, p.length, p.amplitude, p.waves).astype(np.float32)
    markers = sample_path(40, p.length, p.amplitude, p.waves).astype(np.float32)
    return Scenario(
        kind="path", name=name,
        spawn_xy=pts[0].copy(), goal=pts[-1].copy(),
        path_pts=pts, markers=markers, path_length=_arc_length(pts),
    )


def goal_scenario(cfg, *, rng=None, name: str = "goal") -> Scenario:
    """Goal finding: a random goal in the arena; waypoints are a straight line."""
    rng = rng if rng is not None else np.random.default_rng()
    sc = getattr(cfg, "scenario", None)
    goal_cfg = getattr(sc, "goal", None) if sc is not None else None
    x_range = tuple(getattr(goal_cfg, "x_range", (2.0, 5.0))) if goal_cfg else (2.0, 5.0)
    y_range = tuple(getattr(goal_cfg, "y_range", (-1.5, 1.5))) if goal_cfg else (-1.5, 1.5)

    spawn = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([rng.uniform(*x_range), rng.uniform(*y_range)], dtype=np.float32)
    # Straight-line waypoints so the look-ahead controller heads to the goal.
    pts = np.linspace(spawn, goal, 60).astype(np.float32)
    # No breadcrumbs for goal finding — only the goal marker is shown.
    markers = np.empty((0, 2), dtype=np.float32)
    return Scenario(
        kind="goal", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=float(np.linalg.norm(goal - spawn)),
    )


def obstacle_scenario(cfg, *, rng=None, name: str = "obstacle") -> Scenario:
    """Goal finding with random pillars in the way.

    The straight spawn → goal line is blocked by cylinders, so the steering
    policy must plan detours; the scripted controller drives straight and
    gets stuck (that contrast is the point of the task).
    """
    rng = rng if rng is not None else np.random.default_rng()
    sc = getattr(cfg, "scenario", None)
    goal_cfg = getattr(sc, "goal", None) if sc is not None else None
    ob_cfg = getattr(sc, "obstacles", None) if sc is not None else None
    x_range = tuple(getattr(goal_cfg, "x_range", (2.0, 5.0))) if goal_cfg else (2.0, 5.0)
    y_range = tuple(getattr(goal_cfg, "y_range", (-1.5, 1.5))) if goal_cfg else (-1.5, 1.5)
    n_range = tuple(getattr(ob_cfg, "n_range", (3, 6))) if ob_cfg else (3, 6)
    # one fixed radius for all pillars: the sim objects are built once,
    # only their positions move between episodes
    r = float(getattr(ob_cfg, "radius", 0.25)) if ob_cfg else 0.25
    clearance = float(getattr(ob_cfg, "clearance", 0.9)) if ob_cfg else 0.9
    gap = float(getattr(ob_cfg, "min_gap", 0.7)) if ob_cfg else 0.7

    n_blocking = int(getattr(ob_cfg, "n_blocking", 2)) if ob_cfg else 2

    spawn = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([rng.uniform(*x_range), rng.uniform(*y_range)], dtype=np.float32)
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    pillars: list[np.ndarray] = []

    def _fits(p: np.ndarray) -> bool:
        if np.linalg.norm(p - spawn) < clearance + r:
            return False
        if np.linalg.norm(p - goal) < clearance + r:
            return False
        # keep pillar pairs at least `gap` apart so a passage always exists
        return all(np.linalg.norm(p - q[:2]) >= r + q[2] + gap for q in pillars)

    # First place blockers ON the spawn → goal line (small lateral jitter),
    # otherwise random layouts often leave the straight line free and the
    # task degenerates to plain goal finding.
    d = goal - spawn
    length = float(np.linalg.norm(d))
    u = d / length
    perp = np.array([-u[1], u[0]])
    t_lo = (clearance + r) / length
    tries = 0
    while len(pillars) < min(n_blocking, n) and t_lo < 0.5 and tries < 100:
        tries += 1
        t = rng.uniform(t_lo, 1.0 - t_lo)
        p = spawn + (t * length) * u + rng.uniform(-0.2, 0.2) * perp
        if _fits(p):
            pillars.append(np.array([p[0], p[1], r], dtype=np.float32))

    tries = 0
    while len(pillars) < n and tries < 300:
        tries += 1
        p = np.array([rng.uniform(0.6, goal[0] + 0.6), rng.uniform(-2.0, 2.0)])
        if _fits(p):
            pillars.append(np.array([p[0], p[1], r], dtype=np.float32))

    pts = np.linspace(spawn, goal, 60).astype(np.float32)
    return Scenario(
        kind="obstacle", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=np.empty((0, 2), dtype=np.float32),
        path_length=float(np.linalg.norm(goal - spawn)),
        obstacles=np.asarray(pillars, dtype=np.float32).reshape(-1, 3),
    )


def _geodesic_field(walls, bounds, goal, res: float = 0.25, inflate: float = 0.32):
    """Distance-to-goal field over free space (Dijkstra, 8-connected).

    ``inflate`` grows walls by the ball radius, so the field measures where
    the ball CENTRE can actually travel.  Blocked cells hold -1.
    """
    import heapq

    x0, y0, x1, y1 = bounds
    nx = int(round((x1 - x0) / res)) + 1
    ny = int(round((y1 - y0) / res)) + 1
    xs = x0 + np.arange(nx) * res
    ys = y0 + np.arange(ny) * res
    gx, gy = np.meshgrid(xs, ys)          # (ny, nx)
    blocked = np.zeros((ny, nx), dtype=bool)
    for wx1, wy1, wx2, wy2 in np.asarray(walls, dtype=float).reshape(-1, 4):
        ex, ey = wx2 - wx1, wy2 - wy1
        L2 = ex * ex + ey * ey
        t = ((gx - wx1) * ex + (gy - wy1) * ey) / (L2 if L2 > 1e-12 else 1.0)
        t = np.clip(t, 0.0, 1.0)
        d = np.hypot(gx - (wx1 + t * ex), gy - (wy1 + t * ey))
        blocked |= d < inflate

    field = np.full((ny, nx), -1.0, dtype=np.float32)
    gj = int(round((goal[0] - x0) / res))
    gi = int(round((goal[1] - y0) / res))
    gi, gj = np.clip(gi, 0, ny - 1), np.clip(gj, 0, nx - 1)
    if blocked[gi, gj]:                    # goal cell inflated shut: free it
        blocked[gi, gj] = False
    dist = {(gi, gj): 0.0}
    heap = [(0.0, (gi, gj))]
    steps = [(-1, 0, res), (1, 0, res), (0, -1, res), (0, 1, res),
             (-1, -1, res * 2 ** 0.5), (-1, 1, res * 2 ** 0.5),
             (1, -1, res * 2 ** 0.5), (1, 1, res * 2 ** 0.5)]
    while heap:
        d, (i, j) = heapq.heappop(heap)
        if d > dist.get((i, j), np.inf):
            continue
        field[i, j] = d
        for di, dj, c in steps:
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx and not blocked[ni, nj]:
                ndist = d + c
                if ndist < dist.get((ni, nj), np.inf):
                    dist[(ni, nj)] = ndist
                    heapq.heappush(heap, (ndist, (ni, nj)))
    return field, np.array([x0, y0], dtype=np.float32), res


def maze_scenario(cfg, *, rng=None, name: str = "maze") -> Scenario:
    """Maze navigation. Level 1: a serpentine corridor of thin iron walls.

    The reward distance comes from the geodesic field (distance THROUGH the
    corridor), not the straight line — see :meth:`Scenario.nav_distance`.
    """
    mz = getattr(cfg.scenario, "maze", None)
    level = int(getattr(mz, "level", 1)) if mz else 1
    cell = float(getattr(mz, "cell", 1.5)) if mz else 1.5
    cols = int(getattr(mz, "cols", 5)) if mz else 5
    rows = int(getattr(mz, "rows", 4)) if mz else 4
    if level != 1:
        raise NotImplementedError("maze levels 2/3 not built yet")

    # arena bounds; cell (i, j) centre = (i*cell, j*cell)
    x0, y0 = -cell / 2, -cell / 2
    x1, y1 = (cols - 0.5) * cell, (rows - 0.5) * cell
    walls = [(x0, y0, x1, y0), (x0, y1, x1, y1),
             (x0, y0, x0, y1), (x1, y0, x1, y1)]
    # serpentine shelves: each inner wall leaves one cell open at one end
    for j in range(rows - 1):
        yw = (j + 0.5) * cell
        if j % 2 == 0:
            walls.append((x0, yw, x1 - cell, yw))
        else:
            walls.append((x0 + cell, yw, x1, yw))

    # centreline waypoints through the corridor, densified for the tracker
    coarse = []
    for j in range(rows):
        xs = range(cols) if j % 2 == 0 else range(cols - 1, -1, -1)
        coarse += [(i * cell, j * cell) for i in xs]
    pts = [np.array(coarse[0], dtype=np.float32)]
    for a, b in zip(coarse[:-1], coarse[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.linalg.norm(b - a) / 0.15))
        pts += [ (a + (b - a) * k / n).astype(np.float32) for k in range(1, n + 1) ]
    pts = np.asarray(pts, dtype=np.float32)
    markers = pts[:: max(1, len(pts) // 38)]

    spawn, goal = pts[0].copy(), pts[-1].copy()
    field, origin, res = _geodesic_field(walls, (x0, y0, x1, y1), goal)
    return Scenario(
        kind="maze", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=_arc_length(pts),
        walls=np.asarray(walls, dtype=np.float32),
        geo_field=field, geo_origin=origin, geo_res=res,
    )


def roundtrip_scenario(cfg, *, rng=None, name: str = "roundtrip") -> Scenario:
    """Out-and-back: follow the sine out, turn around, return beside the spawn."""
    p = cfg.path
    lane = float(getattr(p, "lane_offset", 3.0 * float(p.amplitude)))
    pts = sample_roundtrip(480, p.length, p.amplitude, p.waves, lane).astype(np.float32)
    markers = pts[::8].copy()
    return Scenario(
        kind="roundtrip", name=name,
        spawn_xy=pts[0].copy(), goal=pts[-1].copy(),
        path_pts=pts, markers=markers, path_length=_arc_length(pts),
    )


_GENERATORS = {"path": path_scenario, "goal": goal_scenario,
               "roundtrip": roundtrip_scenario, "obstacle": obstacle_scenario,
               "maze": maze_scenario}


def generate_scenario(kind: str, cfg, *, seed=None, name: str | None = None) -> Scenario:
    """Generate a scenario of the given ``kind`` (``"path"`` | ``"goal"``)."""
    if kind not in _GENERATORS:
        raise ValueError(f"unknown scenario kind {kind!r}; expected one of {KINDS}")
    rng = np.random.default_rng(seed)
    return _GENERATORS[kind](cfg, rng=rng, name=name or kind)
