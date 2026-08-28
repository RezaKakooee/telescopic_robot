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
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import sample_path, sample_roundtrip

KINDS = ("path", "goal", "roundtrip", "obstacle", "maze", "rocky_terrain", "slopes", "stairs", "glass_pipe", "extreme_gauntlet", "skill_course", "platform_course", "pillar_course", "circle_track", "gap_bridge", "chimney")




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
    gaps: list | np.ndarray = None # Floor trenches/cracks (x, y, hx, hy, depth)
    sand_patches: list | np.ndarray = None # Sand zones (x, y, hx, hy)
    stones: list | np.ndarray = None # Pebble clusters (x, y, hx, hy, n, max_sz)
    steps: list | np.ndarray = None # Ground step curbs (x, y, hx, hy, height)
    ramps: list | np.ndarray = None # Incline slopes / ramps (cx, cy, length, width, h_change, pitch_deg, yaw)
    staircases: list | np.ndarray = None # Multi-step flights (start_x, start_y, n_steps, rise, run, width, yaw, is_down)
    pipes: list | np.ndarray = None # Hollow cylindrical transparent tubes (start_x, start_y, length, in_rad, out_rad, yaw)
    yardlines: list | np.ndarray = None # Painted athletic track distance stripes (x, y, hx, hy, rgba)
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

    # Optional traversable wooden timber plank across the path
    wood_cfg = getattr(sc, "wood_plank", None) if sc is not None else None
    hazards_cfg = getattr(sc, "hazards", None) if sc is not None else None
    steps = []
    gaps = []
    stones = []
    sand_patches = []

    if wood_cfg is not None and getattr(wood_cfg, "enabled", False):
        pos = getattr(wood_cfg, "pos", None)
        if pos is None:
            # Place at 55% along spawn -> goal
            pos = 0.55 * spawn + 0.45 * goal
        half_w = float(getattr(wood_cfg, "half_width", 0.11))
        half_l = float(getattr(wood_cfg, "half_length", 0.75))
        height = float(getattr(wood_cfg, "height", 0.075))
        steps.append([float(pos[0]), float(pos[1]), half_w, half_l, height])

    if hazards_cfg is not None and getattr(hazards_cfg, "enabled", False):
        # Custom gaps/holes
        custom_gaps = getattr(hazards_cfg, "gaps", None)
        if custom_gaps is not None:
            gaps.extend(custom_gaps)
        elif int(getattr(hazards_cfg, "n_gaps", 0)) > 0:
            # Place a floor hole/trench along the path
            mid_gap = 0.30 * spawn + 0.70 * goal
            gdepth = float(getattr(hazards_cfg, "gap_depth", 0.10))
            gaps.append([float(mid_gap[0]), float(mid_gap[1]), 0.11, 0.70, gdepth])

        # Custom stones / boulders
        custom_stones = getattr(hazards_cfg, "stones", None)
        if custom_stones is not None:
            stones.extend(custom_stones)
        elif int(getattr(hazards_cfg, "n_stones", 0)) > 0:
            mid_st = 0.82 * spawn + 0.18 * goal
            stones.append([float(mid_st[0]), float(mid_st[1]), 0.35, 0.50, 16, 0.032])

        # Custom sand patches
        custom_sand = getattr(hazards_cfg, "sand_patches", None)
        if custom_sand is not None:
            sand_patches.extend(custom_sand)
        elif int(getattr(hazards_cfg, "n_sand", 0)) > 0:
            mid_sand = 0.60 * spawn + 0.40 * goal
            sand_patches.append([float(mid_sand[0]), float(mid_sand[1]), 0.45, 0.60])

    return Scenario(
        kind="goal", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=float(np.linalg.norm(goal - spawn)),
        steps=steps if steps else None,
        gaps=gaps if gaps else None,
        stones=stones if stones else None,
        sand_patches=sand_patches if sand_patches else None,
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

    ob_enabled = bool(getattr(ob_cfg, "enabled", True)) if ob_cfg else True
    n_blocking = int(getattr(ob_cfg, "n_blocking", 2)) if (ob_cfg and ob_enabled) else 0

    spawn = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([rng.uniform(*x_range), rng.uniform(*y_range)], dtype=np.float32)
    n = int(rng.integers(n_range[0], n_range[1] + 1)) if ob_enabled else 0
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
            if not (0 <= ni < ny and 0 <= nj < nx):
                continue
            # A diagonal step may not squeeze between two blocked cardinal
            # neighbours (corner-cutting would pass through a wall endpoint).
            cuts_corner = di != 0 and dj != 0 and (
                blocked[i, nj] or blocked[ni, j]
            )
            if not blocked[ni, nj] and not cuts_corner:
                ndist = d + c
                if ndist < dist.get((ni, nj), np.inf):
                    dist[(ni, nj)] = ndist
                    heapq.heappush(heap, (ndist, (ni, nj)))
    return field, np.array([x0, y0], dtype=np.float32), res


def maze_scenario(cfg, *, rng=None, name: str = "maze") -> Scenario:
    """Maze navigation: fixed serpentine level 1 or random perfect level 3.

    The reward distance comes from the geodesic field (distance THROUGH the
    corridor), not the straight line — see :meth:`Scenario.nav_distance`.
    """
    rng = rng if rng is not None else np.random.default_rng()
    mz = getattr(cfg.scenario, "maze", None)
    level = int(getattr(mz, "level", 1)) if mz else 1
    cell = float(getattr(mz, "cell", 1.5)) if mz else 1.5
    cols = int(getattr(mz, "cols", 5)) if mz else 5
    rows = int(getattr(mz, "rows", 4)) if mz else 4
    if level not in (1, 2, 3):
        raise ValueError(f"unknown maze level {level}; expected 1, 2, or 3")

    # arena bounds; cell (i, j) centre = (i*cell, j*cell)
    x0, y0 = -cell / 2, -cell / 2
    x1, y1 = (cols - 0.5) * cell, (rows - 0.5) * cell
    if level in (2, 3):
        return _random_maze_scenario(cfg, rng, name, cell, cols, rows,
                                     (x0, y0, x1, y1), level=level)

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


def _random_maze_scenario(cfg, rng, name, cell, cols, rows, bounds, level: int = 3) -> Scenario:
    """Generate a maze (level 2 multi-loop braid maze or level 3 perfect tree maze).

    Every layout uses the same number of cell-length wall pieces.  This lets
    the simulator move the existing fixed bodies at reset instead of rebuilding
    the physics world.  The policy gets the raw goal direction and geodesic
    distance, but not the solution route through ``path_pts``.
    """
    if cols < 2 or rows < 2:
        raise ValueError("maze level needs at least 2 rows and 2 columns")

    mz = cfg.scenario.maze
    min_route = float(getattr(mz, "min_route", 20.0))
    max_route = float(getattr(mz, "max_route", 35.0))
    attempts = int(getattr(mz, "generation_attempts", 100))
    layout_seed = getattr(mz, "layout_seed", None)
    random_endpoints = bool(getattr(mz, "random_endpoints", False))
    random_start = bool(getattr(mz, "random_start", False))
    random_goal = bool(getattr(mz, "random_goal", False))
    endpoint_min_route = float(getattr(mz, "endpoint_min_route", 7.5))
    endpoint_max_route = float(getattr(mz, "endpoint_max_route", max_route))
    loop_fraction = float(getattr(mz, "loop_fraction", 0.35))
    layout_rng = (np.random.default_rng(int(layout_seed))
                  if layout_seed is not None else rng)
    n_cells = cols * rows

    def neighbours(k):
        x, y = k % cols, k // cols
        out = []
        if x > 0:
            out.append(k - 1)
        if x + 1 < cols:
            out.append(k + 1)
        if y > 0:
            out.append(k - cols)
        if y + 1 < rows:
            out.append(k + cols)
        return out

    def carve():
        """Randomised depth-first spanning tree, plus extra loops if level == 2."""
        adjacency = [[] for _ in range(n_cells)]
        start = int(layout_rng.integers(n_cells))
        seen, stack = {start}, [start]
        while stack:
            a = stack[-1]
            choices = [b for b in neighbours(a) if b not in seen]
            if not choices:
                stack.pop()
                continue
            b = int(layout_rng.choice(choices))
            adjacency[a].append(b)
            adjacency[b].append(a)
            seen.add(b)
            stack.append(b)

        if level == 2:
            # Multi-loop braid maze: open extra passages between adjacent cells
            candidates = []
            for u in range(n_cells):
                for v in neighbours(u):
                    if v > u and v not in adjacency[u]:
                        candidates.append((u, v))
            if candidates:
                layout_rng.shuffle(candidates)
                n_extra = max(1, int(len(candidates) * loop_fraction))
                for u, v in candidates[:n_extra]:
                    adjacency[u].append(v)
                    adjacency[v].append(u)

        return adjacency

    def farthest(adjacency, source):
        parent = {source: None}
        distance = {source: 0}
        queue = deque([source])
        while queue:
            a = queue.popleft()
            for b in adjacency[a]:
                if b not in distance:
                    parent[b] = a
                    distance[b] = distance[a] + 1
                    queue.append(b)
        end = max(distance, key=distance.get)
        return end, distance[end], parent

    def tree_distance(adjacency, source, target):
        distance = {source: 0}
        queue = deque([source])
        while queue:
            a = queue.popleft()
            if a == target:
                return distance[a]
            for b in adjacency[a]:
                if b not in distance:
                    distance[b] = distance[a] + 1
                    queue.append(b)
        raise RuntimeError("generated maze is disconnected")

    # The tree diameter gives the most useful episode available in a layout.
    # Prefer the proposal's 20–35 m range; retain the closest generated maze if
    # a very small custom grid cannot meet it.
    best = None
    best_error = np.inf
    for _ in range(max(attempts, 1)):
        adjacency = carve()
        a, _, _ = farthest(adjacency, 0)
        b, edges, parent = farthest(adjacency, a)
        route_length = edges * cell
        error = max(min_route - route_length, 0.0, route_length - max_route)
        if error < best_error:
            best = adjacency, a, b, edges, parent
            best_error = error
        if error == 0:
            break

    adjacency, start, goal_cell, route_edges, _ = best
    if sum((random_start, random_goal, random_endpoints)) > 1:
        raise ValueError(
            "random_start, random_goal, and random_endpoints are mutually exclusive")
    if random_start:
        candidates = []
        for candidate in range(n_cells):
            if candidate == goal_cell:
                continue
            edges = tree_distance(adjacency, candidate, goal_cell)
            metres = edges * cell
            if endpoint_min_route <= metres <= endpoint_max_route:
                candidates.append((candidate, edges))
        if not candidates:
            raise ValueError(
                "no random starts satisfy the configured endpoint route range")
        start, route_edges = candidates[int(rng.integers(len(candidates)))]
    if random_goal:
        candidates = []
        for candidate in range(n_cells):
            if candidate == start:
                continue
            edges = tree_distance(adjacency, start, candidate)
            metres = edges * cell
            if endpoint_min_route <= metres <= endpoint_max_route:
                candidates.append((candidate, edges))
        if not candidates:
            raise ValueError(
                "no random goals satisfy the configured endpoint route range")
        goal_cell, route_edges = candidates[int(rng.integers(len(candidates)))]
    if random_endpoints:
        candidates = []
        for a in range(n_cells):
            for b in range(a + 1, n_cells):
                edges = tree_distance(adjacency, a, b)
                metres = edges * cell
                if endpoint_min_route <= metres <= endpoint_max_route:
                    candidates.append((a, b, edges))
        if not candidates:
            raise ValueError("no start/goal pairs satisfy the configured "
                             "endpoint route range")
        start, goal_cell, route_edges = candidates[int(rng.integers(len(candidates)))]
    if not (random_start or random_goal) and rng.random() < 0.5:
        start, goal_cell = goal_cell, start

    # Open edges are passages. Every other grid edge is one wall piece.
    passages = {frozenset((a, b)) for a, ns in enumerate(adjacency) for b in ns}
    x0, y0, x1, y1 = bounds
    walls = []
    for x in range(cols):
        xa, xb = (x - 0.5) * cell, (x + 0.5) * cell
        walls.extend(((xa, y0, xb, y0), (xa, y1, xb, y1)))
    for y in range(rows):
        ya, yb = (y - 0.5) * cell, (y + 0.5) * cell
        walls.extend(((x0, ya, x0, yb), (x1, ya, x1, yb)))
    for y in range(rows):
        for x in range(cols - 1):
            a, b = y * cols + x, y * cols + x + 1
            if frozenset((a, b)) not in passages:
                wx = (x + 0.5) * cell
                walls.append((wx, (y - 0.5) * cell, wx, (y + 0.5) * cell))
    for y in range(rows - 1):
        for x in range(cols):
            a, b = y * cols + x, (y + 1) * cols + x
            if frozenset((a, b)) not in passages:
                wy = (y + 0.5) * cell
                walls.append(((x - 0.5) * cell, wy, (x + 0.5) * cell, wy))

    def centre(k):
        return np.array([(k % cols) * cell, (k // cols) * cell], dtype=np.float32)

    spawn, goal = centre(start), centre(goal_cell)

    # Reconstruct true corridor shortest path connecting start to goal around walls
    parent_map = {start: None}
    queue = deque([start])
    while queue:
        curr = queue.popleft()
        if curr == goal_cell:
            break
        for nbr in adjacency[curr]:
            if nbr not in parent_map:
                parent_map[nbr] = curr
                queue.append(nbr)

    curr = goal_cell
    cell_path = []
    while curr is not None:
        cell_path.append(curr)
        curr = parent_map.get(curr)
    cell_path.reverse()

    waypoint_centres = np.array([centre(k) for k in cell_path], dtype=np.float32)
    dense_pts = []
    for i in range(len(waypoint_centres) - 1):
        p0, p1 = waypoint_centres[i], waypoint_centres[i + 1]
        n_seg = max(int(np.linalg.norm(p1 - p0) / 0.04), 2)
        seg = np.linspace(p0, p1, n_seg, endpoint=(i == len(waypoint_centres) - 2))
        dense_pts.append(seg)
    pts = np.concatenate(dense_pts, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Ground Hazards & Obstacle Blockers in Corridor Cells (if enabled)
    # ------------------------------------------------------------------
    gaps, sand_patches, stones, steps, blockers = [], [], [], [], []
    hazards_cfg = getattr(cfg.scenario, "hazards", None)
    if hazards_cfg is not None and getattr(hazards_cfg, "enabled", False):
        n_gaps = int(getattr(hazards_cfg, "n_gaps", 3))
        n_sand = int(getattr(hazards_cfg, "n_sand", 3))
        n_stones = int(getattr(hazards_cfg, "n_stones", 4))
        n_steps = int(getattr(hazards_cfg, "n_steps", 2))
        n_blockers = int(getattr(hazards_cfg, "n_blockers", 0))

        rocky_corridors = bool(getattr(hazards_cfg, "rocky_corridors", True))
        n_rocks_per_cell = int(getattr(hazards_cfg, "n_rocks_per_cell", 18))
        max_rock_sz = float(getattr(hazards_cfg, "max_rock_size", 0.055))

        # 1. Fill entire maze corridor network with dense mountainous stone boulders
        if rocky_corridors:
            for k in range(n_cells):
                # Don't put massive rocks directly on spawn cell center
                if k == start:
                    c = centre(k)
                    stones.append([c[0], c[1], cell * 0.46, cell * 0.46, 8, max_rock_sz * 0.65])
                else:
                    c = centre(k)
                    stones.append([c[0], c[1], cell * 0.48, cell * 0.48, n_rocks_per_cell, max_rock_sz])

        # 2. Choose intermediate corridor cells for blockers, gaps, steps, and sand
        path_cells = cell_path[2:-2] if len(cell_path) > 6 else cell_path[1:-1]
        all_cells = list(range(n_cells))
        rng.shuffle(all_cells)

        ptr = 0
        # Place gaps
        for _ in range(n_gaps):
            if ptr >= len(all_cells):
                break
            c = centre(all_cells[ptr])
            gaps.append([c[0], c[1], 0.12, 0.45, 0.06])
            ptr += 1

        # Place sand patches
        for _ in range(n_sand):
            if ptr >= len(all_cells):
                break
            c = centre(all_cells[ptr])
            sand_patches.append([c[0], c[1], 0.50, 0.50])
            ptr += 1

        # Place step curbs
        for _ in range(n_steps):
            if ptr >= len(all_cells):
                break
            c = centre(all_cells[ptr])
            steps.append([c[0], c[1], 0.10, 0.55, 0.035])
            ptr += 1

        # Place corridor bollards / blockers (offset from cell centre)
        for _ in range(n_blockers):
            if ptr >= len(all_cells):
                break
            c = centre(all_cells[ptr])
            off_x = float(rng.choice([-0.25, 0.25]))
            off_y = float(rng.choice([-0.25, 0.25]))
            blockers.append([c[0] + off_x, c[1] + off_y, 0.18])
            ptr += 1

    field, origin, res = _geodesic_field(walls, bounds, goal)
    return Scenario(
        kind="maze", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=np.empty((0, 2), dtype=np.float32),
        path_length=float(route_edges * cell),
        walls=np.asarray(walls, dtype=np.float32),
        obstacles=np.asarray(blockers, dtype=np.float32).reshape(-1, 3) if blockers else None,
        gaps=gaps,
        sand_patches=sand_patches,
        stones=stones,
        steps=steps,
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


def rocky_terrain_scenario(cfg, *, rng=None, name: str = "rocky_terrain") -> Scenario:
    """Rugged Mountainous Rocky Floor with dense boulders, jagged stone slabs, and crags."""
    rng = rng if rng is not None else np.random.default_rng()
    sc = getattr(cfg, "scenario", None)
    goal_cfg = getattr(sc, "goal", None) if sc is not None else None
    x_range = tuple(getattr(goal_cfg, "x_range", (25.0, 25.0))) if goal_cfg else (25.0, 25.0)
    y_range = tuple(getattr(goal_cfg, "y_range", (0.0, 0.0))) if goal_cfg else (0.0, 0.0)

    spawn = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([rng.uniform(*x_range), rng.uniform(*y_range)], dtype=np.float32)
    pts = np.linspace(spawn, goal, 200).astype(np.float32)
    markers = np.empty((0, 2), dtype=np.float32)

    rock_cfg = getattr(sc, "rocky_terrain", None) if sc is not None else None
    n_rocks = int(getattr(rock_cfg, "n_rocks", 850)) if rock_cfg else 850
    max_sz = float(getattr(rock_cfg, "max_rock_size", 0.065)) if rock_cfg else 0.065

    # Dense rocky corridor spanning between spawn and goal
    mid = 0.5 * (spawn + goal)
    half_span_x = float(np.abs(goal[0] - spawn[0]) * 0.52)
    stones = [
        [float(mid[0]), float(mid[1]), half_span_x, 1.8, n_rocks, max_sz]
    ]

    return Scenario(
        kind="rocky_terrain", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=float(np.linalg.norm(goal - spawn)),
        stones=stones,
    )


def slopes_scenario(cfg, *, rng=None, name: str = "slopes") -> Scenario:
    """Uphill ramp climb (+15 deg incline), elevated ridge plateau, and downhill descent."""
    rng = rng if rng is not None else np.random.default_rng()
    spawn = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([17.5, 0.0], dtype=np.float32)
    pts = np.linspace(spawn, goal, 180).astype(np.float32)
    markers = pts[::8].copy()

    # Ramps list: [cx, cy, length, width, h_change, pitch_deg, yaw]
    # Ramp 1: Uphill from x=2.5 to 6.5 (rise +0.50m, pitch 7.2 deg)
    # Plateau: Elevated ridge from x=6.5 to 11.5 at z=0.50m
    # Ramp 2: Downhill from x=11.5 to 15.5 (descent -0.50m, pitch -7.2 deg)
    ramps = [
        [4.5, 0.0, 4.0, 1.8, 0.50, 7.2, 0.0],
        [9.0, 0.0, 5.0, 1.8, 0.50, 0.0, 0.0],
        [13.5, 0.0, 4.0, 1.8, -0.50, -7.2, 0.0],
    ]

    return Scenario(
        kind="slopes", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=17.5,
        ramps=ramps,
    )


def stairs_scenario(cfg, *, rng=None, name: str = "stairs") -> Scenario:
    """Multi-step ascending flight of stairs, elevated landing, and descending stairs."""
    rng = rng if rng is not None else np.random.default_rng()
    spawn = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([13.5, 0.0], dtype=np.float32)
    pts = np.linspace(spawn, goal, 140).astype(np.float32)
    markers = pts[::8].copy()

    # Staircases list: [start_x, start_y, n_steps, rise, run, width, yaw, is_down]
    # Ascending 5 steps: x=2.5 to 3.8 (top z=0.25m)
    # Descending 5 steps: x=6.8 to 8.1 (back down to floor)
    staircases = [
        [2.5, 0.0, 5, 0.05, 0.26, 1.6, 0.0, False],
        [6.8, 0.0, 5, 0.05, 0.26, 1.6, 0.0, True],
    ]
    # Connecting solid elevated landing plateau at z=0.25m (x=3.8 to 6.8, center=5.3, length=3.0)
    ramps = [
        [5.3, 0.0, 3.0, 1.6, 0.25, 0.0, 0.0],
    ]

    return Scenario(
        kind="stairs", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=13.5,
        staircases=staircases,
        ramps=ramps,
    )


def glass_pipe_scenario(cfg, *, rng=None, name: str = "glass_pipe") -> Scenario:
    """Transparent glass cylindrical conduit / pipe with in-pipe crawling and 360-deg rod bracing."""
    rng = rng if rng is not None else np.random.default_rng()
    # Spawn directly INSIDE the transparent glass pipe at x=2.0
    spawn = np.array([2.0, 0.0], dtype=np.float32)
    goal = np.array([12.0, 0.0], dtype=np.float32)
    pts = np.linspace(spawn, goal, 120).astype(np.float32)
    markers = pts[::8].copy()

    # Pipes list: [start_x, start_y, length, in_rad, out_rad, yaw]
    # 10.5-meter long transparent glass pipe spanning x=1.0 to 11.5
    pipes = [
        [1.0, 0.0, 10.5, 0.38, 0.395, 0.0],
    ]

    return Scenario(
        kind="glass_pipe", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=10.0,
        pipes=pipes,
    )


def extreme_gauntlet_scenario(cfg, *, rng=None, name: str = "extreme_gauntlet") -> Scenario:
    """Integrated extreme multi-stage course: Uphill Incline -> Rocky Ridge -> Stairs -> Downhill -> Glass Pipe."""
    rng = rng if rng is not None else np.random.default_rng()
    spawn = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([27.0, 0.0], dtype=np.float32)
    pts = np.linspace(spawn, goal, 270).astype(np.float32)
    markers = pts[::10].copy()

    # 1. Ramps: Uphill ramp (x=2.0 to 6.0, rise +0.60m) and Downhill ramp (x=14.0 to 17.5, descent -0.60m)
    ramps = [
        [4.0, 0.0, 4.0, 1.8, 0.60, 8.6, 0.0],    # Uphill
        [8.0, 0.0, 4.0, 1.8, 0.0, 0.0, 0.0],     # Elevated Rocky Plateau (z=0.60m)
        [15.75, 0.0, 3.5, 1.8, -0.60, -9.8, 0.0], # Downhill
    ]

    # 2. Stones on elevated plateau: 120 procedural boulders
    stones = [
        [8.0, 0.0, 1.8, 0.8, 120, 0.055],
    ]

    # 3. Staircase: 5 descending steps off plateau (x=10.0 to 11.3, step height 0.04m, landing)
    staircases = [
        [10.0, 0.0, 5, 0.04, 0.26, 1.6, 0.0, True],
    ]

    # 4. Transparent Glass Pipe: x=18.5 to 25.0 (length 6.5m)
    pipes = [
        [18.5, 0.0, 6.5, 0.38, 0.395, 0.0],
    ]

    return Scenario(
        kind="extreme_gauntlet", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=27.0,
        ramps=ramps,
        staircases=staircases,
        stones=stones,
        pipes=pipes,
    )


def jump_track_scenario(cfg, *, rng=None, name="jump_track") -> Scenario:
    """Long Jump & Hurdle Track: High-contrast athletic runway with distance markers, hurdles, and landing zone."""
    spawn = np.array([0.0, 0.0], dtype=float)
    goal = np.array([4.8, 0.0], dtype=float)
    pts = np.linspace(spawn, goal, 15)
    markers = pts.copy()

    # Lateral perimeter guide rails
    walls = np.array([
        [-0.8, 1.20, 5.2, 1.20],
        [-0.8, -1.20, 5.2, -1.20],
    ])

    # Visual Yardlines and Athletic Runway Markings (Zero-drag visual geoms)
    yardlines = [
        # Dark Athletic Runway Base (x = -0.5 to 5.2m)
        [2.35, 0.0, 2.85, 1.15, "0.16 0.18 0.22 1.0"],
        # Orange Takeoff Launch Pad (x = 0.0 to 0.4m)
        [0.20, 0.0, 0.20, 1.10, "0.95 0.42 0.10 1.0"],
        # Painted Distance Stripes every 0.50m
        [0.50, 0.0, 0.025, 1.05, "0.95 0.95 0.95 1.0"],
        [1.00, 0.0, 0.035, 1.05, "0.96 0.78 0.08 1.0"],
        [1.50, 0.0, 0.025, 1.05, "0.95 0.95 0.95 1.0"],
        [2.00, 0.0, 0.035, 1.05, "0.96 0.78 0.08 1.0"],
        # Vivid Green Landing Zone Pad (x = 2.1 to 2.7m)
        [2.40, 0.0, 0.30, 1.05, "0.12 0.82 0.38 1.0"],
        [2.50, 0.0, 0.025, 1.05, "0.95 0.95 0.95 1.0"],
        [3.00, 0.0, 0.035, 1.05, "0.96 0.78 0.08 1.0"],
        [3.50, 0.0, 0.025, 1.05, "0.95 0.95 0.95 1.0"],
        [4.00, 0.0, 0.035, 1.05, "0.96 0.78 0.08 1.0"],
    ]

    # Physical Hurdles to leap over in mid-air
    steps = [
        # Hurdle 1 at x=1.45m (height 8.0cm, width 1.6m)
        [1.45, 0.0, 0.035, 0.80, 0.080],
        # Hurdle 2 at x=3.25m (height 9.0cm, width 1.6m)
        [3.25, 0.0, 0.035, 0.80, 0.090],
    ]

    return Scenario(
        kind="jump_track", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=4.8,
        walls=walls,
        steps=steps,
        yardlines=yardlines,
    )


# ---------------------------------------------------------------------------
# Circle Track — circular arena with painted lane dashes and spokes
# ---------------------------------------------------------------------------

def circle_track_scenario(cfg, *, rng=None, name: str = "circle_track", radius: float = 1.8) -> Scenario:
    """Circular athletic arena with painted circular lane markings, radial spokes,
    and a center hub for demonstrating circular trajectory tracking."""
    spawn = np.array([radius, 0.0], dtype=float)
    goal = np.array([0.0, 0.0], dtype=float)

    # 128 discretized points forming the circular reference path

    thetas = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    pts = np.stack([radius * np.cos(thetas), radius * np.sin(thetas)], axis=1)
    markers = pts[::8].copy()

    # Painted ground markings (zero collision)
    yardlines = []
    # Center hub
    yardlines.append([0.0, 0.0, 0.15, 0.15, "0.96 0.78 0.08 1.0"])

    # 64 dashed segments along the circular track
    n_seg = 64
    seg_len = (2 * np.pi * radius) / n_seg * 0.45
    for k in range(n_seg):
        th = 2 * np.pi * k / n_seg
        xk = radius * np.cos(th)
        yk = radius * np.sin(th)
        # Alternate colors: vivid gold and bright white
        color = "0.96 0.78 0.08 0.95" if k % 4 == 0 else "0.95 0.95 0.95 0.8"
        yardlines.append([xk, yk, 0.035, seg_len, color])

    # 4 radial spoke yardlines (North, South, East, West)
    yardlines.append([radius, 0.0, 0.25, 0.04, "0.95 0.42 0.10 1.0"])   # Start / Finish line (East)
    yardlines.append([0.0, radius, 0.04, 0.25, "0.12 0.82 0.38 1.0"])   # Quarter mark (North)
    yardlines.append([-radius, 0.0, 0.25, 0.04, "0.95 0.42 0.10 1.0"])  # Halfway mark (West)
    yardlines.append([0.0, -radius, 0.04, 0.25, "0.12 0.82 0.38 1.0"])  # 3/4 mark (South)

    # 4 axis-aligned perimeter boundary walls around the 7.6m x 7.6m arena
    walls = [
        [-3.8, -3.8, 3.8, -3.8],
        [3.8, -3.8, 3.8, 3.8],
        [3.8, 3.8, -3.8, 3.8],
        [-3.8, 3.8, -3.8, -3.8],
    ]

    return Scenario(
        kind="circle_track", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=2 * np.pi * radius,
        walls=np.array(walls),
        yardlines=yardlines,
    )


# ---------------------------------------------------------------------------
# Gap Bridge — Dual parallel ledges (Box 1 & Box 2) spanning an open hole
# ---------------------------------------------------------------------------

def gap_bridge_scenario(cfg, *, rng=None, name: str = "gap_bridge", gap_width: float = 0.22, box_height: float = 0.25) -> Scenario:
    """Parallel dual-box arena with a deep central chasm/hole underneath.

    Box 1 (Left) and Box 2 (Right) run parallel along the X axis, starting at x = -0.15m
    (with the ball at x = 0.0m at the very beginning of the track) and extending to x = 5.6m.
    The gap between them is `gap_width` (default 0.22m).
    Directly underneath the center is an open drop to the floor.
    """
    half_gap = gap_width / 2.0
    box_w = 0.60
    box_y_left = half_gap + box_w / 2.0    # +0.41m
    box_y_right = -half_gap - box_w / 2.0  # -0.41m
    x_start = -0.15
    x_end = 5.60
    box_len = x_end - x_start
    cx = (x_start + x_end) / 2.0           # 2.725m
    hx = box_len / 2.0                     # 2.875m

    steps = [
        # Box 1 (Left ledge): pos_x, pos_y, hx, hy, height
        [cx, box_y_left, hx, box_w / 2.0, box_height],
        # Box 2 (Right ledge)
        [cx, box_y_right, hx, box_w / 2.0, box_height],
    ]

    spawn = np.array([0.0, 0.0], dtype=float)
    goal = np.array([5.0, 0.0], dtype=float)

    # Path points along the gap centerline
    pts = np.array([
        [0.0, 0.0],
        [1.5, 0.0],
        [3.0, 0.0],
        [5.0, 0.0],
    ], dtype=float)
    markers = pts.copy()

    # Visual yardlines and distance stripes along both Box 1 and Box 2
    yardlines = [
        # Inner lip warning stripe on Box 1 (y = +half_gap)
        [cx, half_gap + 0.015, hx, 0.012, "0.96 0.78 0.08 1.0"],
        # Inner lip warning stripe on Box 2 (y = -half_gap)
        [cx, -half_gap - 0.015, hx, 0.012, "0.96 0.78 0.08 1.0"],
        # Start Line at the beginning of the track (x = 0.0m)
        [0.0, 0.0, 0.035, half_gap + box_w, "0.95 0.42 0.10 1.0"],
        # Finish Line at the end of the track (x = 5.0m)
        [5.0, 0.0, 0.045, half_gap + box_w, "0.12 0.82 0.38 1.0"],
    ]

    # Distance stripes every 0.50m on both Box 1 and Box 2
    for dist in np.arange(0.5, 5.0, 0.5):
        color = "0.96 0.78 0.08 0.9" if int(round(dist * 10)) % 10 == 0 else "0.95 0.95 0.95 0.75"
        # Stripe on Box 1
        yardlines.append([dist, box_y_left, 0.02, box_w / 2.0 * 0.9, color])
        # Stripe on Box 2
        yardlines.append([dist, box_y_right, 0.02, box_w / 2.0 * 0.9, color])

    walls = np.array([
        [-0.8, -1.8, 6.2, -1.8],
        [6.2, -1.8, 6.2, 1.8],
        [6.2, 1.8, -0.8, 1.8],
        [-0.8, 1.8, -0.8, -1.8],
    ])

    return Scenario(
        kind="gap_bridge", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=5.0,
        walls=walls,
        steps=steps,
        yardlines=yardlines,
    )


# ---------------------------------------------------------------------------
# Chimney — Two tall vertical boxes forming a vertical shaft/fissure
# ---------------------------------------------------------------------------

def chimney_scenario(cfg, *, rng=None, name: str = "chimney", shaft_width: float = 0.40, shaft_height: float = 4.0, shaft_length: float = 1.2) -> Scenario:
    """Vertical chimney arena with two opposing vertical box walls.

    Box 1 (Left) and Box 2 (Right) stand vertically with an open shaft of width `shaft_width`
    (default 0.40m) between them. The ball climbs up and down by pressing outward against both faces.
    """
    half_gap = shaft_width / 2.0
    # Box (wall) thickness. The climb exits over the lip with about 1.5 m/s of
    # sideways carry and lands 0.6-0.9 m out; a 0.5 m top was narrower than
    # the landing scatter, so the default is a full-width wall top.
    ch = getattr(getattr(cfg, "scenario", None), "chimney", None)
    box_w = float(getattr(ch, "box_width", 1.30))
    # The two walls are NOT the same height. A wall push lifts the ball at
    # most ~0.17 m above the push point, so it can never clear a lip level
    # with the wall it just pushed off. It CAN clear the lip of a lower wall
    # opposite: push off the tall wall from just above the low lip, fly over,
    # land on the low box top. Real chimneys are exited the same way.
    low_h = float(getattr(ch, "low_box_height", shaft_height - 0.7))
    box_y_left = half_gap + box_w / 2.0    # +0.45m
    box_y_right = -half_gap - box_w / 2.0  # -0.45m


    steps = [
        # Box 1 (Left vertical box): pos_x, pos_y, hx, hy, height
        [0.0, box_y_left, shaft_length / 2.0, box_w / 2.0, low_h],
        # Box 2 (Right vertical box)
        [0.0, box_y_right, shaft_length / 2.0, box_w / 2.0, shaft_height],
    ]

    spawn = np.array([0.0, 0.0], dtype=float)
    goal = np.array([0.0, 10.0], dtype=float)


    pts = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
    ], dtype=float)
    markers = pts.copy()

    # Visual height markers along the vertical box faces
    yardlines = []
    for h in np.arange(0.5, shaft_height, 0.5):
        color = "0.96 0.78 0.08 1.0" if int(round(h * 10)) % 10 == 0 else "0.95 0.95 0.95 0.8"
        # Stripe on left box face
        yardlines.append([0.0, half_gap + 0.01, shaft_length / 2.0 * 0.9, 0.02, color])
        # Stripe on right box face
        yardlines.append([0.0, -half_gap - 0.01, shaft_length / 2.0 * 0.9, 0.02, color])

    walls = np.array([
        [-shaft_length / 2.0 - 0.2, -1.2, shaft_length / 2.0 + 0.2, -1.2],
        [shaft_length / 2.0 + 0.2, -1.2, shaft_length / 2.0 + 0.2, 1.2],
        [shaft_length / 2.0 + 0.2, 1.2, -shaft_length / 2.0 - 0.2, 1.2],
        [-shaft_length / 2.0 - 0.2, 1.2, -shaft_length / 2.0 - 0.2, -1.2],
    ])

    return Scenario(
        kind="chimney", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=shaft_height,
        walls=walls,
        steps=steps,
        yardlines=yardlines,
    )




# ---------------------------------------------------------------------------
# Skill course — the hand-drawn corridor circuit used to exercise the skills
# ---------------------------------------------------------------------------



# Grid layout, transcribed from the sketch. Row 0 is the TOP corridor and row
# index grows downward; column 0 is the far left. Each entry opens a run of
# cells: ("row", r, c_from, c_to) or ("col", c, r_from, r_to), inclusive.
_SKILL_COURSE_RUNS = [
    ("row", 2, 0, 5),     # start corridor, heading right
    ("col", 5, 0, 2),     # climb to the top corridor
    ("row", 0, 5, 15),    # long top corridor
    ("col", 15, 0, 6),    # right-hand descent
    ("row", 2, 15, 19),   # dead-end spur: drive in, reverse out
    ("row", 6, 5, 15),    # bottom corridor, heading left
    ("col", 5, 6, 9),     # drop toward the goal
    ("row", 9, 3, 5),     # goal corridor
]

# Centreline of the route as (col, row) cell coordinates, including the
# there-and-back through the spur.
_SKILL_COURSE_ROUTE = [
    (0, 2), (5, 2), (5, 0), (15, 0), (15, 2),
    (19, 2), (15, 2),                          # spur out and back
    (15, 6), (5, 6), (5, 9), (3, 9),
]


def _skill_course_cells():
    """Set of open (col, row) cells for the skill course."""
    cells = set()
    for axis, fixed, lo, hi in _SKILL_COURSE_RUNS:
        for k in range(min(lo, hi), max(lo, hi) + 1):
            cells.add((fixed, k) if axis == "col" else (k, fixed))
    return cells


def skill_course_scenario(cfg, *, rng=None, name: str = "skill_course") -> Scenario:
    """Hand-drawn corridor circuit for demonstrating the skill library.

    The route runs start -> right -> up -> long top straight -> down -> into a
    dead-end spur and back out in reverse -> bottom corridor -> down -> goal.
    Walls are placed on every face of an open cell whose neighbour is closed,
    which handles the spur's T-junction without any special casing.
    """
    sc = getattr(cfg, "scenario", None)
    course_cfg = getattr(sc, "skill_course", None) if sc is not None else None
    cell = float(getattr(course_cfg, "cell", 1.8))

    cells = _skill_course_cells()

    def centre(col, row):
        return np.array([col * cell, -row * cell], dtype=float)

    # One wall segment per open face that borders a closed cell.
    half = cell / 2.0
    walls = []
    for (col, row) in sorted(cells):
        cx, cy = centre(col, row)
        # (neighbour offset, wall endpoints relative to the cell centre)
        faces = [
            ((0, -1), (-half, +half, +half, +half)),   # north (row - 1)
            ((0, +1), (-half, -half, +half, -half)),   # south (row + 1)
            ((-1, 0), (-half, -half, -half, +half)),   # west  (col - 1)
            ((+1, 0), (+half, -half, +half, +half)),   # east  (col + 1)
        ]
        for (dc, dr), (ax, ay, bx, by) in faces:
            if (col + dc, row + dr) in cells:
                continue
            walls.append([cx + ax, cy + ay, cx + bx, cy + by])

    route = [centre(c, r) for c, r in _SKILL_COURSE_ROUTE]
    spawn = route[0].copy()
    goal = route[-1].copy()

    # A solid box blocking the middle of the bottom corridor. The agent has to
    # clear it with a running jump; there is no way around it, since the box
    # spans the corridor.
    box_h = float(getattr(course_cfg, "hurdle_height", 0.22))
    box_depth = float(getattr(course_cfg, "hurdle_depth", 0.50))
    bx, by = 0.5 * (route[_SKILL_COURSE_HURDLE_LEG[0]] + route[_SKILL_COURSE_HURDLE_LEG[1]])
    hurdle = [float(bx), float(by), box_depth / 2.0, cell / 2.0 - 0.05, box_h]

    # A second, deeper box on the start corridor. This one is a platform: the
    # agent jumps up ONTO it, steadies itself, then drops off the far edge to
    # carry on. It is deep enough to land on and stand still.
    plat_h = float(getattr(course_cfg, "platform_height", 0.25))
    plat_depth = float(getattr(course_cfg, "platform_depth", 1.20))
    a, b = route[_SKILL_COURSE_PLATFORM_LEG[0]], route[_SKILL_COURSE_PLATFORM_LEG[1]]
    px, py = a + _SKILL_COURSE_PLATFORM_FRAC * (b - a)
    platform = [float(px), float(py), plat_depth / 2.0, cell / 2.0 - 0.05, plat_h]

    # Dense waypoints so the look-ahead controller has something to track.
    pts = [route[0]]
    for a, b in zip(route[:-1], route[1:]):
        n = max(2, int(np.linalg.norm(b - a) / 0.30))
        for t in np.linspace(0.0, 1.0, n)[1:]:
            pts.append(a + t * (b - a))
    pts = np.asarray(pts, dtype=np.float32)

    return Scenario(
        kind="skill_course", name=name,
        spawn_xy=spawn.astype(np.float32),
        goal=goal.astype(np.float32),
        path_pts=pts,
        markers=np.asarray(route, dtype=np.float32),
        path_length=float(_arc_length(pts)),
        walls=np.asarray(walls, dtype=np.float32),
        steps=[platform, hurdle],
    )


# Route indices of the leg whose midpoint carries the jump-over box.
_SKILL_COURSE_HURDLE_LEG = (7, 8)

# Leg and position along it for the jump-onto platform.
_SKILL_COURSE_PLATFORM_LEG = (0, 1)
_SKILL_COURSE_PLATFORM_FRAC = 0.55


def skill_course_platform(cfg) -> dict:
    """Centre, half-extents and height of the platform the agent jumps onto."""
    sc = getattr(cfg, "scenario", None)
    course_cfg = getattr(sc, "skill_course", None) if sc is not None else None
    cell = float(getattr(course_cfg, "cell", 1.8))
    route = skill_course_route(cfg)
    a, b = route[_SKILL_COURSE_PLATFORM_LEG[0]], route[_SKILL_COURSE_PLATFORM_LEG[1]]
    depth = float(getattr(course_cfg, "platform_depth", 1.20))
    return {
        "xy": np.asarray(a + _SKILL_COURSE_PLATFORM_FRAC * (b - a), dtype=np.float32),
        "half_depth": depth / 2.0,
        "half_width": cell / 2.0 - 0.05,
        "height": float(getattr(course_cfg, "platform_height", 0.25)),
    }


def skill_course_hurdle(cfg) -> dict:
    """Centre, half-extents and height of the box blocking the bottom corridor."""
    sc = getattr(cfg, "scenario", None)
    course_cfg = getattr(sc, "skill_course", None) if sc is not None else None
    cell = float(getattr(course_cfg, "cell", 1.8))
    route = skill_course_route(cfg)
    a, b = route[_SKILL_COURSE_HURDLE_LEG[0]], route[_SKILL_COURSE_HURDLE_LEG[1]]
    centre_xy = 0.5 * (a + b)
    depth = float(getattr(course_cfg, "hurdle_depth", 0.50))
    return {
        "xy": np.asarray(centre_xy, dtype=np.float32),
        "half_depth": depth / 2.0,
        "half_width": cell / 2.0 - 0.05,
        "height": float(getattr(course_cfg, "hurdle_height", 0.22)),
    }


def skill_course_route(cfg) -> np.ndarray:
    """The course centreline as world xy, in travel order (spur included)."""
    sc = getattr(cfg, "scenario", None)
    course_cfg = getattr(sc, "skill_course", None) if sc is not None else None
    cell = float(getattr(course_cfg, "cell", 1.8))
    return np.array([[c * cell, -r * cell] for c, r in _SKILL_COURSE_ROUTE],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Platform course — a straight run of boxes, jumped over, onto and between
# ---------------------------------------------------------------------------

# Each entry is (x centre, height, depth, role). Roles:
#   "over"  — a low box on the floor, to be cleared
#   "onto"  — a deck to land on, reached from the floor or from the box before
# Heights and gaps are inside what skills/jump_planner.py can guarantee: the
# leap only carries about 0.25 m horizontally at its take-off height, and about
# 0.5 m if it is dropping 0.15 m, so ascending steps are kept adjacent and only
# descending steps get a real gap.
# Decks are long on purpose. The take-off calibration assumes the ball has
# reached cruise, and after landing it needs most of a deck to build that up
# again before the next hop. Short decks leave it launching from a standstill.
_PLATFORM_BOXES = [
    (3.20, 0.12, 0.20, "over"),    # clear it from the floor
    (6.50, 0.16, 2.60, "onto"),    # climb up from the floor
    (9.15, 0.32, 2.60, "onto"),    # step up, decks all but touching
    (11.75, 0.32, 2.20, "onto"),   # level hop across a 0.20 m gap
    (14.45, 0.16, 2.20, "onto"),   # drop-down hop across a 0.50 m gap
]

_PLATFORM_LENGTH = 19.0            # corridor length (m)


def platform_course_scenario(cfg, *, rng=None, name: str = "platform_course") -> Scenario:
    """A straight corridor studded with boxes, in the manner of a platform game.

    The robot clears one box from the floor, climbs onto a deck, steps up to a
    taller one, hops the gap to a third, drops across to a lower fourth, then
    falls back to the floor and runs to the goal.
    """
    sc = getattr(cfg, "scenario", None)
    course_cfg = getattr(sc, "platform_course", None) if sc is not None else None
    width = float(getattr(course_cfg, "width", 1.8))
    length = float(getattr(course_cfg, "length", _PLATFORM_LENGTH))

    half_w = width / 2.0
    spawn = np.array([0.0, 0.0], dtype=float)
    goal = np.array([length - 0.9, 0.0], dtype=float)

    # Side walls plus end caps: a single straight corridor.
    walls = np.array([
        [-0.9, +half_w, length, +half_w],
        [-0.9, -half_w, length, -half_w],
        [-0.9, -half_w, -0.9, +half_w],
        [length, -half_w, length, +half_w],
    ], dtype=float)

    steps = [[float(x), 0.0, float(d) / 2.0, half_w - 0.05, float(h)]
             for (x, h, d, _role) in _PLATFORM_BOXES]

    pts = np.linspace(spawn, goal, 120).astype(np.float32)
    return Scenario(
        kind="platform_course", name=name,
        spawn_xy=spawn.astype(np.float32),
        goal=goal.astype(np.float32),
        path_pts=pts,
        markers=np.empty((0, 2), dtype=np.float32),
        path_length=float(_arc_length(pts)),
        walls=walls.astype(np.float32),
        steps=steps,
    )


def platform_course_boxes(cfg) -> list[dict]:
    """The boxes as dicts, in travel order, with their edges worked out."""
    sc = getattr(cfg, "scenario", None)
    course_cfg = getattr(sc, "platform_course", None) if sc is not None else None
    width = float(getattr(course_cfg, "width", 1.8))
    out = []
    for i, (x, h, d, role) in enumerate(_PLATFORM_BOXES):
        out.append({
            "index": i,
            "role": role,
            "x": float(x),
            "height": float(h),
            "half_depth": float(d) / 2.0,
            "half_width": width / 2.0 - 0.05,
            "near": float(x) - float(d) / 2.0,
            "far": float(x) + float(d) / 2.0,
        })
    return out


# ---------------------------------------------------------------------------
# Pillar course — tall narrow columns, hopped between from a standstill
# ---------------------------------------------------------------------------

# (x centre, height, top size). The pads are square and small: the ball is
# 0.30 m across the core, so a 0.60 m pad is somewhere to stand, not somewhere
# to roll. Every hop is therefore a standing jump.
#
# Heights are 2.8x to 3.5x the core diameter. Reaching them needs the
# long-stroke build in configs/rl/pillar_course.yaml; the standard build tops
# out at a 0.42 m pad.
# Pads are 0.90 m square. That is still too small to roll on -- the core is
# 0.30 m across, so there is barely one ball-width of deck -- but 0.60 m pads
# proved smaller than the jump's own landing scatter. A pad must be wider than
# the spread of where the jump actually lands.
#
# The course is a LADDER. The guaranteed standing-jump rise, measured across
# random orientations, floors at about 0.84 m -- so a single leap from the
# floor onto the 0.85 m pillar cannot be promised. Each step here rises at
# most 0.45 m, which every calibrated orientation clears; the tall pillars
# are reached by climbing, the way a platformer would stage it.
_PILLAR_COLUMNS = [
    (2.20, 0.40, 0.90),    # starter block, 1.3x the core
    (3.22, 0.85, 0.90),    # +0.45 -> 2.8x the core
    (4.24, 1.05, 0.90),    # +0.20 -> 3.5x the core
    (5.26, 0.65, 0.90),    # -0.40 -> down to 2.2x
]

_PILLAR_LENGTH = 9.0


def pillar_course_scenario(cfg, *, rng=None, name: str = "pillar_course") -> Scenario:
    """A corridor with three tall, narrow columns to hop between.

    The robot drives up, stops, jumps onto the first column, stands still,
    then hops column to column without ever rolling on top of one. The pads
    are too small for that -- which is the point.
    """
    sc = getattr(cfg, "scenario", None)
    course_cfg = getattr(sc, "pillar_course", None) if sc is not None else None
    width = float(getattr(course_cfg, "width", 2.2))
    length = float(getattr(course_cfg, "length", _PILLAR_LENGTH))

    half_w = width / 2.0
    spawn = np.array([0.0, 0.0], dtype=float)
    goal = np.array([length - 0.9, 0.0], dtype=float)

    walls = np.array([
        [-0.9, +half_w, length, +half_w],
        [-0.9, -half_w, length, -half_w],
        [-0.9, -half_w, -0.9, +half_w],
        [length, -half_w, length, +half_w],
    ], dtype=float)

    steps = [[float(x), 0.0, float(top) / 2.0, float(top) / 2.0, float(h)]
             for (x, h, top) in _PILLAR_COLUMNS]

    pts = np.linspace(spawn, goal, 90).astype(np.float32)
    return Scenario(
        kind="pillar_course", name=name,
        spawn_xy=spawn.astype(np.float32),
        goal=goal.astype(np.float32),
        path_pts=pts,
        markers=np.empty((0, 2), dtype=np.float32),
        path_length=float(_arc_length(pts)),
        walls=walls.astype(np.float32),
        steps=steps,
    )


def pillar_course_columns(cfg) -> list[dict]:
    """The columns in travel order, with their edges worked out."""
    out = []
    for i, (x, h, top) in enumerate(_PILLAR_COLUMNS):
        out.append({
            "index": i,
            "x": float(x),
            "height": float(h),
            "half_top": float(top) / 2.0,
            "near": float(x) - float(top) / 2.0,
            "far": float(x) + float(top) / 2.0,
        })
    return out


_GENERATORS = {
    "path": path_scenario, "goal": goal_scenario,
    "roundtrip": roundtrip_scenario, "obstacle": obstacle_scenario,
    "maze": maze_scenario, "rocky_terrain": rocky_terrain_scenario,
    "rocky": rocky_terrain_scenario, "mountain": rocky_terrain_scenario,
    "slopes": slopes_scenario, "incline": slopes_scenario,
    "stairs": stairs_scenario, "staircase": stairs_scenario,
    "glass_pipe": glass_pipe_scenario, "pipe": glass_pipe_scenario, "tunnel": glass_pipe_scenario,
    "extreme_gauntlet": extreme_gauntlet_scenario, "gauntlet": extreme_gauntlet_scenario,
    "skill_course": skill_course_scenario, "course": skill_course_scenario,
    "platform_course": platform_course_scenario, "platforms": platform_course_scenario,
    "pillar_course": pillar_course_scenario, "pillars": pillar_course_scenario,
    "jump_track": jump_track_scenario, "jump": jump_track_scenario, "long_jump": jump_track_scenario, "hurdle": jump_track_scenario,
    "circle_track": circle_track_scenario, "circle": circle_track_scenario,
    "gap_bridge": gap_bridge_scenario, "straddle": gap_bridge_scenario, "chasm": gap_bridge_scenario, "trench": gap_bridge_scenario,
    "chimney": chimney_scenario, "vertical_shaft": chimney_scenario, "vertical_climb": chimney_scenario,
}





def generate_scenario(kind: str, cfg, *, seed=None, name: str | None = None) -> Scenario:
    """Generate a scenario of the given ``kind``."""
    if kind not in _GENERATORS:
        raise ValueError(f"unknown scenario kind {kind!r}; expected one of {KINDS}")
    rng = np.random.default_rng(seed)
    return _GENERATORS[kind](cfg, rng=rng, name=name or kind)

