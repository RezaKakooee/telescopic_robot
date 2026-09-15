"""Ten industrial-inspection courses built from the simple scenario features.

Each course has its own layout, not just its own obstacles:

==  ======================  ==================================================
1   warehouse               random shelving maze (dead ends), pallets, a beam
2   pipe_alley              open yard, diagonal lane between two pipe racks
3   tank_farm               open yard, S-route between nine tanks and bunds
4   substation              row of fenced bays entered one after the other
5   loading_dock            hall with parked trucks, dock ramp, stairs down
6   boiler_house            three rooms joined by doorways, a mezzanine
7   utility_tunnel          narrow L tunnel with a dead-end branch, manhole
8   solar_farm              open field, snake through four post rows, ditch
9   quarry                  switchback road up a hill, rock fall, descent
10  rubble_site             open square, random slabs, collapsed wall, crack
==  ======================  ==================================================

The objects are plain boxes, pillars, pits, ramps, stairs, pipes, pebbles and
sand; the situations are the point. No expert or oracle is defined here yet.

    from radial_sphere.scenario import generate_scenario
    sc = generate_scenario("inspection_warehouse", cfg, seed=0)
"""
from __future__ import annotations

import numpy as np

WALL_HEIGHT = 1.0        # fences and partitions: too tall to jump over by accident


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _densify(corners, step: float = 0.1) -> np.ndarray:
    pts = np.asarray(corners, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t_key = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, t_key[-1], step)
    t = np.append(t, t_key[-1])                      # always end exactly at the goal
    return np.column_stack([np.interp(t, t_key, pts[:, 0]), np.interp(t, t_key, pts[:, 1])]).astype(np.float32)


def _boundary(x0, y0, x1, y1) -> list:
    return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]


def _seg(x0, y0, x1, y1) -> list:
    return [x0, y0, x1, y1]


def _make(name: str, corners, walls, tour: bool = False, **features):
    """Assemble a Scenario for a route through `corners` with explicit walls.

    A ``tour`` is a long demonstration route that revisits obstacles and may
    cross itself; it is tracked in order (``monotonic_path``)."""
    from .scenario import Scenario, _arc_length

    path = _densify(corners)
    obstacles = features.pop("obstacles", None)
    return Scenario(
        kind=name, name=name,
        spawn_xy=np.asarray(corners[0], dtype=float), goal=np.asarray(corners[-1], dtype=float),
        path_pts=path, markers=path[::12].copy(), path_length=_arc_length(path),
        walls=np.asarray(walls, dtype=np.float32).reshape(-1, 4),
        obstacles=np.asarray(obstacles, dtype=np.float32).reshape(-1, 3) if obstacles is not None else None,
        monotonic_path=tour,
        wall_height=WALL_HEIGHT,
        **features,
    )


def _beam(x, y, along: str, height: float, length: float, thickness: float = 0.10) -> list:
    """A curb / beam / floor pipe centred at (x, y), running along `along` ('x' or 'y')."""
    hx, hy = (length / 2, thickness / 2) if along == "x" else (thickness / 2, length / 2)
    return [x, y, hx, hy, height]


def _slope(cx, cy, length, width, height, yaw_deg, up: bool) -> list:
    """Ramp whose local +x end is higher (up=True) or lower (up=False)."""
    pitch = float(np.degrees(np.arctan2(height, length)))
    return [cx, cy, length, width, height, pitch if up else -pitch, yaw_deg]


def _plateau(cx, cy, length, width, height, yaw_deg=0.0) -> list:
    return [cx, cy, length, width, height, 0.0, yaw_deg]


# --------------------------------------------------------------------------- #
# 1. Warehouse: a random shelving maze with dead ends
# --------------------------------------------------------------------------- #
def _maze_walk(sc, cell, cols, rows):
    """Depth-first walk of every maze cell from the spawn, then on to the goal.

    Adjacent cells are connected when no wall covers the edge between them.
    The walk backtracks through corridors, so it retraces itself."""
    walls = np.asarray(sc.walls, dtype=float)

    def blocked(a, b):
        (i0, j0), (i1, j1) = a, b
        mx, my = (i0 + i1) / 2 * cell, (j0 + j1) / 2 * cell
        for x1, y1, x2, y2 in walls:
            if abs(x1 - x2) < 1e-6 and abs(x1 - mx) < 1e-6 and min(y1, y2) - 1e-6 <= my <= max(y1, y2) + 1e-6:
                return True
            if abs(y1 - y2) < 1e-6 and abs(y1 - my) < 1e-6 and min(x1, x2) - 1e-6 <= mx <= max(x1, x2) + 1e-6:
                return True
        return False

    def neighbours(c):
        i, j = c
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (i + d[0], j + d[1])
            if 0 <= n[0] < cols and 0 <= n[1] < rows and not blocked(c, n):
                yield n

    start = (int(round(sc.spawn_xy[0] / cell)), int(round(sc.spawn_xy[1] / cell)))
    goal = (int(round(sc.goal[0] / cell)), int(round(sc.goal[1] / cell)))
    walk, seen = [start], {start}

    def dfs(c):
        for n in neighbours(c):
            if n not in seen:
                seen.add(n)
                walk.append(n)
                dfs(n)
                walk.append(c)
    dfs(start)
    # shortest way from the end of the walk to the goal (BFS)
    from collections import deque
    prev, q = {walk[-1]: None}, deque([walk[-1]])
    while q:
        c = q.popleft()
        if c == goal:
            break
        for n in neighbours(c):
            if n not in prev:
                prev[n] = c
                q.append(n)
    tail, c = [], goal
    while c is not None and c != walk[-1]:
        tail.append(c)
        c = prev[c]
    walk += tail[::-1]
    # drop immediate back-and-forth duplicates created by dead ends of length 1
    return [(i * cell, j * cell) for i, j in walk]


def warehouse(cfg, *, rng=None, name="inspection_warehouse", tour: bool = False):
    from .scenario import _random_maze_scenario

    rng = rng if rng is not None else np.random.default_rng(0)
    cell, cols, rows = 2.0, 6, 5
    bounds = (-cell / 2, -cell / 2, (cols - 0.5) * cell, (rows - 0.5) * cell)
    sc = _random_maze_scenario(cfg, rng, name, cell, cols, rows, bounds, level=3)
    sc.kind = name
    pts = sc.path_pts
    n = len(pts) - 1

    def side_point(frac, off):
        k = int(frac * n)
        p, d = pts[k], pts[min(k + 5, n)] - pts[max(k - 5, 0)]
        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        return p + nrm * off, d

    steps = []
    for k, frac in enumerate((0.2, 0.45, 0.7, 0.9)):            # pallets parked against alternating walls
        q, _ = side_point(frac, 0.72 if k % 2 else -0.72)
        steps.append([float(q[0]), float(q[1]), 0.30, 0.25, 0.14])
    # a fallen beam across the middle of the longest straight run of the route
    d = np.diff(pts, axis=0)
    axis = np.where(np.abs(d[:, 0]) > np.abs(d[:, 1]), 0, 1)
    best, start = (0, 0), 0
    for k in range(1, len(axis) + 1):
        if k == len(axis) or axis[k] != axis[start]:
            if k - start > best[0]:
                best = (k - start, start)
            start = k
    mid = best[1] + best[0] // 2
    q = pts[mid]
    along = "y" if axis[best[1]] == 0 else "x"
    steps.append(_beam(float(q[0]), float(q[1]), along, 0.16, cell - 0.15, 0.12))
    sand = [[float(rng.uniform(bounds[0] + 1, bounds[2] - 1)), float(rng.uniform(bounds[1] + 1, bounds[3] - 1)), 0.4, 0.4]]
    sc.steps, sc.sand_patches = steps, sand
    sc.wall_height = WALL_HEIGHT
    if tour:
        from .scenario import _arc_length
        path = _densify(_maze_walk(sc, cell, cols, rows))
        sc.path_pts, sc.markers, sc.path_length = path, path[::12].copy(), _arc_length(path)
        sc.monotonic_path = True
    return sc


# --------------------------------------------------------------------------- #
# 2. Pipe alley: open yard, a diagonal lane between two pipe racks
# --------------------------------------------------------------------------- #
def pipe_alley(cfg, *, rng=None, name="inspection_pipe_alley", tour: bool = False):
    walls = _boundary(-1.5, -1.5, 13.5, 9.5)
    obstacles = []
    for k in range(4):                                           # two rows of rack posts along the 45 deg lane
        t = k * 1.2
        obstacles.append([0.3 + t, 1.6 + t, 0.10])
        obstacles.append([1.6 + t, 0.3 + t, 0.10])
    steps = [[3.0, 3.0, 0.35, 0.35, 0.10], [4.9, 5.5, 0.35, 0.35, 0.12]]   # pipe saddles on the lane centre line
    pipes = [[9.0, 8.0, 3.0, 0.44, 0.46, 0.0]]                             # conduit x in [9, 12] at y = 8
    lane = [(0.0, 0.0), (4.4, 4.4), (5.4, 6.6), (6.6, 7.8), (8.0, 8.0), (12.5, 8.0)]   # gentle arc, then straight into the mouth
    route = lane
    if tour:   # through the lane and conduit, back round the outside, and through again
        route = lane + [(12.5, 4.5), (9.5, 1.0), (6.5, -0.6), (2.0, -0.9)] + lane
    return _make(name, route, walls, tour=tour, obstacles=obstacles, steps=steps, pipes=pipes)


# --------------------------------------------------------------------------- #
# 3. Tank farm: open yard, S-route between nine tanks and their bund curbs
# --------------------------------------------------------------------------- #
def tank_farm(cfg, *, rng=None, name="inspection_tank_farm", tour: bool = False):
    walls = _boundary(-1.5, -1.5, 13.5, 13.5)
    obstacles = [[2.0 + 4.0 * i, 2.0 + 4.0 * j, 0.75] for i in range(3) for j in range(3)]   # 3 x 3 tanks
    steps = [_beam(6.0, 3.7, "x", 0.20, 11.0, 0.12), _beam(6.0, 7.7, "x", 0.20, 11.0, 0.12)]   # bund curbs between rows
    stones = [[10.0, 10.0, 1.2, 1.2, 40, 0.045]]                                                # gravel at the last tank
    route = [(0.0, 0.0), (4.0, 0.0), (8.0, 1.0), (12.0, 0.0), (12.0, 4.0), (8.0, 5.0), (4.0, 4.0),
             (0.0, 5.0), (0.0, 8.0), (4.0, 9.0), (8.0, 8.0), (12.0, 9.0), (12.0, 12.0)]
    if tour:   # two lanes straight through both bund curbs first, then the S-route
        route = [(0.0, 0.0), (4.0, 0.0), (4.0, 12.0), (8.0, 12.0), (8.0, 0.0), (12.0, 0.0), (12.0, 4.0), (8.0, 5.0),
                 (4.0, 4.0), (0.0, 5.0), (0.0, 8.0), (4.0, 9.0), (8.0, 8.0), (12.0, 9.0), (12.0, 12.0)]
    return _make(name, route, walls, tour=tour, obstacles=obstacles, steps=steps, stones=stones)


# --------------------------------------------------------------------------- #
# 4. Substation: a row of fenced bays, each entered and left in turn
# --------------------------------------------------------------------------- #
def substation(cfg, *, rng=None, name="inspection_substation", tour: bool = False):
    """Four bays in a row. Each bay has a back door to a rear service corridor;
    the front road and the rear corridor are blocked alternately, so the route
    snakes through bay 1 (front to back), bay 2 (back to front), and so on."""
    walls = _boundary(-1.5, -1.5, 15.5, 6.5)
    walls += [_seg(4.0, -1.5, 4.0, 5.0), _seg(12.0, -1.5, 12.0, 5.0)]        # full separators: front road blocked
    walls += [_seg(8.0, 1.0, 8.0, 6.5)]                                     # bay side wall + rear corridor blocked
    walls += [_seg(0.0, 1.0, 0.0, 5.0), _seg(16.0, 1.0, 16.0, 5.0)]
    # back wall y = 5 with a door per bay
    for x0, x1 in ((-1.5, 1.3), (2.7, 5.3), (6.7, 9.3), (10.7, 13.3), (14.7, 15.5)):
        walls.append(_seg(x0, 5.0, x1, 5.0))
    obstacles = [[b * 4.0 + 2.0 + dx, 3.0, 0.12] for b in range(4) for dx in (-1.0, 1.0)]   # insulator posts
    gaps = [[6.0, 1.6, 1.3, 0.2, 0.35], [14.0, 3.4, 1.3, 0.2, 0.35]]                       # cable trenches
    stones = [[2.0, 2.8, 1.0, 0.8, 30, 0.04], [10.0, 2.6, 1.0, 0.8, 30, 0.04]]              # gravel in two bays
    fwd = [(-0.5, -0.5), (2.0, -0.3), (2.0, 5.75), (6.0, 5.75), (6.0, -0.3), (10.0, -0.3),
           (10.0, 5.75), (14.0, 5.75), (14.0, -0.3), (15.0, -0.3)]
    route = fwd
    if tour:   # to the end, back to bay 2, and forward again
        route = fwd + [(14.0, -0.3), (14.0, 5.75), (10.0, 5.75), (10.0, -0.3), (6.0, -0.3), (6.0, 5.75), (3.0, 5.75)] + fwd[3:]
    return _make(name, route, walls, tour=tour, obstacles=obstacles, gaps=gaps, stones=stones)


# --------------------------------------------------------------------------- #
# 5. Loading dock: hall with parked trucks, dock ramp, stairs down
# --------------------------------------------------------------------------- #
def loading_dock(cfg, *, rng=None, name="inspection_loading_dock", tour: bool = False):
    walls = _boundary(-1.5, -1.5, 14.5, 8.5)
    h = 0.40
    ramps = [_slope(3.0, 6.0, 2.0, 2.4, h, 0.0, up=True), _plateau(7.0, 6.0, 6.0, 2.4, h)]   # dock x in [4, 10], y in [4.8, 7.2]
    staircases = [[10.0, 6.0, 2, 0.20, 0.6, 2.4, 0.0, True]]                                   # down, heading +x
    steps = [
        [2.5, 1.5, 1.6, 0.9, 0.55], [8.5, 1.5, 1.6, 0.9, 0.55],                # two parked trucks (tall blocks)
        [6.0, 6.3, 0.45, 0.35, h + 0.15], [8.0, 5.7, 0.45, 0.35, h + 0.15],    # crates on the dock
        [13.6, 4.6, 0.5, 0.4, 0.14],                                            # pallet by the wall near the exit
    ]
    cones = [[5.0, 3.0, 0.10], [5.6, 3.6, 0.10]]
    dock = [(1.5, 6.0), (10.5, 6.0), (11.5, 6.0), (11.5, 3.0)]
    route = [(0.0, 0.0), (0.0, 3.0)] + dock + [(13.0, 0.5)]
    if tour:   # over the dock, back between the trucks, and over the dock again
        route = [(0.0, 0.0), (0.0, 3.0)] + dock + [(11.0, -0.5), (5.5, -0.5), (5.5, 3.2), (0.0, 4.5), (-0.7, 6.0)] + dock[1:] + [(13.0, 0.5)]
    return _make(name, route, walls, tour=tour, ramps=ramps, staircases=staircases, steps=steps, cones=cones)


# --------------------------------------------------------------------------- #
# 6. Boiler house: three rooms joined by doorways, a mezzanine in the middle
# --------------------------------------------------------------------------- #
def boiler_house(cfg, *, rng=None, name="inspection_boiler_house", tour: bool = False):
    walls = _boundary(-1.5, -1.5, 16.5, 6.5)
    walls += [_seg(4.5, -1.5, 4.5, 3.6), _seg(4.5, 5.0, 4.5, 6.5)]          # partition 1, doorway y in [3.6, 5.0]
    walls += [_seg(13.5, -1.5, 13.5, 1.2), _seg(13.5, 3.8, 13.5, 6.5)]      # partition 2, doorway y in [1.2, 3.8] where the ramp lands
    top = 0.80
    staircases = [[6.8, 2.5, 2, 0.40, 1.8, 2.4, 0.0, False]]                 # up, x in [6.8, 10.4]; 2.3 m run-up after the door
    ramps = [_plateau(11.0, 2.5, 1.2, 2.4, top), _slope(12.6, 2.5, 2.0, 2.4, top, 0.0, up=False)]  # deck x in [10.4, 11.6], then down to x = 13.6
    steps = [_beam(1.5, 1.0, "y", 0.10, 3.0, 0.10), _beam(3.0, 1.0, "y", 0.12, 3.0, 0.12),          # floor pipes in room 1, y in [-0.5, 2.5]
             _beam(10.8, 2.5, "y", top + 0.10, 2.2, 0.10)]                                           # pipe on the deck, 2.2 m past the top riser
    obstacles = [[15.0, 4.3, 0.30], [15.0, 0.7, 0.30]]                                               # boiler drums in room 3
    rest = [(4.5, 4.3), (5.2, 2.5), (14.0, 2.5), (15.0, 2.5), (15.8, 4.0), (15.8, 5.5)]
    route = [(0.0, 0.0), (2.2, 1.5)] + rest
    if tour:   # cross both floor pipes three times before the door
        route = [(0.0, 0.0), (4.0, 1.5), (4.0, 4.0), (-0.5, 4.0), (-0.5, 0.5), (4.0, 2.0)] + rest
    return _make(name, route, walls, tour=tour, staircases=staircases, ramps=ramps, steps=steps, obstacles=obstacles)


# --------------------------------------------------------------------------- #
# 7. Utility tunnel: narrow L tunnel with a dead-end branch and a manhole
# --------------------------------------------------------------------------- #
def utility_tunnel(cfg, *, rng=None, name="inspection_utility_tunnel", tour: bool = False):
    w = 0.75                                                                 # tunnel half-width
    walls = [_seg(-1.0, -w, -1.0, w), _seg(-1.0, w, 9.0 - w, w)]            # leg A along +x: start cap, top wall
    walls += [_seg(-1.0, -w, 3.25, -w), _seg(4.75, -w, 9.0 + w, -w)]         # bottom wall with a branch opening
    walls += [_seg(3.25, -w, 3.25, -3.0), _seg(4.75, -w, 4.75, -3.0), _seg(3.25, -3.0, 4.75, -3.0)]   # dead-end side branch
    walls += [_seg(9.0 - w, w, 9.0 - w, 8.5), _seg(9.0 + w, -w, 9.0 + w, 8.5), _seg(9.0 - w, 8.5, 9.0 + w, 8.5)]  # leg B, closed corner
    pipes = [[1.0, 0.0, 2.4, 0.44, 0.46, 0.0], [9.0, 3.5, 2.4, 0.44, 0.46, 90.0]]   # second pipe 3.5 m after the corner
    gaps = [[7.4, 0.0, 0.25, 0.45, 0.50]]                                    # open manhole on leg A, 0.5 m along the travel
    sand = [[6.0, 0.0, 0.4, 0.5]]                                             # standing water
    steps = [_beam(4.4, 0.0, "y", 0.10, 1.4, 0.10)]                           # cable cover
    route = [(0.0, 0.0), (9.0, 0.0), (9.0, 8.0)]
    if tour:   # visit the dead-end branch, go to the end, come back through the pipe, and go again
        route = [(0.0, 0.0), (4.0, 0.0), (4.0, -2.4), (4.0, 0.0), (9.0, 0.0), (9.0, 7.6), (9.0, 1.0), (5.0, 0.0), (9.0, 0.0), (9.0, 8.0)]
    return _make(name, route, walls, tour=tour, pipes=pipes, gaps=gaps, sand_patches=sand, steps=steps)


# --------------------------------------------------------------------------- #
# 8. Solar farm: open field, snake through four post rows, cross a ditch
# --------------------------------------------------------------------------- #
def solar_farm(cfg, *, rng=None, name="inspection_solar_farm", tour: bool = False):
    walls = _boundary(-1.5, -1.5, 13.5, 10.5)
    obstacles = [[float(x), 2.0 * r + 0.5, 0.08] for r in range(4) for x in np.arange(0.0, 12.1, 1.5)]   # 4 post rows
    gaps = [[6.0, 4.5, 7.5, 0.18, 0.30]]                                                              # drainage ditch, wall to wall
    sand = [[3.0, 1.5, 0.8, 0.5], [9.0, 5.5, 0.8, 0.5]]                                               # mud
    snake = [(0.0, -0.5), (12.0, -0.5), (12.5, 1.5), (0.0, 1.5), (-0.5, 3.5), (12.0, 3.5),
             (12.5, 5.5), (0.0, 5.5), (-0.5, 7.5), (12.0, 7.5)]
    route = snake + [(12.0, 9.5)]
    if tour:   # after the snake, cross the ditch twice more between the post rows
        route = snake + [(6.75, 7.5), (6.75, 3.5), (9.75, 3.5), (9.75, 7.5), (12.0, 7.5), (12.0, 9.5)]
    return _make(name, route, walls, tour=tour, obstacles=obstacles, gaps=gaps, sand_patches=sand)


# --------------------------------------------------------------------------- #
# 9. Quarry: switchback road up a hill, rock fall, long descent
# --------------------------------------------------------------------------- #
def quarry(cfg, *, rng=None, name="inspection_quarry", tour: bool = False):
    walls = _boundary(-1.5, -1.5, 13.5, 9.5)
    h = 0.45
    ramps = [
        _slope(3.0, 0.0, 4.0, 2.4, h, 0.0, up=True),          # climb, x in [1, 5]
        _plateau(6.7, 1.5, 5.4, 3.6, h, 90.0),                 # hairpin landing, x in [4.9, 8.5], y in [-1.2, 4.2]
        _plateau(2.5, 3.0, 6.5, 2.4, h),                       # hill road heading back -x, x in [-0.75, 5.75]
        _plateau(0.0, 5.25, 4.9, 2.4, h, 90.0),                # corner, y in [2.8, 7.7]
        _slope(5.2, 6.5, 8.0, 2.4, h, 0.0, up=False),           # long descent heading +x, x in [1.2, 9.2]
    ]
    stones = [[11.0, 6.5, 1.0, 0.9, 30, 0.07]]                # rock fall at the foot of the descent
    gaps = [[11.5, 2.5, 0.25, 0.25, 0.15], [12.0, 4.5, 0.25, 0.25, 0.15]]   # potholes on the flat return
    cones = [[10.5, 8.0, 0.10], [12.0, 5.5, 0.10]]
    hill = [(0.0, 0.0), (5.5, 0.0), (7.0, 1.5), (5.5, 3.0), (0.0, 3.0), (0.0, 6.3), (10.5, 6.5)]
    route = hill + [(12.0, 5.0), (12.0, 1.0)]
    if tour:   # over the hill, back over it in reverse, and over it once more
        route = hill + hill[::-1][1:] + hill[1:] + [(12.0, 5.0), (12.0, 1.0)]
    return _make(name, route, walls, tour=tour, ramps=ramps, stones=stones, gaps=gaps, cones=cones)


# --------------------------------------------------------------------------- #
# 10. Rubble site: open square, random slabs, a collapsed wall, a floor crack
# --------------------------------------------------------------------------- #
def rubble_site(cfg, *, rng=None, name="inspection_rubble_site", tour: bool = False):
    rng = rng if rng is not None else np.random.default_rng(0)
    walls = _boundary(-1.5, -1.5, 11.5, 11.5)
    walls += [_seg(0.0, 10.0, 4.2, 5.8), _seg(5.8, 4.2, 10.0, 0.0)]         # collapsed wall, one opening near (5, 5)
    route = [(0.0, 0.0), (2.5, 1.5), (4.0, 4.0), (6.75, 6.2), (6.75, 8.6), (10.0, 10.0)]
    if tour:   # beam and crack twice, through the wall opening both ways
        route = [(0.0, 0.0), (2.5, 1.5), (4.0, 4.0), (6.75, 6.2), (6.75, 8.6), (9.3, 9.0), (9.0, 6.0), (6.0, 5.2),
                 (4.5, 3.0), (2.5, 1.5), (4.0, 4.0), (6.75, 6.2), (6.75, 8.6), (10.0, 10.0)]
    path = _densify(route)
    steps = []
    for _ in range(9):                                                      # random slabs kept 0.9 m off the route
        for _try in range(30):
            x, y = rng.uniform(0.0, 10.0, size=2)
            if np.min(np.linalg.norm(path - [x, y], axis=1)) > 0.9:
                steps.append([float(x), float(y), float(rng.uniform(0.3, 0.6)), float(rng.uniform(0.3, 0.6)),
                              float(rng.uniform(0.15, 0.40))])
                break
    steps.append(_beam(3.25, 2.75, "x", 0.18, 2.0, 0.10))                  # a fallen beam across the route
    gaps = [[6.75, 7.5, 0.9, 0.25, 0.40]]                                   # crack across the route
    stones = [[2.0, 6.0, 1.5, 1.5, 40, 0.06], [8.5, 2.5, 1.5, 1.5, 40, 0.06]]
    return _make(name, route, walls, tour=tour, steps=steps, gaps=gaps, stones=stones)


# --------------------------------------------------------------------------- #
# 11-13. Jump drills: many jumps in a row, 3 m of straight approach each
# --------------------------------------------------------------------------- #
def hurdle_lane(cfg, *, rng=None, name="inspection_hurdle_lane", tour: bool = False):
    """A long lane with 14 beams across it, heights between 0.10 and 0.20 m."""
    rng = rng if rng is not None else np.random.default_rng(0)
    walls = _boundary(-1.5, -1.5, 46.5, 1.5)
    steps = [_beam(3.0 + 3.0 * i, 0.0, "y", float(rng.uniform(0.10, 0.20)), 2.8, 0.10) for i in range(14)]
    route = [(0.0, 0.0), (45.0, 0.0)]
    if tour:   # there and back: 28 jumps
        route = [(0.0, 0.0), (45.0, 0.0), (45.0, 0.6), (0.0, 0.6), (0.0, 0.0), (45.0, 0.0)]
    return _make(name, route, walls, tour=tour, steps=steps)


def trench_field(cfg, *, rng=None, name="inspection_trench_field", tour: bool = False):
    """Four 20 m lanes of a snake, each crossed by three trenches 0.30 to 0.50 m wide.

    Five metres between a corner and the first trench: a 90 degree turn at
    speed swings the ball almost a metre off the lane, and it needs that
    distance to settle before a jump."""
    rng = rng if rng is not None else np.random.default_rng(0)
    walls = _boundary(-1.5, -1.5, 21.5, 10.5)
    gaps = []
    for lane in range(4):
        y = 3.0 * lane
        for x in (5.0, 10.0, 15.0):
            w = float(rng.uniform(0.30, 0.50))
            gaps.append([x, y, w / 2, 1.2, 0.40])          # across the lane (half_x = width/2)
    route = [(0.0, 0.0), (20.0, 0.0), (20.0, 3.0), (0.0, 3.0), (0.0, 6.0), (20.0, 6.0), (20.0, 9.0), (0.0, 9.0)]
    return _make(name, route, walls, tour=tour, gaps=gaps)      # the tour is the same snake: 12 jumps


def box_steps(cfg, *, rng=None, name="inspection_box_steps", tour: bool = False):
    """A lane that alternates low walls (0.15 to 0.30 m) and gaps (0.35 to 0.50 m)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    walls = _boundary(-1.5, -1.5, 34.5, 1.5)
    steps, gaps = [], []
    for i in range(10):
        x = 3.0 + 3.0 * i
        if i % 2 == 0:
            steps.append(_beam(x, 0.0, "y", float(rng.uniform(0.15, 0.30)), 2.8, 0.12))
        else:
            w = float(rng.uniform(0.35, 0.50))
            gaps.append([x, 0.0, w / 2, 1.4, 0.40])
    route = [(0.0, 0.0), (33.0, 0.0)]
    if tour:
        route = [(0.0, 0.0), (33.0, 0.0), (33.0, 0.6), (0.0, 0.6), (0.0, 0.0), (33.0, 0.0)]
    return _make(name, route, walls, tour=tour, steps=steps, gaps=gaps)


INSPECTION_SCENARIOS = {
    "inspection_warehouse": warehouse,
    "inspection_pipe_alley": pipe_alley,
    "inspection_tank_farm": tank_farm,
    "inspection_substation": substation,
    "inspection_loading_dock": loading_dock,
    "inspection_boiler_house": boiler_house,
    "inspection_utility_tunnel": utility_tunnel,
    "inspection_solar_farm": solar_farm,
    "inspection_quarry": quarry,
    "inspection_rubble_site": rubble_site,
    "inspection_hurdle_lane": hurdle_lane,
    "inspection_trench_field": trench_field,
    "inspection_box_steps": box_steps,
}
