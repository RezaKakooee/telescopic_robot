"""Real-time perception engine for RoboBall: Local Elevation/Occupancy Map Patch and Global Waypoint Tracking."""
from __future__ import annotations

import numpy as np


#: Half-size (m) from which a `steps` box counts as a deck the ball can land on.
DECK_MIN_HALF_SIZE = 0.4


class LocalMapPatchExtractor:
    """Extracts an egocentric 2D local elevation and obstacle occupancy patch around the robot.

    Generates a high-resolution precomputed global heightmap and occupancy mask from
    the scenario geometry (walls, stairs, pipes, pits, and obstacles), then extracts
    an N x N egocentric patch around the ball in O(1) time.
    """

    def __init__(
        self,
        scenario,
        bounds: tuple[float, float, float, float] | None = None,
        res: float = 0.1,
    ):
        self.scenario = scenario
        if bounds is None:
            points = np.concatenate((np.asarray(scenario.path_pts).reshape(-1, 2),
                                     np.asarray(scenario.walls).reshape(-1, 2),
                                     np.asarray(scenario.goal).reshape(1, 2)))
            lo, hi = points.min(axis=0) - 2.0, points.max(axis=0) + 2.0
            bounds = (lo[0], hi[0], lo[1], hi[1])
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.res = res

        self.nx = int(np.round((self.xmax - self.xmin) / self.res))
        self.ny = int(np.round((self.ymax - self.ymin) / self.res))

        # Channel 0: Ground Elevation Z (metres)
        self.global_elevation = np.zeros((self.ny, self.nx), dtype=np.float32)
        # Channel 1: Impassable Static Obstacle Occupancy (0.0 = free, 1.0 = solid)
        self.global_occupancy = np.zeros((self.ny, self.nx), dtype=np.float32)

        self._rasterize_scenario()

    def rasterize_physics(self, model, data, geomgroup) -> None:
        """Read terrain from collision geometry, including custom playground props.

        The environment's ray mask excludes robot bodies and visual decals.
        Four samples per cell retain thin crossbars that a centre ray misses.
        This is a static map, built once; no physics steps are advanced.
        """
        import mujoco

        mujoco.mj_forward(model, data)
        ray_z = max(10.0, float(np.max(data.geom_xpos[:, 2] + model.geom_rbound)) + 1.0)
        direction = np.array([0., 0., -1.])
        origin = np.array([0., 0., ray_z])
        geom_id = np.zeros(1, dtype=np.int32)
        for iy in range(self.ny):
            for ix in range(self.nx):
                height = -np.inf
                for ox, oy in ((.25, .25), (.25, .75), (.75, .25), (.75, .75)):
                    origin[:2] = (self.xmin + (ix + ox) * self.res,
                                  self.ymin + (iy + oy) * self.res)
                    distance = mujoco.mj_ray(model, data, origin, direction,
                                             geomgroup, True, -1, geom_id)
                    if distance >= 0:
                        height = max(height, ray_z - distance)
                if np.isfinite(height):
                    self.global_elevation[iy, ix] = height

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(np.clip((x - self.xmin) / self.res, 0, self.nx - 1))
        gy = int(np.clip((y - self.ymin) / self.res, 0, self.ny - 1))
        return gx, gy

    def _rasterize_scenario(self):
        xs = np.linspace(self.xmin + self.res / 2, self.xmax - self.res / 2, self.nx)
        ys = np.linspace(self.ymin + self.res / 2, self.ymax - self.res / 2, self.ny)
        XX, YY = np.meshgrid(xs, ys)

        # 1. Building walls: line segments with thickness ~0.2m
        walls = getattr(self.scenario, "walls", None)
        if walls is not None and len(walls) > 0:
            w_arr = np.asarray(walls, dtype=np.float32).reshape(-1, 4)
            for w in w_arr:
                x1, y1, x2, y2 = w
                dx, dy = x2 - x1, y2 - y1
                L2 = dx * dx + dy * dy
                if L2 > 1e-6:
                    t = np.clip(((XX - x1) * dx + (YY - y1) * dy) / L2, 0.0, 1.0)
                    px = x1 + t * dx
                    py = y1 + t * dy
                    dist = np.sqrt((XX - px) ** 2 + (YY - py) ** 2)
                    self.global_occupancy[dist < 0.25] = 1.0
                    self.global_elevation[dist < 0.25] = 0.5  # wall height

        # 2. Obstacles (boxes, crates, pillars)
        obs = getattr(self.scenario, "obstacles", None)
        if obs is not None and len(obs) > 0:
            obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1, 3)
            for ob in obs_arr:
                cx, cy, r = ob
                dist = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
                self.global_occupancy[dist < r] = 1.0
                self.global_elevation[dist < r] = 0.4

        # 3. Staircases (ascending steps)
        stairs = getattr(self.scenario, "staircases", None)
        if stairs is not None and len(stairs) > 0:
            for flight in stairs:
                sx, sy, n_steps, rise, run, width, yaw, is_down = flight
                yaw_rad = np.radians(yaw)
                c, s = np.cos(yaw_rad), np.sin(yaw_rad)
                for step_i in range(int(n_steps)):
                    # step center
                    d_run = (step_i + 0.5) * run
                    step_cx = sx + d_run * c
                    step_cy = sy + d_run * s
                    step_z = (step_i + 1) * rise if not is_down else -(step_i + 1) * rise

                    # bounding box around step
                    dx = XX - step_cx
                    dy = YY - step_cy
                    # rotate to step frame
                    along = dx * c + dy * s
                    across = -dx * s + dy * c
                    in_step = (np.abs(along) <= run / 2.0) & (np.abs(across) <= width / 2.0)
                    self.global_elevation[in_step] = step_z

        # 4. Ramps / Elevated Terrace Landing
        ramps = getattr(self.scenario, "ramps", None)
        if ramps is not None and len(ramps) > 0:
            for r in ramps:
                cx, cy, lx, ly, h_change, pitch, yaw = r
                yaw_rad = np.radians(yaw)
                c, s = np.cos(yaw_rad), np.sin(yaw_rad)
                dx = XX - cx
                dy = YY - cy
                along = dx * c + dy * s
                across = -dx * s + dy * c
                in_ramp = (np.abs(along) <= lx / 2.0) & (np.abs(across) <= ly / 2.0)
                self.global_elevation[in_ramp] = h_change

        # 5. Cross-walk Utility Pipes
        pipes = getattr(self.scenario, "pipes", None)
        if pipes is not None and len(pipes) > 0:
            for pipe in pipes:
                sx, sy, length, in_r, out_r, yaw = pipe
                yaw_rad = np.radians(yaw)
                c, s = np.cos(yaw_rad), np.sin(yaw_rad)
                for step_d in np.linspace(0, length, int(length / (self.res / 2))):
                    px = sx + step_d * c
                    py = sy + step_d * s
                    dist = np.sqrt((XX - px) ** 2 + (YY - py) ** 2)
                    in_pipe = dist <= out_r
                    self.global_elevation[in_pipe] = np.maximum(self.global_elevation[in_pipe], out_r)

        # 6. Recessed Floor Gaps / Pits
        gaps = getattr(self.scenario, "gaps", None)
        if gaps is not None and len(gaps) > 0:
            for gap in gaps:
                gx, gy, hx, hy, depth = gap
                in_gap = (np.abs(XX - gx) <= hx) & (np.abs(YY - gy) <= hy)
                self.global_elevation[in_gap] = -depth

        # 7. Decks: `steps` boxes wide enough to land on. The jump option ends
        # when the ball is near the ground under it, so a deck that is not in
        # the map keeps the option airborne for its whole 2.4 s and the ball
        # coasts off the far edge. Beams and curbs stay out on purpose: they
        # are not landing surfaces, and a beam in the map would start the
        # landing phase while the ball is still passing over it.
        steps = getattr(self.scenario, "steps", None)
        if steps is not None and len(steps) > 0:
            for step in steps:
                sx, sy, hx, hy, height = (float(v) for v in step[:5])
                if min(hx, hy) < DECK_MIN_HALF_SIZE:
                    continue
                on_deck = (np.abs(XX - sx) <= hx) & (np.abs(YY - sy) <= hy)
                self.global_elevation[on_deck] = np.maximum(self.global_elevation[on_deck], height)

    def get_patch(self, core_pos: np.ndarray, grid_size: int = 16, patch_span: float = 4.0) -> np.ndarray:
        """Extract a local grid_size x grid_size patch centered on core_pos.

        Returns:
            np.ndarray of shape (2, grid_size, grid_size):
                Channel 0: Relative ground elevation (z_ground - z_core)
                Channel 1: Static obstacle occupancy [0.0, 1.0]
        """
        cx, cy, cz = float(core_pos[0]), float(core_pos[1]), float(core_pos[2])
        half_span = patch_span / 2.0

        sample_xs = np.linspace(cx - half_span, cx + half_span, grid_size)
        sample_ys = np.linspace(cy - half_span, cy + half_span, grid_size)

        # Map sample coords to grid indices
        ix = np.clip(((sample_xs - self.xmin) / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip(((sample_ys - self.ymin) / self.res).astype(int), 0, self.ny - 1)

        # Pool each output pixel's footprint so downsampling cannot erase a
        # narrow hurdle between the 16 observation sample centres.
        radius = int(np.ceil(patch_span / max(grid_size - 1, 1) / self.res / 2))
        offsets = np.arange(-radius, radius + 1)
        xs = np.clip(ix[:, None] + offsets, 0, self.nx - 1)
        ys = np.clip(iy[:, None] + offsets, 0, self.ny - 1)
        half_pixel = patch_span / max(grid_size - 1, 1) / 2
        valid_x = np.abs(self.xmin + (xs + .5) * self.res - sample_xs[:, None]) <= half_pixel + 1e-6
        valid_y = np.abs(self.ymin + (ys + .5) * self.res - sample_ys[:, None]) <= half_pixel + 1e-6
        # Always retain the nearest source cell when upsampling or at map edges.
        valid_x[:, radius] = True
        valid_y[:, radius] = True
        footprint = valid_y[:, :, None, None] & valid_x[None, None, :, :]
        heights = self.global_elevation[ys[:, :, None, None], xs[None, None, :, :]]
        occupancy = self.global_occupancy[ys[:, :, None, None], xs[None, None, :, :]]
        # A ceil-rounded radius is only a search window. Pooling all of it
        # expands a 27 cm pixel to 50 cm and erases the 45 cm parkour valleys.
        patch_elev = np.where(footprint, heights, -np.inf).max(axis=(1, 3)) - cz
        patch_occ = np.where(footprint, occupancy, 0).max(axis=(1, 3))

        return np.stack([patch_elev, patch_occ], axis=0).astype(np.float32)


class WaypointTracker:
    """Tracks global navigation waypoints and provides egocentric guidance vectors."""

    def __init__(self, path_pts: np.ndarray, goal: np.ndarray, lookahead_steps: int = 12):
        self.path_pts = np.asarray(path_pts, dtype=np.float32).reshape(-1, 2)
        self.goal = np.asarray(goal, dtype=np.float32)[:2]
        self.lookahead_steps = lookahead_steps
        self.n_pts = len(self.path_pts)
        if self.n_pts == 0:
            raise ValueError("WaypointTracker requires at least one path point")
        self._segments = np.diff(self.path_pts.astype(np.float64), axis=0)
        self._segment_lengths = np.linalg.norm(self._segments, axis=1)
        self._arc_lengths = np.concatenate(([0.0], np.cumsum(self._segment_lengths)))

    def _closest_index(self, pos: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(self.path_pts - pos, axis=1)))

    def get_path_progress(self, core_pos: np.ndarray) -> tuple[int, float]:
        """Return (closest waypoint index, remaining XY path distance in metres).

        Project onto the closest polyline segment so progress is continuous
        between waypoints, including when their spacing is uneven. Backtracking
        increases the remaining distance. Height does not affect route progress.
        """
        pos = np.asarray(core_pos[:2], dtype=np.float64)
        closest_idx = self._closest_index(pos)
        if self.n_pts == 1:
            return closest_idx, float(np.linalg.norm(self.path_pts[0] - pos))

        lengths_sq = self._segment_lengths ** 2
        fractions = np.divide(
            np.sum((pos - self.path_pts[:-1]) * self._segments, axis=1),
            lengths_sq,
            out=np.zeros_like(lengths_sq),
            where=lengths_sq > 0.0,
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = self.path_pts[:-1] + fractions[:, None] * self._segments
        segment_idx = int(np.argmin(np.sum((projections - pos) ** 2, axis=1)))
        travelled = (self._arc_lengths[segment_idx]
                     + fractions[segment_idx] * self._segment_lengths[segment_idx])
        return closest_idx, float(self._arc_lengths[-1] - travelled)

    def get_guidance(self, core_pos: np.ndarray) -> np.ndarray:
        """Computes relative vectors to lookahead waypoint and final goal.

        Returns:
            np.ndarray of shape (6,):
                [wx_hat, wy_hat, dist_waypoint, gx_hat, gy_hat, dist_goal]
        """
        pos = np.asarray(core_pos[:2], dtype=np.float32)

        # Find closest path point
        closest_idx = self._closest_index(pos)

        # Lookahead point
        look_idx = min(closest_idx + self.lookahead_steps, self.n_pts - 1)
        look_pt = self.path_pts[look_idx]

        # Waypoint vector
        diff_w = look_pt - pos
        d_w = float(np.linalg.norm(diff_w))
        dir_w = diff_w / (d_w + 1e-6) if d_w > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)

        # Goal vector
        diff_g = self.goal - pos
        d_g = float(np.linalg.norm(diff_g))
        dir_g = diff_g / (d_g + 1e-6) if d_g > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)

        return np.array([
            dir_w[0], dir_w[1], d_w,
            dir_g[0], dir_g[1], d_g,
        ], dtype=np.float32)


class MonotonicWaypointTracker(WaypointTracker):
    """A WaypointTracker for routes that cross or retrace themselves.

    The base tracker takes the closest point on the whole route, so at a
    crossing it may jump to a later (or earlier) part of the route. This one
    remembers where it was and only looks in a window around that index:
    ``back`` points behind and ``ahead`` points in front (0.1 m spacing gives
    2 m and 4 m). Call ``reset()`` at the start of an episode.
    """

    def __init__(self, path_pts, goal, lookahead_steps: int = 12, back: int = 20, ahead: int = 40,
                 back_bias: float = 0.01):
        super().__init__(path_pts, goal, lookahead_steps)
        self.back, self.ahead = back, ahead
        self.back_bias = back_bias     # metres of penalty per index behind the last match: ties go forward
        self.reset()

    def reset(self) -> None:
        self._last_idx = 0

    def _closest_index(self, pos: np.ndarray) -> int:
        lo = max(0, self._last_idx - self.back)
        hi = min(self.n_pts, self._last_idx + self.ahead + 1)
        dist = np.linalg.norm(self.path_pts[lo:hi] - pos, axis=1)
        # Where the route retraces itself (a dead end), the outbound and return
        # points coincide; the penalty makes the later, return-leg point win.
        cost = dist + self.back_bias * np.maximum(0, self._last_idx - np.arange(lo, hi))
        idx = lo + int(np.argmin(cost))
        self._last_idx = idx
        return idx

    def get_path_progress(self, core_pos: np.ndarray) -> tuple[int, float]:
        pos = np.asarray(core_pos[:2], dtype=np.float64)
        closest_idx = self._closest_index(pos)
        if self.n_pts == 1:
            return closest_idx, float(np.linalg.norm(self.path_pts[0] - pos))
        lo = max(0, closest_idx - self.back)
        hi = min(self.n_pts - 1, closest_idx + self.ahead)
        seg = self._segments[lo:hi]
        lengths_sq = self._segment_lengths[lo:hi] ** 2
        fractions = np.divide(np.sum((pos - self.path_pts[lo:hi]) * seg, axis=1), lengths_sq,
                              out=np.zeros_like(lengths_sq), where=lengths_sq > 0.0)
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = self.path_pts[lo:hi] + fractions[:, None] * seg
        k = int(np.argmin(np.sum((projections - pos) ** 2, axis=1)))
        travelled = self._arc_lengths[lo + k] + fractions[k] * self._segment_lengths[lo + k]
        return closest_idx, float(self._arc_lengths[-1] - travelled)
