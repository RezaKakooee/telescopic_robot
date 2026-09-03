"""Horizontal wall run: jump at a wall, ride it on momentum, push off again.

The parkour move. A runner sprints beside a wall, leaps at it, takes two or
three strides along it well above the ground, and pushes off before gravity
collects the debt. Nothing holds them there. Speed does.

That last point is why this skill exists at all. Friction cannot hold a rolling
body on a vertical wall: the friction that holds it also spins it, so it rolls
straight down. That was measured on the drome bowl, across wall radii from
0.70 m to 1.80 m, and it failed every time. So this is not a ride. It is a
long, shallow bounce off a vertical surface, and the whole skill is about
making that bounce soft at one end and hard at the other.

The manoeuvre has ten parts:

===========  ==================================================================
sprint       Build speed parallel to the wall without spending the lane gap.
approach     Angle toward the wall only after the sprint speed is real.
crouch       Briefly retract before takeoff so the rods have stroke to fire.
launch       Push off the floor, up and inward, to arrive at the wall high.
fly          Every rod out. The default robot becomes a 0.473 m sphere, so it meets the
             wall on rod tips rather than on its core.
ride         The soft end. Each rod facing the wall is commanded to the length
             that just reaches it, so as the centre keeps moving in, those rods
             retract to match. The robot sinks into its own shell instead of
             rebounding off it, and as it rotates, the next rods around take
             over the tracking.
push         The hard end. Those same rods drive back out and throw the robot
             off the wall.
land         Every rod out again, as a crash cage, so the core never takes the
             floor directly.
settle       Once down and still, fold back to an ordinary stance.
recover      Only when going again: peel away from the wall until the run-up
             lane is back, because the approach can only ever close the gap.
===========  ==================================================================

The one number that does the work is the reach to the wall, per rod:

    e_i = wall_dist / (rod_i . n)  -  foot_base

`wall_dist` is measured live, so the same expression is a soft catch during
`ride` (command a little past it and the rods yield as the centre closes) and a
shove during `push` (command far past it and they drive out).
"""
from __future__ import annotations

import numpy as np

from radial_sphere.geometry import quat_to_rotmat

from .locomotion import move, stop


#: Distance from the core centre to a foot's outer surface at zero stroke.
FOOT_BASE = 0.173

#: Rods pointing toward the wall hemisphere are compressed against the wall contour.
REACH_CONE = 0.05

#: Order the phases run in.
PHASES = (
    "recover", "sprint", "approach", "crouch", "launch",
    "fly", "ride", "push", "land", "settle",
)


def _frame(quat, dirs_body, wall_normal, travel):
    """Rod directions in world, and their components in the wall's frame."""
    n = np.asarray(wall_normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    t = np.asarray(travel, dtype=np.float64)
    t = t - float(np.dot(t, n)) * n
    t = t / max(float(np.linalg.norm(t)), 1e-9)
    dirs_world = np.asarray(dirs_body) @ quat_to_rotmat(quat).T
    return dirs_world, n, t


def wall_reach(dirs_world, wall_normal, wall_dist, max_extend,
               *, foot_base=FOOT_BASE, cone=REACH_CONE):
    """Stroke at which each rod's foot just meets the wall.

    Returns ``(reach, facing)``. `facing` marks the rods pointing close enough
    to the wall to be worth asking; the rest are left to the caller.
    """
    u_n = dirs_world @ np.asarray(wall_normal, dtype=np.float64)
    facing = u_n > cone
    reach = np.zeros(len(dirs_world), dtype=np.float64)
    reach[facing] = wall_dist / u_n[facing] - foot_base
    return np.clip(reach, 0.0, max_extend), facing


def wall_run(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    *,
    phase: str,
    wall_normal: np.ndarray,
    wall_dist: float,
    travel: np.ndarray,
    lin_vel: np.ndarray,
    speed: float = 7.5,
    recover_speed: float = 1.5,
    approach_angle: float = 28.0,
    lane_gap: float | None = None,
    lane_gain: float = 0.85,
    launch_in: float = 3.6,
    launch_up: float = 5.2,
    back_gain: float | None = 5.0,
    servo_band: float = 0.60,
    give: float = 0.13,
    squash_span: float = 0.26,
    cushion_max: float = 0.025,
    push_frac: float = 1.00,
    along_drive: float = 0.55,
    upward_bias: float = 0.35,
    min_offset: float = 0.025,
) -> np.ndarray:
    """One step of a horizontal wall run. Returns (n_bars,) rod targets."""
    vel = np.asarray(lin_vel, dtype=np.float64)
    dirs_world, n, t = _frame(quat, dirs_body, wall_normal, travel)
    u_n = dirs_world @ n
    u_t = dirs_world @ t
    u_z = dirs_world[:, 2]
    n_bars = len(dirs_body)
    full_reach = FOOT_BASE + max_extend

    if phase == "recover":
        head = t - np.tan(np.radians(approach_angle)) * n
        head = head / max(float(np.linalg.norm(head)), 1e-9)
        return move(quat, dirs_body, max_extend, np.array([head[0], head[1]]),
                    speed=recover_speed)

    if phase == "sprint":
        # A high-gain wave has small orientation-dependent lateral bias.  Close
        # that loop so the run-up stays parallel instead of quietly spending a
        # metre of lane before the commanded turn-in.
        correction = 0.0
        if lane_gap is not None:
            correction = float(np.clip(
                (wall_dist - lane_gap) * lane_gain, -0.40, 0.40))
        head = t + correction * n
        head /= max(float(np.linalg.norm(head)), 1e-9)
        return move(quat, dirs_body, max_extend, np.array([head[0], head[1]]),
                    speed=speed, back_gain=back_gain)

    if phase == "approach":
        head = t + np.tan(np.radians(approach_angle)) * n
        head = head / max(float(np.linalg.norm(head)), 1e-9)
        return move(quat, dirs_body, max_extend, np.array([head[0], head[1]]),
                    speed=speed, back_gain=back_gain)

    if phase == "crouch":
        return np.zeros(n_bars, dtype=np.float32)

    if phase == "launch":
        # Leave the floor with high vertical and inward speed. Downward and away-facing
        # rods deliver a maximum explosive stroke against the floor.
        want = launch_in * n + launch_up * np.array([0.0, 0.0, 1.0])
        want = want / max(float(np.linalg.norm(want)), 1e-9)
        aim = -(dirs_world @ want)
        short = max((launch_in - float(np.dot(vel, n))) / servo_band,
                    (launch_up - float(vel[2])) / servo_band)
        gain = float(np.clip(short, 0.0, 1.0))
        wave = np.clip((aim - 0.05) / 0.95, 0.0, 1.0) * gain
        wave[u_z > 0.08] = 0.0          # only downward/floor-contacting rods push
        return (min_offset + (max_extend - min_offset) * wave).astype(np.float32)

    if phase == "settle":
        return stop(quat, dirs_body, max_extend, lin_vel=vel)

    if phase in ("fly", "land"):
        return np.full(n_bars, max_extend, dtype=np.float32)

    reach, facing = wall_reach(dirs_world, n, wall_dist, max_extend)

    if phase == "ride":
        depth = max(full_reach - wall_dist, 0.0)
        soft = float(np.clip(depth / max(squash_span, 1e-6), 0.0, 1.0))
        cushion = soft * cushion_max - (1.0 - soft) * give

        targets = np.full(n_bars, max_extend, dtype=np.float64)
        targets[facing] = np.clip(reach[facing] + cushion, 0.0, max_extend)

        # Active along-wall propulsion and upward lift to sustain high-altitude rolling
        trailing = facing & (u_t < -0.02)
        targets[trailing] = np.clip(
            targets[trailing] + along_drive * max_extend * (-u_t[trailing]),
            0.0, max_extend
        )
        lower = facing & (u_z < -0.02)
        targets[lower] = np.clip(
            targets[lower] + upward_bias * max_extend * (-u_z[lower]),
            0.0, max_extend
        )

        return targets.astype(np.float32)

    if phase == "push":
        targets = np.full(n_bars, max_extend, dtype=np.float64)
        targets[facing] = push_frac * max_extend
        return targets.astype(np.float32)

    raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")


def get_wall_frame(pos: np.ndarray, scenario) -> tuple[float, np.ndarray, np.ndarray]:
    """Return distance to the wall *surface*, inward normal, and travel tangent.

    Scenario wall segments describe geom centre lines.  Subtracting the geom's
    half-thickness matters here: a 60 mm error is 20% of the available stroke.
    The banked wall is a rotated box, so its plane must also pass through the
    compiled geom centre at half wall height rather than through z=0.
    """
    mode = getattr(scenario, "wall_mode", "curved") if scenario is not None else "curved"
    walls = np.asarray(scenario.walls, dtype=float).reshape(-1, 4) if scenario is not None and scenario.walls is not None else np.array([[0.0, 1.30, 62.0, 1.30]])
    wall_y = float(walls[0, 1]) if len(walls) > 0 else 1.30
    half_thickness = 0.5 * float(getattr(scenario, "wall_thickness", 0.0))

    if mode == "curved":
        p = np.asarray(pos[:2], dtype=np.float64)
        x1 = walls[:, 0]
        y1 = walls[:, 1]
        x2 = walls[:, 2]
        y2 = walls[:, 3]
        dx = x2 - x1
        dy = y2 - y1
        lens = np.hypot(dx, dy)
        valid = lens > 1e-6
        if not np.any(valid):
            return float(wall_y - pos[1]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
        ux = dx / lens
        uy = dy / lens
        vx = p[0] - x1
        vy = p[1] - y1
        proj = np.clip(vx * ux + vy * uy, 0.0, lens)
        cx = x1 + proj * ux
        cy = y1 + proj * uy
        dist_sq = (p[0] - cx)**2 + (p[1] - cy)**2
        best_i = int(np.argmin(dist_sq))
        best_d = max(float(np.sqrt(dist_sq[best_i])) - half_thickness, 0.0)
        best_n = np.array([-uy[best_i], ux[best_i], 0.0])
        best_t = np.array([ux[best_i], uy[best_i], 0.0])
        return best_d, best_n, best_t

    elif mode == "banked":
        bank_deg = float(getattr(scenario, "wall_bank_deg", 12.0))
        rad = np.radians(bank_deg)
        # MuJoCo rotates the box's +Y face normal upward for a positive X roll.
        # This is the direction from the lane-side robot toward the wall.
        n = np.array([0.0, np.cos(rad), np.sin(rad)])
        t = np.array([1.0, 0.0, 0.0])
        wall_height = float(getattr(scenario, "wall_height", 0.0))
        centre_z = 0.5 * wall_height
        centre_to_robot = np.array([0.0, wall_y - pos[1], centre_z - pos[2]])
        wall_dist = float(np.dot(centre_to_robot, n)) - half_thickness
        return max(float(wall_dist), 0.0), n, t

    else:  # "flat" or "flat_multistep"
        wall_dist = float(wall_y - pos[1]) - half_thickness
        n = np.array([0.0, 1.0, 0.0])
        t = np.array([1.0, 0.0, 0.0])
        return wall_dist, n, t


def next_phase(
    phase: str,
    *,
    wall_dist: float,
    lin_vel: np.ndarray,
    height: float,
    on_ground: bool,
    max_extend: float,
    speed_ready: float,
    launch_gap: float,
    lane_gap: float = 0.0,
    wall_normal: np.ndarray,
    foot_base: float = FOOT_BASE,
    squash_frac: float = 0.45,
    turns_completed: float = 0.0,
    min_turns: float = 0.0,
    wall_mode: str = "curved",
    ride_distance: float = 0.0,
    ride_time: float = 0.0,
    target_distance: float = 1.0,
    min_ride_time: float = 0.10,
    push_height: float = 0.54,
    phase_elapsed: float = 0.0,
    crouch_time: float = 0.06,
) -> str:
    """Next phase, decided by geometry and velocity. Never by a step count.

    The approach speed has to be real before the robot commits, the wall has to
    be within reach before it starts tracking it, and the push only comes once
    the inward motion is spent. Time it instead and the same script works at
    one speed and nowhere else.

    `launch_gap` is the distance to the wall at which to leave the floor, and
    `lane_gap` is the distance to get back out to before running in again.
    """
    vel = np.asarray(lin_vel, dtype=np.float64)
    n = np.asarray(wall_normal, dtype=np.float64)
    closing = float(np.dot(vel, n))             # positive means still closing
    full_reach = foot_base + max_extend

    if phase == "recover":
        return "sprint" if wall_dist >= lane_gap else "recover"

    if phase == "sprint":
        along = float(np.linalg.norm(vel[:2]))
        return "approach" if along >= speed_ready * 0.88 else "sprint"

    if phase == "approach":
        along = float(np.linalg.norm(vel[:2]))
        ready = (along >= speed_ready * 0.72 and wall_dist <= launch_gap) or (wall_dist <= launch_gap * 0.75)
        return "crouch" if ready else "approach"

    if phase == "crouch":
        return "launch" if phase_elapsed >= crouch_time else "crouch"

    if phase == "launch":
        return "fly" if (not on_ground and height > 0.35) else "launch"

    if phase == "fly":
        return "ride" if wall_dist <= full_reach else "fly"

    if phase == "ride":
        lost_wall = wall_dist > full_reach + 0.08
        too_low = height < push_height
        travelled = ride_distance >= target_distance
        rolled = min_turns > 0.0 and turns_completed >= min_turns
        impact_spent = closing <= 0.05
        committed = ride_time >= min_ride_time
        if lost_wall or too_low:
            return "push"
        if committed and (travelled or rolled or impact_spent):
            return "push"
        return "ride"

    if phase == "push":
        clear = wall_dist > foot_base + 0.75 * max_extend
        if clear and closing < -0.20:
            return "land"
        spent = float(np.linalg.norm(vel)) < 0.60 and abs(closing) < 0.10
        if spent or wall_dist > foot_base + 0.95 * max_extend:
            return "land"
        return "push"

    if phase == "land":
        return "settle" if (on_ground and abs(float(vel[2])) < 0.35) else "land"

    return "settle"
