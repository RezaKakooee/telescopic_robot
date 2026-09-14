"""In-pipe crawling: roll along a round conduit without touching its walls.

The plain drive gait scrapes a pipe for two reasons. Its rear-bottom rods
extend to full stroke and, sitting at the bottom of a curve, a rod only 50
degrees from straight down already reaches the wall. And nothing steers the
ball back to the pipe axis, so a diagonal entry drifts into one side.

``crawl_pipe`` keeps the calibrated ``move`` gait and adds two things:

* **Wall cap.** Every rod's extension is limited to the free length between
  the core and the wall along that rod, minus the foot and a margin. The
  cap follows from the pipe radius and where the ball sits in the pipe's
  cross-section, so it holds for a ball at the bottom, off to one side, or
  bouncing.
* **Centering.** The heading is turned towards the axis in proportion to the
  lateral offset, like the cross-track term in ``move``.

Callers supply the axis direction ``d_hat`` and ``axis_offset`` — the ball
centre relative to the pipe axis in the cross-section plane as
``(lateral, vertical)`` metres, lateral positive to the LEFT of ``d_hat``.
``skills.runner`` and the RL option backend fill these in from the scenario's
pipe list.
"""
from __future__ import annotations

import numpy as np

from radial_sphere.gait import MIN_OFFSET
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mjcf import FOOT_RADIUS, SLEEVE_STUB, TIP_GAP

from .locomotion import move

#: Foot centre distance from the core centre at zero extension, plus the foot.
_TIP_REACH = SLEEVE_STUB + TIP_GAP + FOOT_RADIUS
#: Rods within ~55 deg of straight down are floor pushers: they may push this far
#: "into" the floor (that is the peristaltic stroke), side rods stop short of the wall.
SUPPORT_UZ = 0.55
PUSH_DEPTH = 0.04


def wall_extension_cap(
    dirs_world: np.ndarray,
    d_hat: np.ndarray,
    pipe_radius: float,
    axis_offset: tuple[float, float],
    core_radius: float,
    margin: float = 0.02,
    push_depth: float = PUSH_DEPTH,
) -> np.ndarray:
    """Largest extension each rod may have before its foot touches the pipe wall.

    Works in the cross-section plane: a rod with direction (u_lat, u_z) there,
    starting from the ball centre ``c = axis_offset``, meets the circle of
    radius ``pipe_radius`` at length ``t`` with ``|c + t u| = R``. Rods that
    run nearly along the axis get no cap (``inf``).
    """
    u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
    u_z = dirs_world[:, 2]
    u = np.column_stack([u_lat, u_z])
    c = np.asarray(axis_offset, dtype=np.float64)
    a = np.sum(u * u, axis=1)
    b = 2.0 * (u @ c)
    cc = float(c @ c) - pipe_radius ** 2          # negative while the centre is inside the pipe
    cap = np.full(len(u), np.inf)
    ok = a > 1e-6
    disc = np.maximum(b[ok] ** 2 - 4.0 * a[ok] * cc, 0.0)
    t_wall = (-b[ok] + np.sqrt(disc)) / (2.0 * a[ok])
    # Floor pushers keep a stroke into the floor (that is how the ball rolls);
    # rods aimed at the side and upper walls stop a margin short of them.
    allowance = np.where(u_z[ok] < -SUPPORT_UZ, push_depth, -margin)
    cap[ok] = t_wall - core_radius - _TIP_REACH + allowance
    return cap


def crawl_pipe(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    pipe_radius: float = 0.44,
    axis_offset: tuple[float, float] = (0.0, -0.25),
    core_radius: float = 0.15,
    speed: float = 0.7,
    centering_gain: float = 0.8,
    centering_damp: float = 0.6,
    max_steer_rad: float = 0.25,
    min_offset: float = MIN_OFFSET,
    lin_vel: np.ndarray | None = None,
    rod_mechanism: str | None = None,
    margin: float = 0.02,
    push_depth: float = PUSH_DEPTH,
) -> np.ndarray:
    """Rod targets for one control step of centred, wall-free rolling inside a pipe.

    Parameters
    ----------
    d_hat : (2,) unit vector along the pipe axis, in the travel direction.
    pipe_radius : inner radius of the pipe (m).
    axis_offset : ball centre relative to the axis, (lateral, vertical) metres;
        lateral is positive to the left of ``d_hat``.
    speed : cruising speed for the drive gait (m/s). Keep it moderate: the
        faster the ball, the further its trailing rods swing out.
    """
    d = np.asarray(d_hat, dtype=np.float64)[:2]
    d = d / (np.linalg.norm(d) + 1e-9)
    # Steer back towards the axis: an offset to the left turns the heading right.
    # Lateral speed is damped too, or the ball swings across the axis and
    # catches the far rim of the mouth.
    v_lat = 0.0
    if lin_vel is not None:
        v = np.asarray(lin_vel, dtype=np.float64)[:2]
        v_lat = float(-v[0] * d[1] + v[1] * d[0])
    steer = float(np.clip(-(centering_gain * float(axis_offset[0]) + centering_damp * v_lat),
                          -max_steer_rad, max_steer_rad))
    cs, sn = np.cos(steer), np.sin(steer)
    heading = np.array([cs * d[0] - sn * d[1], sn * d[0] + cs * d[1]])

    targets = move(quat, dirs_body, max_extend, heading, speed=speed, min_offset=min_offset,
                   lin_vel=lin_vel, rod_mechanism=rod_mechanism)

    dirs_world = np.asarray(dirs_body, dtype=np.float64) @ quat_to_rotmat(quat).T
    cap = wall_extension_cap(dirs_world, d, pipe_radius, axis_offset, core_radius, margin, push_depth)
    return np.clip(np.minimum(targets, cap), min_offset, max_extend).astype(np.float32)


#: Approach zone before a pipe mouth (m) in which the skill centres the ball but
#: does not cap the rods yet, and how far past the far end it still counts.
ENTRY_ZONE = 0.25
APPROACH = 2.0
EXIT_PAD = 0.35


def pipe_frame(pipes, pos: np.ndarray):
    """Which pipe (if any) the point ``pos`` is near, and the ball's place in it.

    ``pipes`` rows are ``(start_x, start_y, length, inner_radius, outer_radius,
    yaw_deg)`` as in ``Scenario.pipes``. Returns ``(d_hat, radius, axis_offset,
    along, length)`` or ``None``. ``d_hat`` points along the pipe from its
    start and ``along`` is the distance from the start; flip both if you
    travel the other way. The caller decides from the travel direction whether
    the ball is in the approach zone (steer hard, no cap), inside (cap) or
    past the far end (no cap).
    """
    if pipes is None:
        return None
    from radial_sphere.terrain import Pipe, rows

    p = np.asarray(pos, dtype=np.float64)
    for pipe in rows(Pipe, pipes):
        yaw = np.radians(float(pipe.yaw_deg or 0.0))
        d = np.array([np.cos(yaw), np.sin(yaw)])
        rel = p[:2] - np.array([pipe.x, pipe.y])
        along = float(rel @ d)
        lateral = float(rel[0] * (-d[1]) + rel[1] * d[0])
        if -APPROACH <= along <= pipe.length + APPROACH and abs(lateral) < pipe.inner_radius + 0.3:
            axis_z = pipe.inner_radius + 0.02            # pipe_xml sets the axis at in_rad + 0.02
            vertical = float(p[2]) - axis_z if len(p) > 2 else -pipe.inner_radius * 0.55
            return d, float(pipe.inner_radius), (lateral, vertical), along, float(pipe.length)
    return None
