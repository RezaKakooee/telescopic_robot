"""Locomotion: one parametric gait, plus named presets of it.

The shell is a Fibonacci sphere, so the robot has no front, back or side --
driving in any direction is the same manoeuvre. There is therefore only one
locomotion skill, :func:`move`, and it takes two continuous numbers:

    turn   -- radians to rotate the commanded heading by
    speed  -- desired cruise speed, in metres per second

Everything else in this module is a preset. ``go_fast`` is ``move`` at
2.2 m/s, ``move_right`` is ``move`` at ``turn=-pi/2``, ``reverse`` is ``move``
at ``turn=pi``. They exist because named commands read better in a plan, not
because they are different gaits.

Speed is given in m/s rather than as an internal gain so that a planner -- or
a policy -- can ask for a physical quantity. The conversion comes from
measurement, not theory: see :data:`SPEED_CURVE`.

Every function follows the same contract:
    Input:  robot state (quat, dirs_body, max_extend) + skill params.
    Output: np.ndarray of shape (n_bars,) with rod extension targets.
"""
from __future__ import annotations

import numpy as np

from radial_sphere.gait import MIN_OFFSET

from radial_sphere.geometry import quat_to_rotmat


#: Cruise speed a skill assumes when the caller does not say.
DEFAULT_SPEED = 1.2

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decompose(quat, dirs_body, d_hat):
    """Decompose rod directions into longitudinal / lateral / vertical."""
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T
    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_lat  = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
    u_z    = dirs_world[:, 2]
    return dirs_world, u_long, u_lat, u_z


# Measured cruise speed against push-wave amplitude, at full stroke, averaged
# over the last 250 of 600 steps on flat ground. This is the map that lets a
# caller ask for m/s instead of a gain. Re-measure it if the robot changes.
#
# Two builds are calibrated, because the same amplitude is not the same speed
# on a longer rod. The relation is not a simple scale: at the lowest amplitude
# the 0.30 m build is 4.0x faster than the 0.16 m one, at the highest only
# 1.95x. So both curves are stored and the answer is interpolated between them
# by stroke, rather than stretched from one.
SPEED_CURVE = (
    # back_gain, cruise speed (m/s), stroke 0.16 m
    (1.00, 0.33),
    (1.15, 0.44),
    (1.30, 0.57),
    (1.60, 0.86),
    (2.00, 1.22),
    (2.40, 1.57),
    (2.80, 1.90),
    (3.20, 2.25),
    (4.00, 2.80),
)

#: The same amplitudes on the long-stroke build.
SPEED_CURVE_LONG = (
    # back_gain, cruise speed (m/s), stroke 0.30 m
    (1.00, 1.32),
    (1.15, 1.58),
    (1.30, 1.86),
    (1.60, 2.34),
    (2.00, 2.98),
    (2.40, 3.54),
    (2.80, 4.03),
    (3.20, 4.42),
    (4.00, 5.46),
)

# Current antenna mechanism, 16 cm stroke, default config, flat ground,
# seed 42: mean forward speed during seconds 4--6 of each six-second run.
# Reproduce with scripts/skills/calibrate_move.py. Keep the older curves for
# other mechanisms and strokes; they are also used by existing skill presets.
SPEED_CURVE_MULTI_STAGE = (
    (1.00, 0.46714), (1.15, 0.61039), (1.30, 0.79409),
    (1.60, 1.14873), (2.00, 1.55258), (2.40, 1.89108),
    (2.80, 2.22560), (3.20, 2.59613), (4.00, 3.18586),
)

#: The strokes those two curves were measured at.
CURVE_STROKES = (0.16, 0.30)

_GAINS = np.array([g for g, _v in SPEED_CURVE])
_SPEEDS = np.array([v for _g, v in SPEED_CURVE])
_SPEEDS_LONG = np.array([v for _g, v in SPEED_CURVE_LONG])

#: Slowest and fastest cruise the default 0.16 m build can hold.
MIN_SPEED = float(_SPEEDS[0])
MAX_SPEED = float(_SPEEDS[-1])

# Braking, measured by asking for a stop distance and recording the real one.
# The relation is not exactly coast ~ 1/gain (the wave clips at full stroke),
# so this constant is a least-squares fit over 0.25-1.0 m rather than a law.
# Expect the achieved distance to be within about 0.1 m of the request.
#: Drive amplitude for `straddle_gap`. Tuned, not derived from a speed
#: request: the load rides on two narrow flank strips and needs the
#: traction. `speed_for_gain` puts it near 2.7 m/s on flat ground.
STRADDLE_GAIN = 3.8

COAST_PER_SPEED = 0.62
MAX_BRAKE_GAIN = 3.0


def speed_curve(max_extend=None):
    """Cruise speeds for :data:`SPEED_CURVE`'s amplitudes, at a given stroke.

    Interpolated between the two measured builds and held flat outside them.
    Pass None for the 0.16 m default.
    """
    if max_extend is None:
        return _SPEEDS
    t = float(np.clip((float(max_extend) - CURVE_STROKES[0])
                      / (CURVE_STROKES[1] - CURVE_STROKES[0]), 0.0, 1.0))
    return _SPEEDS * (1.0 - t) + _SPEEDS_LONG * t


def speed_range(max_extend=None):
    """``(slowest, fastest)`` cruise this build can hold, in m/s."""
    sp = speed_curve(max_extend)
    return float(sp[0]), float(sp[-1])


def gain_for_speed(speed, max_extend=None):
    """Push-wave amplitude that cruises at `speed` m/s, from the measured curve.

    Speeds outside the measured range are clamped: below the slowest the rods
    stop reaching the ground and the robot stalls rather than crawling, and
    above the fastest there is no evidence to extrapolate from.

    Pass `max_extend` whenever the build is not the default 0.16 m stroke.
    Without it a request for 1.2 m/s on the 0.30 m build returns the amplitude
    that would cruise at 1.2 m/s on the short one, which on the long one is
    nearer 2.9 m/s.
    """
    sp = speed_curve(max_extend)
    v = float(np.clip(speed, sp[0], sp[-1]))
    return float(np.interp(v, sp, _GAINS))


def speed_for_gain(gain, max_extend=None):
    """Inverse of :func:`gain_for_speed`; cruise speed a given amplitude holds."""
    return float(np.interp(float(gain), _GAINS, speed_curve(max_extend)))


def _rotate(d_hat, radians):
    """Turn a 2-vector by `radians`, counter-clockwise."""
    c, sn = np.cos(radians), np.sin(radians)
    d = np.asarray(d_hat, dtype=np.float64)
    return np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]])


# ---------------------------------------------------------------------------
# 1. move_forward
# ---------------------------------------------------------------------------

def move(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    speed: float = DEFAULT_SPEED,
    turn: float = 0.0,
    min_offset: float = MIN_OFFSET,
    back_gain: float | None = None,
    lin_vel: np.ndarray | None = None,
    cross_track_error: float = 0.0,
    rod_mechanism: str | None = None,
) -> np.ndarray:
    """Roll at `speed` m/s, in the direction `d_hat` rotated by `turn` radians.

    This is the only locomotion gait in the library. Trailing rods in the
    lower hemisphere extend to make rolling torque; leading and upper rods
    stay shut.

    Parameters
    ----------
    d_hat : (2,) reference direction in world xy.
    turn : radians to rotate that reference by. 0 drives along it, +pi/2 a
        quarter turn anticlockwise, pi drives back along it. Continuous --
        the right angles are not special, they are just the ones with names.
    speed : signed speed in m/s. Positive follows the rotated reference;
        negative travels opposite it. Zero delegates to stop (active braking
        when lin_vel is supplied). Nonzero magnitude is clamped to the
        selected calibration's range and converted to push strength.
    back_gain : escape hatch. Set it to drive the amplitude directly and
        override nonzero speed magnitude; the direction still follows its sign.
    lin_vel : optional measured world velocity. Enables speed correction and
        lateral velocity damping. Omit to retain the feedforward gait.
    cross_track_error : signed distance (m) left of the actual requested travel
        direction, after applying turn and the sign of speed.
        Used with lin_vel to steer back toward that line. The caller owns
        the line origin; zero provides velocity damping only.
    rod_mechanism : pass the actual mechanism to select its calibration.
        The multi_stage table is measured at 0.16 m stroke only.

    Notes
    -----
    Speed comes from the wave amplitude, never from the rod stroke. Shortening
    the stroke does not slow the robot, it stops the rods reaching the ground,
    and below about 60 % stroke the gait stalls outright.

    The two arguments are deliberately a continuous pair: ``(turn, speed)`` is
    a complete, physically-scaled action for this robot, so a policy can emit
    it directly.
    """
    heading = _rotate(d_hat, turn) if turn else np.asarray(d_hat, dtype=np.float64)
    heading_norm = float(np.linalg.norm(heading))
    if heading.shape != (2,) or not np.isfinite(heading).all() or heading_norm < 1e-9:
        raise ValueError("d_hat must be a finite nonzero horizontal 2-vector")
    heading = heading / heading_norm
    if not np.isfinite(speed):
        raise ValueError("speed must be finite")
    if speed == 0:
        return stop(quat, dirs_body, max_extend, lin_vel=lin_vel)
    if speed < 0:
        heading = -heading
    magnitude = abs(float(speed))
    gain = float(back_gain) if back_gain is not None else gain_for_speed(magnitude, max_extend)
    if back_gain is None:
        curve = speed_curve(max_extend)
        if rod_mechanism == "multi_stage" and np.isclose(max_extend, 0.16):
            curve = np.array([v for _, v in SPEED_CURVE_MULTI_STAGE])
        desired_speed = float(np.clip(magnitude, curve[0], curve[-1]))
        gain = float(np.interp(desired_speed, curve, _GAINS))
        if lin_vel is not None:
            velocity = np.asarray(lin_vel, dtype=float)[:2]
            if velocity.shape != (2,) or not np.isfinite(velocity).all():
                raise ValueError("lin_vel must contain finite horizontal velocity")
            if not np.isfinite(cross_track_error):
                raise ValueError("cross_track_error must be finite")
            sideways = np.array([-heading[1], heading[0]])
            forward_speed = float(velocity @ heading)
            lateral_speed = float(velocity @ sideways)
            # Bounded proportional speed correction around the calibrated gain.
            gain = float(np.clip(gain + 2.0 * (desired_speed - forward_speed), 0.5, 4.0))
            # Position restores the requested line; velocity damps sideways motion.
            correction = np.clip(-4.0 * lateral_speed - 4.0 * cross_track_error,
                                 -0.6 * desired_speed, 0.6 * desired_speed)
            heading = heading * desired_speed + sideways * correction
            heading /= np.linalg.norm(heading)

    _, u_long, u_lat, u_z = _decompose(quat, dirs_body, heading)

    rear = np.clip((-u_long - 0.10) / 0.90, 0.0, 1.0)
    down = np.clip(1.0 - abs(u_z + 0.35) / 0.85, 0.0, 1.0)
    tuck = np.clip(1.0 - 1.8 * (u_lat ** 2), 0.0, 1.0)
    wave = np.clip((rear ** 1.1) * down * gain * tuck, 0.0, 1.0)

    wave[u_long > -0.05] = 0.0
    wave[u_z > 0.10] = 0.0

    targets = min_offset + (max_extend - min_offset) * wave
    return targets.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. move_forward  (preset)
# ---------------------------------------------------------------------------

def turn(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    angle_deg: float = 0.0,
    speed: float = DEFAULT_SPEED,
    lin_vel: np.ndarray | None = None,
    cross_track_error: float = 0.0,
    rod_mechanism: str | None = None,
    back_gain: float | None = None,
    min_offset: float = MIN_OFFSET,
) -> np.ndarray:
    """Drive at a signed angle relative to d_hat: positive right, negative left.

    angle_deg is a heading offset in degrees, not an angular velocity or an
    extra rotation applied each update. Hold d_hat fixed during this command.
    Internally move retains its established counter-clockwise radians API.
    cross_track_error is measured left of the resulting requested heading.
    """
    if not np.isfinite(angle_deg):
        raise ValueError("angle_deg must be finite")
    return move(quat, dirs_body, max_extend, d_hat,
                turn=-float(np.deg2rad(angle_deg)), speed=speed,
                lin_vel=lin_vel, cross_track_error=cross_track_error,
                rod_mechanism=rod_mechanism, back_gain=back_gain,
                min_offset=min_offset)


def move_forward(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    speed: float = DEFAULT_SPEED,
    back_gain: float | None = None,
    min_offset: float = MIN_OFFSET,
    lin_vel: np.ndarray | None = None,
    cross_track_error: float = 0.0,
    rod_mechanism: str | None = None,
) -> np.ndarray:
    """Drive along *d_hat* at `speed` m/s. Default 1.2 m/s.

    A preset of :func:`move`. Ask `move` for any other heading or speed.
    """
    return move(quat, dirs_body, max_extend, d_hat, speed=speed,
                turn=0.0, back_gain=back_gain, min_offset=min_offset,
                lin_vel=lin_vel, cross_track_error=cross_track_error,
                rod_mechanism=rod_mechanism)


# ---------------------------------------------------------------------------
# 2. move_right
# ---------------------------------------------------------------------------

def move_right(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    speed: float = DEFAULT_SPEED,
    back_gain: float | None = None,
    min_offset: float = MIN_OFFSET,
) -> np.ndarray:
    """Drive a quarter turn clockwise of *d_hat*.

    A preset of :func:`move`. For any other angle call ``move`` with the `turn` you want.
    """
    return move(quat, dirs_body, max_extend, d_hat, speed=speed,
                turn=-np.pi / 2, back_gain=back_gain, min_offset=min_offset)


# ---------------------------------------------------------------------------
# 3. move_left
# ---------------------------------------------------------------------------

def move_left(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    speed: float = DEFAULT_SPEED,
    back_gain: float | None = None,
    min_offset: float = MIN_OFFSET,
) -> np.ndarray:
    """Drive a quarter turn anticlockwise of *d_hat*.

    A preset of :func:`move`. For any other angle call ``move`` with the `turn` you want.
    """
    return move(quat, dirs_body, max_extend, d_hat, speed=speed,
                turn=+np.pi / 2, back_gain=back_gain, min_offset=min_offset)


# ---------------------------------------------------------------------------
# 4. stop
# ---------------------------------------------------------------------------

def stop(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    *,
    lin_vel: np.ndarray | None = None,
    stop_distance: float | None = None,
    stance_height: float = 0.045,
    brake_gain: float = 1.0,
    brake_speed_ref: float = 0.6,
    brake_min_speed: float = 0.05,
) -> np.ndarray:
    """Bring the ball to rest, then hold a stable stance.

    Two regimes, selected automatically from *lin_vel*:

    **Active braking** (speed above ``brake_min_speed``). Rods in the
    *leading* downward sector extend into the floor ahead of the contact
    point. This kickstand contact produces a counter-torque
    :math:`\\tau = r \\times F` opposing the roll, so the ball decelerates
    instead of coasting. Brake stroke scales with speed, so the ball settles
    smoothly rather than slamming to a halt.

    **Passive stance** (near rest, or ``lin_vel`` not supplied). Bottom
    cluster at a low standoff, everything else retracted. The robot stays
    upright with no forward, lateral, or rotational drive.

    Parameters
    ----------
    lin_vel : (2,) or (3,) world-frame velocity, optional.
        Supply it to get active braking. Omit it for the passive stance only.
    stop_distance : metres you want to stop within. The brake strength is
        worked out from the measured braking fit, so a caller asks for a
        distance rather than tuning a gain. Accurate to roughly 0.1 m. There
        is a floor: from 2 m/s the shortest stop is about 0.45 m, and asking
        for less simply gets full braking.
    brake_gain : scales the leading-rod brake stroke. Ignored if
        `stop_distance` is given.
    brake_speed_ref : speed (m/s) at which the brake reaches full stroke.
    brake_min_speed : below this speed the passive stance takes over.
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T
    u_z = dirs_world[:, 2]

    targets = np.zeros(len(dirs_body), dtype=np.float32)
    bottom_mask = u_z < -0.30
    targets[bottom_mask] = stance_height

    if lin_vel is None:
        return targets

    v = np.asarray(lin_vel, dtype=np.float64)[:2]
    speed = float(np.linalg.norm(v))
    if speed < brake_min_speed:
        return targets

    if stop_distance is not None and stop_distance > 1e-3:
        # coast ~ COAST_PER_SPEED * speed / brake_gain, so invert for the gain.
        brake_gain = float(np.clip(COAST_PER_SPEED * speed / stop_distance,
                                   0.25, MAX_BRAKE_GAIN))

    # Travel direction; the brake acts on the rods leading the roll.
    d_hat = v / speed
    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]

    # Leading-sector kickstand: ahead of the contact point and below the core.
    front = np.clip((u_long - 0.05) / 0.75, 0.0, 1.0)
    down = np.clip(1.0 - abs(u_z + 0.45) / 0.75, 0.0, 1.0)
    lat_tuck = np.clip(1.0 - 1.8 * (u_lat ** 2), 0.0, 1.0)

    strength = min(1.0, speed / max(brake_speed_ref, 1e-6)) * brake_gain
    wave = np.clip(front * down * lat_tuck * strength, 0.0, 1.0)
    wave[u_z > 0.05] = 0.0

    brake = stance_height + (max_extend - stance_height) * wave
    targets = np.maximum(targets, brake).astype(np.float32)
    return targets


# ---------------------------------------------------------------------------
# 5. go_fast
# ---------------------------------------------------------------------------

def go_fast(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    speed: float = 2.25,
    back_gain: float | None = None,
    min_offset: float = MIN_OFFSET,
) -> np.ndarray:
    """Drive along *d_hat* at 2.25 m/s, near the gait's ceiling.

    A preset of :func:`move`. Fast and slow are one gait; only the wave amplitude differs.
    """
    return move(quat, dirs_body, max_extend, d_hat, speed=speed,
                turn=0.0, back_gain=back_gain, min_offset=min_offset)


# ---------------------------------------------------------------------------
# 6. go_slow
# ---------------------------------------------------------------------------

def go_slow(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    speed: float = 0.45,
    back_gain: float | None = None,
    min_offset: float = MIN_OFFSET,
) -> np.ndarray:
    """Drive along *d_hat* at 0.45 m/s, near the gait's floor.

    A preset of :func:`move`. Below about 0.33 m/s the rods stop reaching the ground.
    """
    return move(quat, dirs_body, max_extend, d_hat, speed=speed,
                turn=0.0, back_gain=back_gain, min_offset=min_offset)


# ---------------------------------------------------------------------------
# 7. reverse
# ---------------------------------------------------------------------------

def reverse(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    speed: float = DEFAULT_SPEED,
    back_gain: float | None = None,
    min_offset: float = MIN_OFFSET,
) -> np.ndarray:
    """Drive back along *d_hat*.

    A preset of :func:`move`. Backwards is not a separate manoeuvre on a symmetric shell.
    """
    return move(quat, dirs_body, max_extend, d_hat, speed=speed,
                turn=np.pi, back_gain=back_gain, min_offset=min_offset)


# ---------------------------------------------------------------------------
# 14. circle (circular trajectory / orbital steering)
# ---------------------------------------------------------------------------

def circle(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    *,
    ball_xy: np.ndarray | None = None,
    d_hat: np.ndarray | None = None,
    center_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
    radius: float = 1.8,
    speed: float = 1.0,
    clockwise: bool = False,
    lookahead: float = 0.25,
    radial_gain: float = 2.0,
    min_offset: float = MIN_OFFSET,
    back_gain: float | None = None,
) -> np.ndarray:
    """Drive continuously in a circular orbit of specified radius.

    Uses pure-pursuit orbital steering with dynamic understeer compensation:
    computes an orbital target point ahead along the circumference and adjusts
    the instantaneous heading demand so the robot holds a steady circular trajectory
    with sub-centimetre radial variance.

    Parameters
    ----------
    ball_xy : (2,) world-frame xy position of the ball.
    d_hat : (2,) reference direction (fallback if ball_xy is None).
    center_xy : (2,) center of the circle in world xy (default [0, 0]).
    radius : desired circle radius in metres (default 1.8 m).
    speed : target cruise speed in m/s (default 1.0 m/s).
    clockwise : True for clockwise circle, False for counter-clockwise.
    lookahead : look-ahead arc distance in metres (default 0.25 m).
    radial_gain : stiffness of radial correction (default 2.0).
    """
    if ball_xy is not None:
        p = np.asarray(ball_xy, dtype=np.float64)[:2]
        c = np.asarray(center_xy, dtype=np.float64)[:2]
        rel = p - c
        r = float(np.linalg.norm(rel))
        th_now = float(np.arctan2(rel[1], rel[0]))

        # Direction of travel: +1 for counter-clockwise, -1 for clockwise
        sign = -1.0 if clockwise else +1.0

        # Look-ahead angle along the circular path
        d_th = sign * (lookahead / max(radius, 0.2))
        th_target = th_now + d_th

        # Dynamic radius target: pulls inward when drifting out to counteract turning lag
        # Calibrated feedforward lead: offset target by 0.06m to center steady-state tracking
        nominal_r = max(0.2, radius - 0.065)
        r_target_dynamic = max(0.2, nominal_r - radial_gain * (r - radius))
        p_target = c + r_target_dynamic * np.array([np.cos(th_target), np.sin(th_target)])

        # Heading vector from current ball position to the lookahead target point
        heading_vec = p_target - p
        h_norm = float(np.linalg.norm(heading_vec))
        d_cmd = heading_vec / max(h_norm, 1e-6)
    elif d_hat is not None:
        d_cmd = np.asarray(d_hat, dtype=np.float64)
    else:
        d_cmd = np.array([1.0, 0.0], dtype=np.float64)

    return move(quat, dirs_body, max_extend, d_hat=d_cmd, speed=speed,
                turn=0.0, back_gain=back_gain, min_offset=min_offset)


# ---------------------------------------------------------------------------
# 14b. curve (continuous curved path / arc / winding street navigation)
# ---------------------------------------------------------------------------

def curve(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    radius: float = 2.0,
    speed: float = DEFAULT_SPEED,
    direction: str = "right",
    curvature: float | None = None,
    ball_xy: np.ndarray | None = None,
    center_xy: np.ndarray | tuple[float, float] | None = None,
    lookahead: float = 0.35,
    radial_gain: float = 2.5,
    back_gain: float | None = None,
    min_offset: float = MIN_OFFSET,
    rod_mechanism: str | None = None,
) -> np.ndarray:
    """Carve a smooth, continuous curve or arc with commandable radius or curvature.

    Unlike `turn` -- which commands a discrete polygonal heading change --
    `curve` steers continuously along a curved path of radius `radius` (or
    curvature `curvature = 1 / radius`), ideal for city streets, winding roads,
    roundabouts, and S-curves.

    Parameters
    ----------
    d_hat : (2,) current travel heading vector.
    radius : curve radius in metres (default 2.0 m). If negative, inverts direction.
    speed : cruising speed in m/s (default 1.2 m/s).
    direction : "right" (clockwise) or "left" (counter-clockwise).
    curvature : signed path curvature in 1/m (e.g. +0.5 for R=2.0m right, -0.5 for R=2.0m left).
        If provided, overrides `radius` and `direction`.
    ball_xy : (2,) world-frame xy position of the ball. If provided, enables
        closed-loop arc tracking with dynamic understeer compensation.
    center_xy : optional explicit center of curvature. If None, established
        instantaneously from ball_xy and d_hat.
    lookahead : lookahead arc distance along the curve (default 0.35 m).
    radial_gain : feedback stiffness to suppress outward centripetal drift (default 2.5).
    rod_mechanism : "multi_stage" or "single_stage". Defaults to None,
        the generic calibration, like every other skill here. It used to
        default to "multi_stage", which asks for about 15 % less drive
        gain at the same commanded speed than the generic curve, so a
        caller on a single-stage build was quietly under-driven.
    """
    if curvature is not None:
        if abs(curvature) > 1e-6:
            radius = 1.0 / abs(curvature)
            direction = "right" if curvature > 0 else "left"
        else:
            # Zero curvature is straight motion
            return move(quat, dirs_body, max_extend, d_hat=d_hat, speed=speed,
                        turn=0.0, back_gain=back_gain, min_offset=min_offset,
                        rod_mechanism=rod_mechanism)

    if radius < 0:
        radius = abs(radius)
        direction = "left" if direction == "right" else "right"

    d_norm = float(np.linalg.norm(d_hat))
    d_unit = np.asarray(d_hat, dtype=np.float64) / max(d_norm, 1e-6)

    # Inward normal pointing toward center of curvature:
    if direction == "right":
        n_inward = np.array([d_unit[1], -d_unit[0]], dtype=np.float64)
        rot_dir = -1.0  # clockwise
    else:
        n_inward = np.array([-d_unit[1], d_unit[0]], dtype=np.float64)
        rot_dir = +1.0  # counter-clockwise

    if ball_xy is not None:
        p = np.asarray(ball_xy, dtype=np.float64)[:2]
        c = np.asarray(center_xy, dtype=np.float64)[:2] if center_xy is not None else p + n_inward * radius
        rel = p - c
        curr_r = float(np.linalg.norm(rel))
        th_now = float(np.arctan2(rel[1], rel[0]))

        d_th = rot_dir * (lookahead / max(radius, 0.4))
        th_target = th_now + d_th

        # Feedforward understeer offset to counteract centripetal acceleration (v^2 / R)
        lead_offset = min(0.20, 0.10 * (speed ** 2) / max(radius, 0.5))
        nominal_r = max(0.2, radius - lead_offset)
        r_target = max(0.2, nominal_r - radial_gain * (curr_r - radius))
        p_target = c + r_target * np.array([np.cos(th_target), np.sin(th_target)])

        heading_vec = p_target - p
        h_norm = float(np.linalg.norm(heading_vec))
        d_cmd = heading_vec / max(h_norm, 1e-6)
    else:
        # Open-loop heading rate steering
        d_th = rot_dir * (speed / max(radius, 0.4)) * 0.01
        d_cmd = _rotate(d_unit, d_th)

    return move(quat, dirs_body, max_extend, d_hat=d_cmd, speed=speed,
                turn=0.0, back_gain=back_gain, min_offset=min_offset,
                rod_mechanism=rod_mechanism)


# ---------------------------------------------------------------------------
# 15. straddle_gap (dual-flank outrigger locomotion across a central hole/trench)
# ---------------------------------------------------------------------------

def straddle_gap(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray | None = None,
    *,
    min_lat: float = 0.10,
    back_gain: float | None = None,
    lateral_offset: float = 0.0,
    centering_gain: float = 1.8,
) -> np.ndarray:
    """Drive forward while straddling a longitudinal hole, trench, or gap between two ledges.

    When rolling across a central gap between two parallel elevated platforms (Box 1 and Box 2),
    downward rods in the central strip point into empty air. This skill:
    1. Tucks the central underbelly (|u_lat| < min_lat + 0.02) to prevent dragging on inner lips or voids.
    2. Extends trailing rods on the left and right downward flanks simultaneously with a high-traction peristaltic wave.
    3. Keeps leading rods retracted so they never kickstand or brake against the platform surfaces.
    4. Applies active heading centering from `lateral_offset` to keep the ball locked onto the gap centerline.

    Parameters
    ----------
    quat : (4,) core orientation quaternion.
    dirs_body : (60, 3) body-frame rod unit vectors.
    max_extend : float, maximum rod extension in metres.
    d_hat : (2,) reference forward heading along the gap (default [1.0, 0.0]).
    min_lat : minimum lateral coordinate (|u_lat|) to activate flank pusher (default 0.10).
        This is a direction cosine, not a width in metres. It is what sets how
        wide the central tuck is; there is no separate gap-width parameter.
    back_gain : drive amplitude. Defaults to `STRADDLE_GAIN`.
    lateral_offset : measured y-offset from the gap centerline for active centering.
    centering_gain : proportional heading gain to steer back to gap centerline (default 1.8).

    Notes
    -----
    Unlike the rolling skills, this one takes no `speed`. It drives at a fixed
    amplitude, because the whole load rides on two narrow flank strips and the
    traction there is what decides whether the robot crosses at all. Override
    it with `back_gain` if you need to. `STRADDLE_GAIN` is 3.8, which the
    speed calibration puts near 2.7 m/s on flat ground.

    It also takes no `min_offset`. Tucked rods are commanded to a true zero,
    not to the usual retracted baseline, so nothing reaches into the void or
    catches an inner lip.
    """
    heading = np.asarray(d_hat, dtype=np.float64) if d_hat is not None else np.array([1.0, 0.0])
    gain = float(back_gain) if back_gain is not None else STRADDLE_GAIN

    # Active heading centering
    turn_angle = float(np.clip(-centering_gain * lateral_offset, -0.25, 0.25))
    d_cmd = _rotate(heading, turn_angle) if abs(turn_angle) > 1e-4 else heading

    dirs_world, u_long, u_lat, u_z = _decompose(quat, dirs_body, d_cmd)

    targets = np.zeros(len(dirs_body), dtype=np.float32)

    # Trailing flank push wave on Box 1 (Left) and Box 2 (Right)
    rear_flank = (u_long < -0.02) & (np.abs(u_lat) > min_lat) & (u_z < 0.30)
    rear_w = np.clip((-u_long) / 0.85, 0.0, 1.0) ** 0.90
    flank_w = np.clip((np.abs(u_lat) - min_lat) / 0.25, 0.0, 1.0)
    down_w = np.clip(1.0 - np.abs(u_z + 0.30) / 0.85, 0.0, 1.0)

    wave = np.clip(rear_w * flank_w * down_w * gain, 0.0, 1.0)
    targets[rear_flank] = max_extend * wave[rear_flank]

    # Explicit central void tuck (never touch hole or inner edges)
    central_void = (np.abs(u_lat) < min_lat + 0.02) & (u_z < 0.12)
    targets[central_void] = 0.0

    # Leading rods tucked to prevent braking
    targets[u_long > 0.01] = 0.0

    return targets















# ---------------------------------------------------------------------------
# 17. surface_drive (the gait, re-aimed at any surface -- floor, bank, or wall)
# ---------------------------------------------------------------------------

def surface_drive(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    normal: np.ndarray,
    along: np.ndarray,
    *,
    speed: float = DEFAULT_SPEED,
    min_offset: float = MIN_OFFSET,
    back_gain: float | None = None,
    press: float = 0.0,
    brake: float = 0.0,
    reach_cap: np.ndarray | None = None,
) -> np.ndarray:
    """Drive along a surface whose normal is `normal`, in the direction `along`.

    `move` assumes the ground is below: its stance window is built on the rod's
    z-component, so the wave always pushes down. That is one special case of a
    more general rule -- push against whatever surface is carrying you.

    Here the surface is given explicitly:

    normal : (3,) unit vector from the ball's centre TOWARD the surface.
        Floor -> (0, 0, -1). A vertical wall on the outside of a cylinder ->
        the outward radial direction. A bank at angle t -> sin(t)*radial
        - cos(t)*z.
    along : (3,) desired travel direction. Its component along `normal` is
        removed, so it may be passed loosely (e.g. tangential plus a climb
        component) and will be projected into the surface.

    With normal = (0, 0, -1) this reproduces `move` exactly: the stance window
    peaks at rods 0.35 into the surface, the leading sector is shut, and the
    lateral tuck is unchanged. The only difference is that "into the surface"
    is now a direction the caller chooses.

    `reach_cap` is a per-rod ceiling on the stroke, in metres: the extension at
    which that rod's foot just meets the surface. Without it a rod commanded
    past the surface keeps pushing, and on a wall that push throws the ball
    back off -- measured at 31 N inward against 9.5 N of centrifugal force,
    which is what wrecked every earlier attempt at this. Gravity hides the
    problem on a floor, because it re-seats the ball each time. Nothing
    re-seats it on a wall.

    `brake` is the same trick :func:`stop` uses, moved into the surface frame.
    A rod that reaches the ground *in front* of the contact point on a rolling
    body is a kickstand, not a push. Extending the leading sector therefore
    slows the robot down. 0 is off, 1 is a full kickstand.

    `press` adds a floor to the extension of rods pointing at the surface,
    independent of the travelling wave. On a wall ridden by centrifugal force
    it keeps a few feet in contact through the gaps in the wave, which is what
    stops the ball skipping off the wall between pushes.
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    n = np.asarray(normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-9)

    d = np.asarray(along, dtype=np.float64)
    d = d - float(np.dot(d, n)) * n              # project into the surface
    dn = float(np.linalg.norm(d))
    if dn < 1e-9:                                # no usable direction: just press
        u_n_only = dirs_world @ n
        targets = np.full(len(dirs_body), min_offset, dtype=np.float32)
        if press > 0.0:
            hold = np.clip((u_n_only - 0.30) / 0.70, 0.0, 1.0)
            targets = np.maximum(targets, min_offset + press * (max_extend - min_offset) * hold)
        if reach_cap is not None:
            targets = np.minimum(targets, reach_cap)
        return targets
    d = d / dn

    lat = np.cross(n, d)

    u_long = dirs_world @ d
    u_lat = dirs_world @ lat
    u_n = dirs_world @ n                         # +1 = pointing straight at the surface

    gain = float(back_gain) if back_gain is not None else gain_for_speed(speed, max_extend)

    rear = np.clip((-u_long - 0.10) / 0.90, 0.0, 1.0)
    # Stance window: peaks at rods 0.35 into the surface, i.e. at the contact
    # patch and slightly trailing -- the same shape `move` uses against z.
    stance = np.clip(1.0 - np.abs(u_n - 0.35) / 0.85, 0.0, 1.0)
    tuck = np.clip(1.0 - 1.8 * (u_lat ** 2), 0.0, 1.0)
    wave = np.clip((rear ** 1.1) * stance * gain * tuck, 0.0, 1.0)

    wave[u_long > -0.05] = 0.0                   # leading sector is a brake
    wave[u_n < -0.10] = 0.0                      # rods pointing away from the surface

    targets = min_offset + (max_extend - min_offset) * wave

    if press > 0.0:
        hold = np.clip((u_n - 0.30) / 0.70, 0.0, 1.0)
        targets = np.maximum(targets, min_offset + press * (max_extend - min_offset) * hold)

    if brake > 0.0:
        lead = np.clip((u_long - 0.10) / 0.90, 0.0, 1.0)
        foot = np.clip((u_n - 0.20) / 0.80, 0.0, 1.0)
        kick = np.clip(float(brake) * lead * foot, 0.0, 1.0)
        targets = np.maximum(targets, min_offset + (max_extend - min_offset) * kick)

    if reach_cap is not None:
        targets = np.minimum(targets, np.asarray(reach_cap, dtype=np.float64))

    return targets.astype(np.float32)
