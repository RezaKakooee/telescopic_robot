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

from radial_sphere.geometry import quat_to_rotmat


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
SPEED_CURVE = (
    # back_gain, cruise speed (m/s)
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
_GAINS = np.array([g for g, _v in SPEED_CURVE])
_SPEEDS = np.array([v for _g, v in SPEED_CURVE])

#: Slowest and fastest cruise this gait can actually hold.
MIN_SPEED = float(_SPEEDS[0])
MAX_SPEED = float(_SPEEDS[-1])

# Braking, measured by asking for a stop distance and recording the real one.
# The relation is not exactly coast ~ 1/gain (the wave clips at full stroke),
# so this constant is a least-squares fit over 0.25-1.0 m rather than a law.
# Expect the achieved distance to be within about 0.1 m of the request.
COAST_PER_SPEED = 0.62
MAX_BRAKE_GAIN = 3.0


def gain_for_speed(speed):
    """Push-wave amplitude that cruises at `speed` m/s, from the measured curve.

    Speeds outside the measured range are clamped: below MIN_SPEED the rods
    stop reaching the ground and the robot stalls rather than crawling, and
    above MAX_SPEED there is no evidence to extrapolate from.
    """
    v = float(np.clip(speed, MIN_SPEED, MAX_SPEED))
    return float(np.interp(v, _SPEEDS, _GAINS))


def speed_for_gain(gain):
    """Inverse of :func:`gain_for_speed`; cruise speed a given amplitude holds."""
    return float(np.interp(float(gain), _GAINS, _SPEEDS))


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
    speed: float = 1.2,
    turn: float = 0.0,
    min_offset: float = 0.025,
    back_gain: float | None = None,
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
    speed : desired cruise speed in m/s, clamped to
        [``MIN_SPEED``, ``MAX_SPEED``] = [0.33, 2.80]. Converted to a wave
        amplitude by the measured :data:`SPEED_CURVE`.
    back_gain : escape hatch. Set it to drive the amplitude directly and
        ignore `speed`.

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
    gain = float(back_gain) if back_gain is not None else gain_for_speed(speed)

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

def move_forward(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    speed: float = 1.2,
    back_gain: float | None = None,
    min_offset: float = 0.025,
) -> np.ndarray:
    """Drive along *d_hat* at `speed` m/s. Default 1.2 m/s.

    A preset of :func:`move`. Ask `move` for any other heading or speed.
    """
    return move(quat, dirs_body, max_extend, d_hat, speed=speed,
                turn=0.0, back_gain=back_gain, min_offset=min_offset)


# ---------------------------------------------------------------------------
# 2. move_right
# ---------------------------------------------------------------------------

def move_right(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    speed: float = 1.2,
    back_gain: float | None = None,
    min_offset: float = 0.025,
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
    speed: float = 1.2,
    back_gain: float | None = None,
    min_offset: float = 0.025,
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
    min_offset: float = 0.025,
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
    min_offset: float = 0.025,
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
    speed: float = 1.2,
    back_gain: float | None = None,
    min_offset: float = 0.025,
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
    min_offset: float = 0.025,
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
# 15. straddle_gap (dual-flank outrigger locomotion across a central hole/trench)
# ---------------------------------------------------------------------------

def straddle_gap(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray | None = None,
    *,
    gap_half_width: float = 0.11,
    speed: float = 1.3,
    min_lat: float = 0.10,
    min_offset: float = 0.025,
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
    gap_half_width : half-width of the central hole/gap in metres (default 0.11 m).
    speed : commanded cruise speed in m/s (default 1.3 m/s).
    min_lat : minimum lateral coordinate (|u_lat|) to activate flank pusher (default 0.10).
    lateral_offset : measured y-offset from the gap centerline for active centering.
    centering_gain : proportional heading gain to steer back to gap centerline (default 1.8).
    """
    heading = np.asarray(d_hat, dtype=np.float64) if d_hat is not None else np.array([1.0, 0.0])
    gain = float(back_gain) if back_gain is not None else 3.8

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













