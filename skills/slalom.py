"""Training Cones / Slalom Weave: navigate between linear or curved obstacle cones.

Supports arbitrary 2D cone layouts (straight lines, meandering S-curves, circular arcs)
and non-uniform, uneven cone spacings. The controller constructs local Frenet frames
(tangent and normal unit vectors) along the cone sequence, positions alternating left/right
gates perpendicular to the local curve tangent, and executes high-authority pure-pursuit
steering to carve cleanly between every cone with zero collisions.
"""
from __future__ import annotations

import numpy as np

from .locomotion import move, stop


def slalom(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    *,
    ball_xy: np.ndarray,
    lin_vel: np.ndarray | None = None,
    cones: np.ndarray | list,
    speed: float = 1.1,
    lateral_offset: float = 0.80,
    lead_distance: float = 0.40,
    lateral_gain: float = 5.0,
    min_offset: float = 0.025,
    back_gain: float | None = None,
) -> np.ndarray:
    """Steer through arbitrary 2D linear or curvy cones in an alternating slalom weave.

    Parameters
    ----------
    quat : (4,) body orientation quaternion [w, x, y, z].
    dirs_body : (60, 3) rod direction vectors in body frame.
    max_extend : maximum rod extension (m).
    ball_xy : (2,) current world (x, y) position of the ball center.
    lin_vel : (3,) linear velocity vector (optional).
    cones : (N, 3) or (N, 2) cone positions (x, y, [radius]).
    speed : target forward speed in m/s (default 1.1).
    lateral_offset : lateral weave amplitude perpendicular to track (default 0.80 m).
    lead_distance : lookahead distance for turn-reversal anticipation (default 0.40 m).
    lateral_gain : lateral error amplification for high turning authority (default 5.0).
    """
    p = np.asarray(ball_xy, dtype=np.float64)[:2]
    c_arr = np.asarray(cones, dtype=np.float64)[:, :2]
    n_cones = len(c_arr)

    if n_cones == 0:
        return move(quat, dirs_body, max_extend, np.array([1.0, 0.0]), speed=speed)

    # 1. Compute local Frenet tangent and normal for each cone
    tangents = np.zeros((n_cones, 2), dtype=np.float64)
    normals = np.zeros((n_cones, 2), dtype=np.float64)

    for i in range(n_cones):
        if n_cones == 1:
            t = np.array([1.0, 0.0])
        elif i == 0:
            t = c_arr[1] - c_arr[0]
        elif i == n_cones - 1:
            t = c_arr[-1] - c_arr[-2]
        else:
            t = c_arr[i + 1] - c_arr[i - 1]
        t_norm = float(np.linalg.norm(t))
        t = t / max(t_norm, 1e-6)
        n = np.array([-t[1], t[0]])  # Left normal (90 deg CCW)
        tangents[i] = t
        normals[i] = n

    # 2. Compute alternating 2D gate coordinates
    gates = np.zeros((n_cones, 2), dtype=np.float64)
    for i in range(n_cones):
        sign = +1.0 if (i % 2 == 0) else -1.0
        gates[i] = c_arr[i] + sign * lateral_offset * normals[i]

    # 3. Find active target gate along the sequence
    target_pos = c_arr[-1] + 2.5 * tangents[-1]
    active_t = tangents[-1]
    active_n = normals[-1]

    for i in range(n_cones):
        # Along-track projection relative to cone i
        s_i = float(np.dot(p - c_arr[i], tangents[i]))
        if s_i < -lead_distance:
            # Approaching cone i
            target_pos = gates[i]
            active_t = tangents[i]
            active_n = normals[i]
            break
        elif s_i < 0.35:
            # Passing cone i apex: pull slightly along track while holding width
            target_pos = gates[i] + 0.40 * tangents[i]
            active_t = tangents[i]
            active_n = normals[i]
            break

    # 4. Decompose error into along-track (tangent) and cross-track (normal) components
    vec_err = target_pos - p
    v_tangent = float(np.dot(vec_err, active_t))
    v_normal = float(np.dot(vec_err, active_n))

    # Apply amplified lateral steering authority
    cmd_t = max(v_tangent, 0.35)
    cmd_n = v_normal * lateral_gain

    heading_vec = cmd_t * active_t + cmd_n * active_n
    d_hat = heading_vec / max(float(np.linalg.norm(heading_vec)), 1e-6)

    return move(
        quat, dirs_body, max_extend,
        d_hat=d_hat,
        speed=speed,
        min_offset=min_offset,
        back_gain=back_gain,
    )


# Alias
training_cones = slalom
curved_slalom = slalom
