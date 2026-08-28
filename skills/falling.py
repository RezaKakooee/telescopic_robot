"""Falling skills: stepping off a ledge and absorbing the landing.

Dropping off a platform is not the same problem as jumping. There is no
impulse to produce; the work is in leaving the edge under control, staying
compact while the ball is unsupported, and soaking up the touchdown so the
robot does not bounce away from where it meant to land.

Like the jump skills these are **phase-based**: the caller passes a `phase`
string and gets the rod targets for that single step.
"""
from __future__ import annotations

import numpy as np

from radial_sphere.geometry import quat_to_rotmat


# ---------------------------------------------------------------------------
# 12. fall_down  (step off a ledge and land under control)
# ---------------------------------------------------------------------------

def fall_down(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    *,
    phase: str = "edge",
    edge_speed: float = 0.45,
    tuck: float = 0.015,
    absorb: float = 0.075,
    drop_height: float | None = None,
    gear: float = 0.5,
    brace_front: float = 0.0,
    stance_height: float = 0.045,
    min_offset: float = 0.025,
) -> np.ndarray:
    """Drop off a ledge in the direction *d_hat*.

    Phases
    ------
    "edge"    : Creep forward at `edge_speed` m/s until the ball tips over
                the lip. A hard push here throws the robot clear of the ledge
                and it lands flat on its side instead of rolling on.
    "freefall": Landing gear. The rods underneath extend to `gear` of full
                stroke so the ball comes down on rods, not on its shell, and
                the actuators' compliance takes the hit. Everything above is
                tucked so nothing catches the lip on the way out. With
                `brace_front` > 0 the leading rods also extend part way, as a
                bumper for a wall the drop may carry the ball into.
    "absorb"  : Downward rods extend part way to a soft standoff. They act as
                a spring on touchdown and take the shock out of the landing.
                Given `drop_height`, the standoff is scaled to the impact the
                fall will actually produce: a longer drop lands harder and
                needs more travel to soak it up.
    "settle"  : Low stance on the bottom cluster, ready to drive again.

    Typical sequencing (managed by the caller, by height rather than by step
    count, since how long the drop takes depends on the ledge):

        "edge" until the core starts dropping
        -> "freefall" while it is falling
        -> "absorb" just before touchdown
        -> "settle" once it is down
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
    u_z = dirs_world[:, 2]

    targets = np.zeros(len(dirs_body), dtype=np.float32)

    if phase == "edge":
        # Same peristaltic drive as move_forward, held down to a crawl.
        rear = np.clip((-u_long - 0.10) / 0.90, 0.0, 1.0)
        down = np.clip(1.0 - abs(u_z + 0.35) / 0.85, 0.0, 1.0)
        lat = np.clip(1.0 - 1.8 * (u_lat ** 2), 0.0, 1.0)
        from .locomotion import gain_for_speed
        wave = np.clip((rear ** 1.1) * down * gain_for_speed(edge_speed) * lat,
                       0.0, 1.0)
        wave[u_long > -0.05] = 0.0
        wave[u_z > 0.10] = 0.0
        targets[:] = min_offset + (max_extend - min_offset) * wave

    elif phase == "freefall":
        targets[:] = tuck
        g = float(np.clip(gear, 0.0, 1.0)) * max_extend
        if drop_height is not None and drop_height < 0.5:
            # SHORT DROP — the ball still has forward carry from the creep and
            # may not have cleared the lip yet.  Only extend rods that point
            # both DOWNWARD and TRAILING (behind the roll direction).  A
            # leading rod that extends while the ball is still near the lip
            # touches the upper deck first and pole-vaults the ball forward,
            # overshooting the target pad.
            rear_bottom = (u_long < 0.0) & (u_z < -0.35)
            targets[rear_bottom] = g
        else:
            # TALL DROP (>= 0.5 m) or unknown height — the ball clears the lip
            # with room to spare, so the full bottom hemisphere is safe and
            # gives better cushioning.
            targets[u_z < -0.35] = g
        if brace_front > 0.0:
            front = (u_long > 0.40) & (np.abs(u_z) < 0.55)
            targets[front] = float(np.clip(brace_front, 0.0, 1.0)) * max_extend


    elif phase == "absorb":
        cushion = absorb
        if drop_height is not None:
            # Touchdown speed goes as sqrt(2 g h); scale the spring with it,
            # against the 0.25 m ledge the default was tuned on.
            ratio = float(np.sqrt(max(drop_height, 0.01) / 0.25))
            cushion = float(np.clip(absorb * ratio, 0.03, max_extend * 0.75))
        # Never pull the gear back in at the moment it is about to be used.
        cushion = max(cushion, float(np.clip(gear, 0.0, 1.0)) * max_extend * 0.6)
        targets[u_z < -0.20] = cushion
        if brace_front > 0.0:
            front = (u_long > 0.40) & (np.abs(u_z) < 0.55)
            targets[front] = float(np.clip(brace_front, 0.0, 1.0)) * max_extend

    else:  # "settle"
        targets[u_z < -0.30] = stance_height

    return targets
