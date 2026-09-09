"""Wall of Death: spiral up the inside of a drome, the way a rider does.

The ride is one equation. A body in a circle of radius ``r`` at speed ``v``
needs an inward acceleration of ``v^2 / r``. On a bank of angle ``b`` the
boards supply ``g * tan(b)`` of it for free. Balance the two and

    v^2 = g * r * tan(b)

That single line decides everything: how high the robot can ride, how wide the
arena may be, and how fast it must go to get there. Faster means further out,
and further out on a bowl means higher. Riders climb by opening the throttle,
not by steering upward, and so does this.

**Why the wall is a bowl and not a cylinder.** A vertical wall was tried first
and it cannot work. Friction is the only thing that can hold a body on a
vertical wall, and the friction that holds a *rolling* ball also spins it, so
the ball rolls straight down the wall. Measured over wall radii from 0.70 m to
1.80 m and speeds up to twice the friction limit, every attempt slid down at
about the same rate. A bank needs no friction at all: the boards push inward by
themselves. So the arena is a drome bowl, steepening outward, with the vertical
wall left above the rim where it belongs.

**The climb is a spiral, not a jump.** The robot accelerates on the flat floor,
then widens its circle a little at a time. Each new radius is a slightly
steeper bank, which holds a slightly higher speed, which pays for the next
radius out. Reaching for the bank in one move fails: the robot arrives too slow
for the angle and slides straight back down.

Three pieces:

* :class:`Bowl` -- the arena's shape, and the one question worth asking it:
  what radius will this speed ride?
* :func:`advance_radius` -- open the spiral, but only as fast as the measured
  speed has earned.
* :func:`wall_of_death` -- one control step: drive along the boards, steer
  gently toward the commanded radius.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from radial_sphere.geometry import quat_to_rotmat

from .locomotion import surface_drive, speed_range


G = 9.81

#: Distance from the core centre to a foot's outer surface at zero stroke:
#: core radius 0.150 + sleeve stub 0.006 + tip gap 0.004 + foot radius 0.013.
#: A rod extended by ``e`` reaches ``FOOT_BASE + e``.
FOOT_BASE = 0.173

#: Extra stroke a rod may be commanded past the wall, in metres. Actuator gain
#: is 900 N/m, so 8 mm is about 7 N of press.
TOUCH_MARGIN = 0.008

#: Sideways acceleration the gait can make by itself, in m/s^2, measured on
#: flat ground. It is small, and that is why a flat floor cannot be used to
#: reach high speed: at 5 m/s the tightest flat circle is about 5 m across.
LATERAL_LIMIT = 4.5

#: How far above the equilibrium speed counts as ready for the next radius.
READY_MARGIN = 1.06


# ---------------------------------------------------------------------------
# The arena
# ---------------------------------------------------------------------------

@dataclass
class Bowl:
    """The drome's cross-section: radius and height, from the middle outward.

    Built straight from the arena definition, so the controller and the MJCF
    can never disagree about the shape.
    """

    r: np.ndarray
    z: np.ndarray
    floor_r: float = 0.0
    lateral_limit: float = LATERAL_LIMIT
    #: Radius where the ride line ends. Past it the boards curl up to meet the
    #: vertical wall, and nothing rides that: a robot pushed out there runs
    #: wide into boards far too steep for its speed and loses the lot.
    ride_limit: float = 0.0

    tan_bank: np.ndarray = field(init=False)

    def __post_init__(self):
        self.r = np.asarray(self.r, dtype=np.float64)
        self.z = np.asarray(self.z, dtype=np.float64)
        seg = np.diff(self.z) / np.maximum(np.diff(self.r), 1e-9)
        # Slope sampled at the profile points: the mean of the two segments
        # meeting there, so it varies smoothly instead of stepping at a ring.
        node = np.empty_like(self.r)
        node[0], node[-1] = seg[0], seg[-1]
        node[1:-1] = 0.5 * (seg[1:] + seg[:-1])
        self.tan_bank = node
        if self.ride_limit <= 0.0:
            self.ride_limit = float(self.r[-1])

    @classmethod
    def from_motordrome(cls, md, lateral_limit: float = LATERAL_LIMIT) -> "Bowl":
        """Build from one row of ``scenario.motordromes``."""
        floor_r, rim_r = float(md[2]), float(md[3])
        profile = md[7] if len(md) > 7 and md[7] is not None else None
        if profile is None:
            profile = [(floor_r, 0.0), (rim_r, float(md[4]))]
        r = [0.0] + [float(a) for a, _b in profile]
        z = [0.0] + [float(b) for _a, b in profile]
        ride_limit = float(md[11]) if len(md) > 11 and md[11] else rim_r
        return cls(np.asarray(r), np.asarray(z), floor_r=floor_r,
                   lateral_limit=lateral_limit, ride_limit=ride_limit)

    @property
    def rim_r(self) -> float:
        return float(self.r[-1])

    @property
    def rim_z(self) -> float:
        return float(self.z[-1])

    def slope_at(self, rr: float) -> float:
        """tan(bank) at radius `rr`. Zero on the flat floor."""
        return float(np.interp(rr, self.r, self.tan_bank))

    def height_at(self, rr: float) -> float:
        return float(np.interp(rr, self.r, self.z))

    def hold_speed(self, rr: float) -> float:
        """Speed the bank alone will carry at radius `rr`: sqrt(g r tan b)."""
        return float(np.sqrt(max(G * rr * self.slope_at(rr), 0.0)))

    def target_speed(self, rr: float, reserve: float = 0.5) -> float:
        """Speed to ask for while circling at radius `rr`.

        Not the robot's top speed. Going faster than the circle can hold is
        not an advantage, it is a crash: the robot runs wide, meets the steep
        boards at an angle it cannot ride, and loses everything it had. So the
        throttle follows the radius rather than the other way round.

        The circle can supply ``g tan(b)`` from the bank plus a share of what
        the gait itself can make sideways. `reserve` is that share, kept below
        1 so there is always something left for steering.
        """
        budget = G * self.slope_at(rr) + reserve * self.lateral_limit
        want = float(np.sqrt(max(rr * budget, 0.0)))
        # Never ask for more than the top of the ride line can carry on the
        # bank alone. The reserve is there to help the robot corner, not to
        # push it past the last board it can hold.
        return min(want, self.hold_speed(self.ride_limit))

    def ride_radius(self, speed: float) -> float:
        """Widest circle this speed can hold, in metres.

        The test is two-sided, and both sides matter. Too fast for the bank
        and the robot runs wide off the top; too slow and it slides down the
        inside. What the gait itself can add or take away is
        ``lateral_limit``, so a radius is ridable when

            | v^2 / r  -  g tan(b) |  <=  lateral_limit

        On the flat floor ``tan(b)`` is zero and this reduces to the ordinary
        cornering limit. The answer is the widest radius that passes.
        """
        v2 = float(speed) ** 2
        need = v2 / np.maximum(self.r, 1e-6)
        ok = np.nonzero((np.abs(need - G * self.tan_bank) <= self.lateral_limit)
                        & (self.r <= self.ride_limit + 1e-9))[0]
        return float(self.r[ok[-1]]) if len(ok) else float(self.r[0])

    def next_radius_ready(self, speed: float, r_cmd: float, step: float) -> bool:
        """Is the robot fast enough to hold ``r_cmd + step``?"""
        rr = min(r_cmd + step, self.rim_r)
        need = self.hold_speed(rr)
        if need <= 1e-6:                        # still on the flat floor
            return speed ** 2 <= rr * self.lateral_limit
        return speed >= READY_MARGIN * need


def advance_radius(
    r_cmd: float,
    ball_r: float,
    speed: float,
    bowl: Bowl,
    *,
    dt: float = 0.01,
    open_rate: float = 0.10,
    close_rate: float = 0.35,
    lead: float = 0.25,
    caught_up: float = 0.92,
) -> float:
    """Open the spiral by one step, or pull it back in.

    The target is :meth:`Bowl.ride_radius` of the *measured* speed: the widest
    circle the robot can currently hold. Nothing here counts steps, and nothing
    decides in advance where the robot should be at a given time. It goes as
    far out as it has earned, and it gives radius back when it slows.

    `open_rate` is metres of radius per second, and slow is the point. Letting
    the robot pick its own radius makes it lunge at the bank, arrive too slow
    for the angle, and slide straight back down. That was the failure in every
    earlier attempt at this arena.

    `lead` keeps the command from running more than that far outside the robot
    itself, so the steering always has a reachable target. `caught_up` is the
    fraction of :meth:`Bowl.target_speed` the robot must actually be doing
    before the circle is allowed to open at all.
    """
    if speed < caught_up * bowl.target_speed(r_cmd):
        # Still accelerating into the circle it already has. Widening now
        # would ask for a bank the robot has not paid for yet.
        target = min(r_cmd, bowl.ride_radius(speed))
    else:
        target = min(bowl.ride_radius(speed), ball_r + lead)
    delta = float(np.clip(target - r_cmd, -close_rate * dt, open_rate * dt))
    return float(np.clip(r_cmd + delta, bowl.r[1], bowl.ride_limit))


def descend_radius(
    r_cmd: float,
    ball_r: float,
    speed: float,
    bowl: Bowl,
    *,
    dt: float = 0.01,
    close_rate: float = 0.09,
    aim: float | None = None,
    lag_band: float = 0.35,
) -> float:
    """Wind the spiral back in, for a controlled way down.

    Coming down is not the climb run backwards. On the way up the bank is the
    thing that has to be earned; on the way down it is the speed that has to be
    given away, and a bank only lets go of speed as fast as the robot can shed
    it. Wind in faster than that and the robot is left going too fast for the
    circle it has been handed, so it runs wide and climbs again instead of
    descending.

    So the rate backs off whenever the robot is lagging behind the commanded
    circle, which is exactly the sign that it has not shed the speed yet.
    """
    target = bowl.floor_r if aim is None else float(aim)
    lag = ball_r - r_cmd
    rate = close_rate if lag < lag_band else close_rate * 0.25
    return float(max(target, r_cmd - rate * dt))


# ---------------------------------------------------------------------------
# Reading the surface
# ---------------------------------------------------------------------------

def surface_frame(model, data, core_pos, *, min_force: float = 1e-3):
    """Where the surface is: ``(normal, distance)``, or ``(None, None)`` in air.

    Read from the live contact list rather than from arena geometry, so the
    answer swings smoothly across the joins between floor, bowl and wall with
    no zone test anywhere in the controller.

    Use MuJoCo's own contact normal, not the direction from the core to the
    contact point. Those two are the same only for a plain sphere. This robot
    stands on rods, so the contact sits far off-axis: a ball resting on a rod
    tilted 45 degrees puts its contact 45 degrees off vertical, and the
    core-to-contact direction reports a 45-degree slope on flat ground. That
    error fed straight into the gait and was what kept the earlier controller
    from ever building speed.

    MuJoCo's ``contact.frame`` holds the normal in its first three numbers,
    pointing from ``geom1`` toward ``geom2``. Flip it where needed so it always
    points from the robot toward the surface. Anything whose body root is the
    world is scenery; the robot is the only free body.
    """
    import mujoco

    acc = np.zeros(3)
    hits = []
    force = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        con = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force)
        w = abs(float(force[0]))
        if w < min_force:
            continue
        n_i = np.array(con.frame[0:3], dtype=np.float64)
        if model.body_rootid[model.geom_bodyid[con.geom1]] == 0:
            n_i = -n_i                       # geom1 was the scenery; flip
        acc += n_i * w
        hits.append((np.asarray(con.pos, dtype=np.float64) - core_pos, w))

    n = float(np.linalg.norm(acc))
    if n < 1e-9 or not hits:
        return None, None
    n_hat = acc / n
    wsum = sum(w for _v, w in hits)
    dist = sum(w * float(np.dot(v, n_hat)) for v, w in hits) / max(wsum, 1e-9)
    return n_hat, float(dist)


def surface_normal(model, data, core_pos, *, min_force: float = 1e-3):
    """Just the normal from :func:`surface_frame`."""
    n_hat, _d = surface_frame(model, data, core_pos, min_force=min_force)
    return n_hat


def reach_caps(core_pos, dirs_world, max_extend, *, wall_radius,
               margin=TOUCH_MARGIN, foot_base=FOOT_BASE):
    """Stroke at which each rod's foot would meet the cylinder wall.

    Only the wall is capped, and only so no rod is commanded through it. Do not
    cap against the floor or the bowl: pushing past the ground is how the gait
    makes thrust, and capping there locks the robot solid.
    """
    caps = np.full(len(dirs_world), float(max_extend), dtype=np.float64)
    x, y = float(core_pos[0]), float(core_pos[1])
    ux, uy = dirs_world[:, 0], dirs_world[:, 1]
    a = ux * ux + uy * uy
    b = 2.0 * (x * ux + y * uy)
    c = x * x + y * y - wall_radius * wall_radius
    disc = b * b - 4.0 * a * c
    ok = (a > 1e-9) & (disc > 0.0)
    L = np.full(len(dirs_world), 1e9)
    L[ok] = (-b[ok] + np.sqrt(disc[ok])) / (2.0 * a[ok])
    return np.minimum(caps, np.clip(L - foot_base + margin, 0.0, max_extend))


def grip_margin(speed: float, orbit_r: float, mu: float) -> float:
    """Traction in hand on a vertical wall. 1.0 is the point of sliding."""
    if orbit_r < 1e-6:
        return 0.0
    return float(mu * speed * speed / (orbit_r * G))


# ---------------------------------------------------------------------------
# The control step
# ---------------------------------------------------------------------------

def wall_of_death(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    ball_pos: np.ndarray,
    lin_vel: np.ndarray,
    *,
    r_cmd: float,
    bowl: Bowl,
    normal: np.ndarray | None = None,
    wall_radius: float | None = None,
    ccw: bool | None = None,
    steer_gain: float = 0.55,
    max_steer: float = 0.35,
    brake_gain: float = 0.0,
    brake_band: float = 0.80,
    speed: float | None = None,
    min_offset: float = 0.025,
    touch_margin: float = TOUCH_MARGIN,
):
    """One step of the ride. Returns ``(targets, info)``.

    Parameters
    ----------
    r_cmd : the circle the robot is being asked to hold, in metres. Move it
        with :func:`advance_radius`; never from a step count.
    bowl : the arena shape, from :meth:`Bowl.from_motordrome`.
    normal : measured surface normal from :func:`surface_frame`. None means
        airborne, and the floor normal is used instead.
    steer_gain, max_steer : how hard to steer toward `r_cmd`, as a fraction of
        the travel direction. Both are small on purpose. Hard steering on a
        bank scrubs off the speed that is holding the robot up, and losing
        speed on a bank means sliding down it.
    speed : cruise speed to ask the gait for, m/s. Left as None it comes from
        :meth:`Bowl.target_speed` of whichever is wider, the commanded radius
        or the robot's own, which is what keeps the throttle and the circle in
        step. Full throttle at a small radius just throws it into the boards.

    Notes
    -----
    The travel direction is the tangent plus a small radial nudge, and
    :func:`~skills.locomotion.surface_drive` projects it into whatever surface
    the robot is standing on. So one command means "circle, and drift outward a
    little" on the flat floor, on the bowl and on the wall alike, with no zone
    test anywhere in the controller.
    """
    pos = np.asarray(ball_pos, dtype=np.float64)
    vel = np.asarray(lin_vel, dtype=np.float64)

    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    r = float(np.hypot(x, y))
    r_hat = np.array([1.0, 0.0, 0.0]) if r < 1e-4 else np.array([x / r, y / r, 0.0])
    t_hat = np.array([-r_hat[1], r_hat[0], 0.0])

    v_t = float(np.dot(vel, t_hat))
    go_ccw = (v_t >= 0.0 if abs(v_t) > 0.15 else True) if ccw is None else bool(ccw)
    if not go_ccw:
        t_hat = -t_hat
        v_t = -v_t

    if normal is not None:
        n_hat = np.asarray(normal, dtype=np.float64)
        nn = float(np.linalg.norm(n_hat))
        n_hat = n_hat / nn if nn > 1e-9 else np.array([0.0, 0.0, -1.0])
    else:
        n_hat = np.array([0.0, 0.0, -1.0])

    # Steer toward the commanded circle. Gently: this is the only place the
    # robot chooses its height, and every metre of radius is paid for in speed.
    steer = float(np.clip(steer_gain * (r_cmd - r), -max_steer, max_steer))
    along = t_hat + steer * r_hat

    # Throttle for the *commanded* circle, not the robot's own. Using its own
    # was tried and is positive feedback: drifting outward raises the throttle,
    # which drifts it further out. Measured, it never settled once in 150 s,
    # against a steady ride with the commanded radius.
    throttle_r = float(r_cmd)
    v_lo, v_hi = speed_range(max_extend)
    v_cmd = bowl.target_speed(throttle_r) if speed is None else float(speed)
    v_cmd = float(np.clip(v_cmd, v_lo, v_hi))

    # On the way down the commanded speed keeps falling, and the robot has to
    # actually give the speed away or it just runs wide and climbs again.
    brake = 0.0
    if brake_gain > 0.0:
        excess = (float(np.linalg.norm(vel)) - v_cmd) / max(brake_band, 1e-6)
        brake = float(np.clip(brake_gain * excess, 0.0, 1.0))

    caps = None
    if wall_radius is not None:
        dirs_world = np.asarray(dirs_body) @ quat_to_rotmat(quat).T
        caps = reach_caps(pos, dirs_world, max_extend,
                          wall_radius=wall_radius, margin=touch_margin)

    targets = surface_drive(
        quat, dirs_body, max_extend, n_hat, along,
        speed=v_cmd, min_offset=min_offset, brake=brake, reach_cap=caps,
    )

    info = {
        "r": r,
        "z": z,
        "r_cmd": float(r_cmd),
        "v_t": abs(v_t),
        "speed": float(np.linalg.norm(vel)),
        "hold_speed": bowl.hold_speed(r),
        "bank_deg": float(np.degrees(np.arctan(bowl.slope_at(r)))),
        "steer": steer,
        "throttle_r": throttle_r,
        "brake": brake,
        "v_cmd": v_cmd,
        "ccw": go_ccw,
        "airborne": normal is None,
    }
    return targets.astype(np.float32), info
