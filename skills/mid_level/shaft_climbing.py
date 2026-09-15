"""Wall-jump zig-zag up between two walls and climb out onto a wall top.

Two facing walls. The ball gets to one wall, pushes off it toward the
other, pushes again, and so on up the gap. Each push lifts it. Once it is
above a lip it flies over and lands on the wall top, where it brakes and
stands.

Two skills share this machine. They differ in the gap:

* `zigzag_climb` -- a **chimney**, 0.40 m wide, a little wider than the
  ball. The ball is never far from a wall. It jumps straight up off the
  floor and the zig-zag starts in the air. The walls differ in height and
  the exit is a special push off the tall wall, over the lower lip.
* `wall_jump_climb` -- a **wide gap**, about 1 m. The ball flies freely
  between pushes. It has to jump *at* the first wall from the floor, and
  each push has to throw it right across. The walls are the same height
  and it lands on whichever top it clears first. This needs a stiffer rod
  than the default robot; see `configs/rl/wall_jump.yaml`.

The rod physics is `chimney_climb` (the wall push) and `jump_to` (the
aimed floor jump). This module is the sequencer around them: which phase
to run this step, decided from position and velocity, never from a step
count alone. The chimney machine can also stop part way up, clamp both
walls and slide back down (`Shaft.target_z`).

Phases, in the order they usually run:

===========  ==================================================================
launch       (chimney) Straight up off the floor. Down there the wall-push
             rods point at the floor, not the wall.
crouch       (wide) Rods in, ready to fire.
takeoff      (wide) `jump_to` at the nearer wall: sideways and up.
fly          Every rod tucked. Crossing the gap.
push         Fire the rods that point into the wall and downward. The
             reaction throws the ball up and across to the other wall.
exit         (chimney) The final push, from just above the lower lip, with
             a wider rod band for the most lift.
fly_out      (chimney) Over the lower wall's top.
land         Landing cage underneath, then ...
brake        ... `stop` against the sideways carry before the far edge.
stand        Stopped on the top. The climb is over.
recentre     Drifted along the gap toward an open end: roll back to the
             middle before launching again.
settle       Wait at the middle until the ball is still, then launch.
hold         (target_z mode) Clamp both walls and hang.
descend      (target_z mode) Clamp extension servoed on vz: slide down.
done         Terminal. Landed somewhere it cannot continue from.
===========  ==================================================================
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..low_level.climbing import chimney_climb
from ..low_level.jumping import jump_to
from ..low_level.locomotion import move, stop


#: Order the phases run in, plus the side branches.
PHASES = (
    "launch", "crouch", "takeoff", "fly", "push", "exit", "fly_out", "land",
    "brake", "stand", "recentre", "settle", "hold", "descend", "done",
)


@dataclass(frozen=True)
class ZigzagTuning:
    """Every number the phase machine uses, declared once.

    :data:`TUNING` was measured on the 0.40 m chimney with the default
    robot. :data:`WIDE_TUNING` on the 1.0 m gap with the stiffer rod.
    """
    #: Fraction of full stroke the wall-push rods fire.
    push_frac: float = 1.0
    #: Distance from the core to the wall face at which `fly` turns into
    #: `push`. The rods reach about 0.30 m; firing them as the ball
    #: arrives, not before, is what makes the push a push and not a bounce.
    push_reach: float = 0.18
    #: `push_lat`, `push_z_lo`, `push_z_hi` for the exit push. The normal
    #: band measured the most lift; steeper bands gave less.
    exit_band: tuple[float, float, float] = (0.45, 0.20, 0.85)
    #: Exit once the core is this far above the lower lip. Negative means
    #: from below it: the wide gap eases off early so the ball just clears
    #: the lip instead of flying far over the top.
    exit_from: float = 0.02
    #: Fraction of full stroke the exit push fires.
    exit_frac: float = 1.0
    #: `push` ends once the ball leaves the wall this fast (m/s) ...
    push_leave_speed: float = 0.40
    #: ... or after this many steps regardless.
    push_max_steps: int = 25
    #: `exit` ends once the ball leaves the wall this fast (m/s) ...
    exit_leave_speed: float = 0.25
    #: ... or after this many steps regardless.
    exit_max_steps: int = 14
    #: Back on the floor: below this height, slower than this sideways,
    #: after this many fly steps, relaunch.
    floor_z: float = 0.30
    floor_v_lat: float = 0.15
    floor_fly_steps: int = 20
    #: Drift along the gap that needs a recentre before relaunching.
    drift_limit: float = 0.12
    #: Recentre: roll at this speed until this close to the middle, then
    #: stop and wait this many steps for the ball to settle.
    recentre_speed: float = 0.45
    centre_tol: float = 0.05
    centre_v_tol: float = 0.10
    settle_steps: int = 10
    #: A launch, push or exit runs at least this many steps before its
    #: leave-speed test counts.
    burst_min_steps: int = 4
    #: Straight-up `launch` ends after this many steps, or once the upward
    #: speed has dropped below `launch_v_up` above `launch_z`.
    launch_max_steps: int = 14
    launch_v_up: float = 0.20
    launch_z: float = 0.35
    #: Aimed launch (wide gap): sideways take-off speed toward the nearer
    #: wall, m/s. Zero selects the straight-up chimney launch instead.
    launch_lateral: float = 0.0
    #: Aimed launch: upward take-off speed, and the burn ends once the
    #: measured vz reaches this fraction of it, or after this many steps.
    launch_up: float = 3.0
    takeoff_done: float = 0.90
    takeoff_max_steps: int = 30
    crouch_steps: int = 20
    #: Aimed launch: fraction of the rod stroke the take-off may use. A
    #: stiff rod at full stroke throws the ball far past the target.
    launch_power: float = 1.0
    #: Over a wall: land once this far past the lip and this far below the
    #: top plus a margin, falling.
    over_lip: float = 0.02
    land_below: float = 0.35
    #: Land vs brake: brake once above the top by this much, falling slower.
    brake_above: float = 0.10
    brake_vz: float = 0.60
    brake_stop_distance: float = 0.12
    #: Settled on the top only this far in from either edge.
    lip_margin: float = 0.20
    #: Land phase length, and what counts as settled on top at its end.
    #: While the ball is still on the top but moving, the landing goes on
    #: past this; the timeout only decides where it went if it left.
    land_steps: int = 120
    settled_v: float = 0.15
    #: Fraction of stroke the landing gear opens to underneath the ball.
    land_gear: float = 0.50
    top_min: float = 0.10
    top_max: float = 0.55
    #: Stand this many steps after arriving, then `done`.
    stand_steps: int = 60
    #: target_z mode: hang this long, then descend at this vz with this
    #: clamp gain and this minimum clamp extension.
    hold_steps: int = 150
    descent_vz: float = -0.40
    descent_gain: float = 0.04
    clamp_min: float = 0.02
    landed_z: float = 0.26
    landed_vz: float = 0.30
    #: Open the landing gear on the way down below this height.
    gear_z: float = 0.45

    @property
    def launch_phase(self) -> str:
        """The phase a climb, or a relaunch, starts from."""
        return "crouch" if self.launch_lateral > 0.0 else "launch"


TUNING = ZigzagTuning()

#: The wide gap. The push starts a little earlier because the ball arrives
#: faster; the floor jump is aimed at the wall; the top is wide enough to
#: brake on, so the brake gets a longer stopping distance.
WIDE_TUNING = ZigzagTuning(
    push_reach=0.25,
    launch_lateral=1.5,
    launch_up=2.0,
    launch_power=0.6,
    exit_band=(0.45, 0.20, 0.85),
    exit_from=-1.0,
    exit_frac=0.7,
    land_below=1.5,
    land_gear=0.25,
    brake_stop_distance=0.30,
    land_steps=160,
)


@dataclass(frozen=True)
class Shaft:
    """Where the walls are, and what the climb is for.

    `axis` points across the gap, from one wall to the other. `along` in
    the state is measured on the perpendicular. Either `top` (climb out) or
    `target_z` (hold and descend) should be set.
    """
    #: (2,) direction across the gap, wall to wall.
    axis: tuple[float, float] = (0.0, 1.0)
    #: Distance from the gap's centre line to each wall face.
    half_width: float = 0.20
    #: Height of the wall top the ball climbs out onto.
    top: float | None = None
    #: Lateral span of that wall top, as distances along `axis` from the
    #: gap centre, (near edge, far edge).
    box_lat: tuple[float, float] | None = None
    #: Which side of the gap the exit wall is on, +1 or -1 along `axis`.
    #: 0 means both walls are `top` high and either top will do.
    low_sign: int = +1
    #: Hold-and-descend mode: clamp both walls at this height instead.
    target_z: float | None = None

    @property
    def unit_axis(self) -> np.ndarray:
        a = np.asarray(self.axis, dtype=np.float64)
        return a / max(float(np.linalg.norm(a)), 1e-9)

    @property
    def unit_along(self) -> np.ndarray:
        # Same handedness as `chimney_climb`'s `x_off`: +x when the axis is +y.
        a = self.unit_axis
        return np.array([a[1], -a[0]])

    def over_top(self, lat: float, margin: float = 0.0) -> bool:
        """True when `lat` is over the exit wall's top, `margin` in from its edges."""
        if self.box_lat is None:
            return False
        s = abs(lat) if self.low_sign == 0 else self.low_sign * lat
        return self.box_lat[0] + margin < s < self.box_lat[1] - margin


def shaft_from_boxes(steps, *, axis=(0.0, 1.0)) -> Shaft:
    """Build a :class:`Shaft` from the two wall boxes of a chimney scenario.

    `steps` rows are ``[x, y, hx, hy, height]``. The lower box is the exit;
    two boxes of one height make either top the exit.
    """
    boxes = np.asarray(steps, dtype=float).reshape(-1, 5)
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-9)
    low = boxes[int(np.argmin(boxes[:, 4]))]
    centre = float(low[0:2] @ a)
    half = float(abs(low[2] * a[0]) + abs(low[3] * a[1]))
    equal = bool(np.allclose(boxes[:, 4], boxes[0, 4]))
    return Shaft(axis=tuple(a), half_width=abs(centre) - half, top=float(low[4]),
                 box_lat=(abs(centre) - half, abs(centre) + half),
                 low_sign=0 if equal else (int(np.sign(centre)) or +1))


@dataclass(frozen=True)
class ZigzagState:
    """The phase machine's memory between steps.

    `phase`, `side` and `timer` are what runs this step. `successor` is the
    phase this one chose on its last step; it takes over on the next call.
    That one-step hand-over is deliberate: every threshold in
    :class:`ZigzagTuning` was measured with it, and without it the climb
    stalls at the floor.
    """
    phase: str = "launch"
    #: +1 pushes off the wall on the +axis side, -1 the other. In `fly` it
    #: is the wall just left, so the ball is heading for the other one.
    side: int = +1
    #: Steps run so far in the current phase, counting this one.
    timer: int = 0
    #: Descend servo: current clamp extension (m).
    clamp_ext: float = 0.0
    #: Height at which the clamp closed (hold creep is measured from it).
    hold_z: float = 0.0
    #: (phase, side, timer) to run next step, chosen by this step's exit test.
    successor: tuple[str, int, int] | None = None


def initial_state(tuning: ZigzagTuning = TUNING) -> ZigzagState:
    """The state a climb starts from: on the floor, about to launch."""
    return ZigzagState(phase=tuning.launch_phase)


def on_wall_top(pos, shaft: Shaft, *, tuning: ZigzagTuning = TUNING) -> bool:
    """True when `pos` sits on the exit wall's top."""
    if shaft.top is None:
        return False
    lat = float(np.asarray(pos[:2]) @ shaft.unit_axis)
    z = float(pos[2])
    return (shaft.top + tuning.top_min < z < shaft.top + tuning.top_max
            and shaft.over_top(lat, tuning.lip_margin))


def next_phase(
    state: ZigzagState,
    *,
    pos: np.ndarray,
    vel: np.ndarray,
    shaft: Shaft,
    max_extend: float,
    tuning: ZigzagTuning = TUNING,
) -> ZigzagState:
    """The state to run this step, decided from position and velocity.

    Call once per step, before the skill, with the current position and
    velocity. Three things happen, in this order:

    1. The successor the previous step chose takes over.
    2. Arrivals take effect at once: the target height (`hold`), the lip
       (`exit`), the wall top (`land`). So do the two per-step choices,
       `land`/`brake` and `recentre`/`settle`.
    3. The phase about to run scores its own exit test on the same state.
       The phase it picks is stored as `successor` and runs next step.

    Never advances on a step count alone: every timer is paired with a
    geometric condition, or is a timeout on one.
    """
    t = tuning
    axis, along_dir = shaft.unit_axis, shaft.unit_along
    p, v = np.asarray(pos, dtype=np.float64), np.asarray(vel, dtype=np.float64)
    lat, along, z = float(p[:2] @ axis), float(p[:2] @ along_dir), float(p[2])
    v_lat, v_along, vz = float(v[:2] @ axis), float(v[:2] @ along_dir), float(v[2])
    top, low = shaft.top, shaft.low_sign
    reach_lat = shaft.half_width - t.push_reach     # |lat| at which the push starts
    over = (abs(lat) if low == 0 else low * lat) > shaft.half_width + t.over_lip

    phase, side, timer = state.successor or (state.phase, state.side, state.timer)
    clamp_ext, hold_z = state.clamp_ext, state.hold_z

    # Arrivals and per-step choices, taking effect now.
    if shaft.target_z is not None and phase in ("push", "fly") and z >= shaft.target_z:
        phase, timer, hold_z = "hold", 0, z
    # Exit: a push off the TALL wall that has not started yet, from above the lip.
    if (top is not None and phase == "push" and timer == 0
            and (low == 0 or side == -low) and z >= top + t.exit_from):
        phase = "exit"
    # Coming down over a wall top.
    if (top is not None and phase in ("exit", "fly_out", "fly", "push")
            and over and z < top + t.land_below and vz < 0):
        phase, timer = "land", 0
    if phase in ("land", "brake"):
        # Same landing, one timer: cage while dropping, brake once on the top.
        phase = "brake" if (z > top + t.brake_above and abs(vz) < t.brake_vz) else "land"
    if phase in ("recentre", "settle"):
        # Same return trip, one timer: roll until close, then stop and wait.
        phase = "settle" if abs(along) < t.centre_tol else "recentre"
    if phase == "descend":
        # Friction servo: loosen while falling slower than wanted, tighten
        # while faster. Extension is the friction knob.
        clamp_ext = float(np.clip(clamp_ext - t.descent_gain * (vz - t.descent_vz),
                                  t.clamp_min, max_extend))
    if phase == "crouch" and timer == 0:
        # Aim at the nearer wall. `side` is the wall the ball is leaving.
        side = -(int(np.sign(lat)) or +1)

    run = (phase, side, timer + 1)
    timer += 1

    # The phase's own exit test. Its pick runs next step.
    if phase == "launch":
        if timer > t.launch_max_steps or (
                timer > t.burst_min_steps and vz < t.launch_v_up and z > t.launch_z):
            phase, timer = "fly", 0
            side = +1 if lat >= 0 else -1

    elif phase == "crouch":
        if timer >= t.crouch_steps:
            phase, timer = "takeoff", 0

    elif phase == "takeoff":
        # No minimum: a stiff rod reaches the target in a step or two.
        if timer > t.takeoff_max_steps or vz >= t.takeoff_done * t.launch_up:
            phase, timer = "fly", 0

    elif phase == "push":
        if (timer > t.burst_min_steps and side * v_lat < -t.push_leave_speed) \
                or timer > t.push_max_steps:
            phase, timer = "fly", 0

    elif phase == "fly":
        if side > 0 and lat < -reach_lat and v_lat < 0:
            side, phase, timer = -1, "push", 0
        elif side < 0 and lat > reach_lat and v_lat > 0:
            side, phase, timer = +1, "push", 0
        elif z < t.floor_z and abs(v_lat) < t.floor_v_lat and timer > t.floor_fly_steps:
            # Back on the floor. Relaunch, after a recentre if it drifted.
            phase = "recentre" if abs(along) > t.drift_limit else t.launch_phase
            timer = 0

    elif phase in ("recentre", "settle"):
        if abs(along) < t.centre_tol and abs(v_along) < t.centre_v_tol \
                and timer > t.settle_steps:
            phase, timer = t.launch_phase, 0

    elif phase == "exit":
        if timer > t.exit_max_steps or (
                timer > t.burst_min_steps and side * v_lat < -t.exit_leave_speed):
            phase, timer = "fly_out", 0

    elif phase == "fly_out":
        if abs(lat) < shaft.half_width and vz < 0 and z < top:
            phase, timer = "fly", 0            # fell back in: resume the zig-zag

    elif phase in ("land", "brake"):
        if timer > t.land_steps:
            settled = abs(v_lat) < t.settled_v and abs(vz) < t.settled_v
            if settled and on_wall_top(p, shaft, tuning=t):
                phase, timer = "stand", 0
            elif z > top and shaft.over_top(lat):
                pass                            # still on the top, still moving: keep braking
            elif abs(lat) < shaft.half_width and z < top:
                phase, timer = "fly", 0
            else:
                phase, timer = "done", 0

    elif phase == "hold":
        if timer >= t.hold_steps:
            phase, timer, clamp_ext = "descend", 0, float(max_extend)

    elif phase == "descend":
        if z < t.landed_z and abs(vz) < t.landed_vz:
            phase, timer = "stand", 0

    elif phase == "stand":
        if timer > t.stand_steps:
            phase, timer = "done", 0

    return ZigzagState(phase=run[0], side=run[1], timer=run[2],
                       clamp_ext=clamp_ext, hold_z=hold_z,
                       successor=(phase, side, timer))


def zigzag_climb(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    *,
    phase: str = "stand",
    side: int = +1,
    wall_axis: np.ndarray | None = None,
    lin_vel: np.ndarray | None = None,
    along_off: float = 0.0,
    core_z: float = 0.0,
    clamp_ext: float | None = None,
    tuning: ZigzagTuning = TUNING,
) -> np.ndarray:
    """Zig-zag wall-jump up a chimney: one step of the phase in `phase`.

    Every phase delegates to one primitive: `chimney_climb` for the wall
    work, `jump_to` for an aimed floor jump, `move` to recentre, `stop` to
    brake on the top. The caller passes the state from :func:`next_phase`,
    which it calls before every step.

    Phases
    ------
    launch, push, fly, fly_out, hold, descend, stand :
        `chimney_climb` phases. `side` and `along_off` shape the push;
        `clamp_ext` sets the descend clamp; `core_z` opens the landing
        gear below `tuning.gear_z`.
    crouch, takeoff :
        `jump_to` toward the wall opposite `side`, at
        `tuning.launch_lateral` sideways and `tuning.launch_up` upward,
        servoed on `lin_vel` (3-vector).
    exit :
        `chimney_climb` push with the wider `tuning.exit_band`.
    land :
        `chimney_climb` hold with the clamp open and the gear out.
    brake :
        `stop` against `lin_vel`, within `tuning.brake_stop_distance`.
    recentre :
        `move` back toward the gap's middle, against `along_off`.
    settle, done :
        `stop`.
    `wall_axis` points across the gap, wall to wall; default +y.
    """
    axis = np.asarray([0.0, 1.0] if wall_axis is None else wall_axis, dtype=np.float64)
    axis = axis[:2] / max(float(np.linalg.norm(axis[:2])), 1e-9)
    vel = None if lin_vel is None else np.asarray(lin_vel, dtype=np.float64)
    vel2 = None if vel is None else vel[:2]
    t = tuning

    if phase in ("launch", "fly", "fly_out", "hold", "stand"):
        wall_phase = "fly" if phase == "fly_out" else phase
        return chimney_climb(quat, dirs_body, max_extend, axis, phase=wall_phase)

    if phase == "push":
        return chimney_climb(quat, dirs_body, max_extend, axis, phase="push",
                             side=side, push_frac=t.push_frac, x_off=along_off)

    if phase == "exit":
        lat, z_lo, z_hi = t.exit_band
        return chimney_climb(quat, dirs_body, max_extend, axis, phase="push",
                             side=side, push_frac=t.exit_frac, x_off=along_off,
                             push_lat=lat, push_z_lo=z_lo, push_z_hi=z_hi)

    if phase in ("crouch", "takeoff"):
        vel3 = None if vel is None else np.array([vel[0], vel[1], vel[2] if len(vel) > 2 else 0.0])
        return jump_to(quat, dirs_body, max_extend * t.launch_power, -side * axis,
                       phase=phase, vel=vel3, vx_target=t.launch_lateral,
                       vz_target=t.launch_up)

    if phase == "descend":
        return chimney_climb(quat, dirs_body, max_extend, axis, phase="descend",
                             clamp_ext=clamp_ext, near_floor=core_z < t.gear_z)

    if phase == "land":
        return chimney_climb(quat, dirs_body, max_extend, axis, phase="hold",
                             clamp_ext=0.0, near_floor=True, gear=t.land_gear)

    if phase == "brake":
        return stop(quat, dirs_body, max_extend, lin_vel=vel2,
                    stop_distance=t.brake_stop_distance)

    if phase == "recentre":
        along_dir = np.array([axis[1], -axis[0]])
        head = -np.sign(along_off) * along_dir if abs(along_off) > 1e-9 else along_dir
        return move(quat, dirs_body, max_extend, head, speed=t.recentre_speed)

    if phase in ("settle", "done"):
        return stop(quat, dirs_body, max_extend, lin_vel=vel2)

    raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")


def wall_jump_climb(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    *,
    phase: str = "stand",
    side: int = +1,
    wall_axis: np.ndarray | None = None,
    lin_vel: np.ndarray | None = None,
    along_off: float = 0.0,
    core_z: float = 0.0,
    clamp_ext: float | None = None,
    tuning: ZigzagTuning = WIDE_TUNING,
) -> np.ndarray:
    """Wall-jump zig-zag up a wide gap: one step of the phase in `phase`.

    The same machine as :func:`zigzag_climb`, with :data:`WIDE_TUNING`: the
    floor jump is aimed at the nearer wall (`crouch`, `takeoff`), each push
    starts as the ball arrives at the wall, and the pushes from within a
    metre of the lip are eased so the ball just clears it. The ball flies
    freely between pushes, so this needs the stiffer rod of
    `configs/rl/wall_jump.yaml`; with the default robot every crossing loses
    height.

    Phases
    ------
    Those of :func:`zigzag_climb`, with the same options. `crouch` and
    `takeoff` replace `launch`; `exit` is the eased push; `fly_out`, `hold`
    and `descend` are not used on this course.
    """
    return zigzag_climb(quat, dirs_body, max_extend, phase=phase, side=side,
                        wall_axis=wall_axis, lin_vel=lin_vel, along_off=along_off,
                        core_z=core_z, clamp_ext=clamp_ext, tuning=tuning)
