"""RL options backed by the existing ``skills/`` controllers.

Keep this action order stable: it is part of the checkpoint interface.
Jumps execute their phases within one policy decision; other skills run for
``decision_every`` low-level steps. No phase state survives between options.
"""
from __future__ import annotations

import math
import numpy as np

from ._gym import spaces
from skills.runner import skill_targets


SKILL_NAMES = (
    "move", "stop", "reverse", "follow_path", "straddle_gap",
    "traverse_rough_terrain", "jump_up", "jump_forward_while_stopped",
    "jump_forward_while_moving", "jump_to",
    "crawl_pipe",      # appended: earlier indices stay valid for old checkpoints
    "flip",            # turn the travel direction around; "move" then goes the other way
)
JUMPS = frozenset(SKILL_NAMES[6:10])
MAX_PARAMS = 3
#: Above this ground speed a running jump skips its 0.55 s sprint run-up.
RUNNING_SPEED = 0.5
#: Macro-mode running jump. Measured on the playground with the oracle in
#: scripts/data/generate_skills_vla_dataset.py: power 0.9 with a 0.04 landing
#: rollout clears the hurdle, both boxes, the stairs and the wall without a
#: touch. Full power (1.0) or the skill's 0.10 rollout carries the ball off the
#: far end of the 1.2 m boxes; 0.85 and below no longer reaches the box top.
MACRO_JUMP_POWER = 0.9
MACRO_ROLLOUT_GAIN = 0.04


def action_space(mode):
    size = len(SKILL_NAMES) + (MAX_PARAMS if mode == "hybrid" else 0)
    return spaces.Box(-1.0, 1.0, shape=(size,), dtype=np.float32)


def decode(action, mode, waypoint_heading):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    expected = len(SKILL_NAMES) + (MAX_PARAMS if mode == "hybrid" else 0)
    if action.size != expected or not np.isfinite(action).all():
        raise ValueError(f"skills backend expects {expected} finite action values")
    index = int(np.argmax(action[:len(SKILL_NAMES)]))
    name = SKILL_NAMES[index]
    angle, speed, power = waypoint_heading, 1.1, MACRO_JUMP_POWER
    vx, vz = .6, 2.6
    if mode == "hybrid":
        heading_param, speed_param, power_param = np.clip(action[-MAX_PARAMS:], -1, 1)
        angle += float(heading_param) * np.pi / 2
        speed = .2 + .7 * (float(speed_param) + 1)
        power = .35 + .325 * (float(power_param) + 1)
        vx, vz = speed, 1.4 + .8 * (float(power_param) + 1)
    params = {}
    if name in {"move", "reverse", "follow_path", "traverse_rough_terrain"}:
        params["speed"] = speed
    if name == "crawl_pipe":
        params["speed"] = min(speed, 0.8)
    if name in JUMPS - {"jump_to"}:
        params["power"] = power
    if name == "jump_forward_while_moving":
        params["rollout_gain"] = MACRO_ROLLOUT_GAIN
    if name == "jump_to":
        params.update(vx_target=vx, vz_target=vz, wall_lock=True)
    heading = np.array([np.cos(angle), np.sin(angle)])
    return index, name, params, heading


#: The running jump is the one option that times itself from the terrain:
#: given a plan (an edge ahead and a firing distance) it rolls up to the
#: edge and fires at that distance, so the policy's decision can come a
#: metre early and still be right.
TIMED_JUMPS = frozenset({"jump_forward_while_moving"})
#: Approach speed (the macro `move` speed) and its budget: at worst the ball
#: starts from rest, so allow the distance at half speed, 1 to 4 s.
APPROACH_SPEED = 1.1
APPROACH_MIN_S, APPROACH_MAX_S = 1.0, 4.0
#: With less run-up than this before the trigger point, and the ball slow,
#: the running jump cannot get up to speed: the option hops instead (an
#: aimed standing hop, `jump_to`, planned by skills.mid_level.hop_planner
#: from the edge's own geometry). Short decks are jumped this way.
RUNUP_MIN = 0.8
HOP_SPEED_MAX = 0.5
#: Creep to the hop's stand point in short pushes (the gait cannot go slow),
#: this near is near enough.
HOP_STAND_TOL = 0.08
#: Re-measure the edge with the probe this often during the approach (every
#: control step: the heading follows the route, so the edge distance is
#: measured along the current heading); odometry fills in when the probe
#: loses the edge for a moment.
REPROBE_EVERY = 1


def hop_calibration_path(env):
    """The aimed-hop calibration for this build, or None when the build cannot hop.

    Only the long-stroke build (0.26 m rods) has a usable table. On the
    standard 0.16 m build the aimed hop rises 0.20 to 0.33 m on a bad
    orientation and bounces off a 0.20 m deck face 0.55 m away (0 of 8);
    the full-power standing jump lands on a 1 m deck 3 times in 8 at best.
    Its table (hop_calibration_standard.json) is kept for the record; with
    it in use the maze deck row got worse (15 hits per episode, from 4).
    """
    from pathlib import Path
    stroke = float(getattr(env, "max_extend", 0.16))
    if stroke <= 0.2:
        return None
    p = Path(__file__).resolve().parents[1] / "skills" / "mid_level" / "hop_calibration.json"
    return str(p) if p.exists() else None


class SkillOption:
    """State routing and bounded jump sequencing, interruptible each env step.

    Crouch/burn timings match skills.runner at its nominal 10 ms control
    interval. Flight/landing use local terrain height, so elevated platforms
    do not look like perpetual flight. Landing is latched to avoid chatter.

    With ``plan`` (from ``terrain_probe.plan_jump``) a running jump first
    approaches: it drives the macro ``move`` along the heading until the
    edge is ``plan["trigger"]`` metres away, then runs the jump schedule.
    ``result`` says how it ended: "success" (landed and settled past the
    edge), "short" (landed and settled before the edge: it hit it or fell
    short), "timed_out" (the jump budget ran out mid-air or mid-bounce),
    "approach_timeout" (never got to the edge), "too_close" (the edge got
    under the rods before the ball was settled enough to fire; it was
    rolled), "interrupted" (the episode ended during the option), or None
    while running.
    """

    def __init__(self, env, name, params, heading, decision_every, ground_height,
                 plan=None, probe=None, heading_fn=None):
        self.env, self.name = env, name
        self.params, self.heading = dict(params), heading
        self.start_xy = env.data.qpos[:2].copy()
        self.ground_height = ground_height
        self.dt = float(env.model.opt.timestep * env.action_repeat)
        self.is_jump = name in JUMPS
        jump_budget = math.ceil((2.4 if name == "jump_forward_while_moving" else 1.6) / self.dt)
        self.max_steps = jump_budget if self.is_jump else decision_every
        self.phase = "drive"
        self.landing_started = None
        self.burn_finished = False
        self.skip_sprint = float(np.linalg.norm(env.data.qvel[:2])) > RUNNING_SPEED
        self.result = None
        # Self-timed approach (running jump with a plan).
        self.plan = plan if (plan and name in TIMED_JUMPS) else None
        self.probe = probe
        self.heading_fn = heading_fn                # the route's heading now; the approach follows it
        self.jump_start = 0                         # control step at which the jump schedule began
        self.approach_budget = 0
        self.remaining = None                       # distance to the edge, metres
        self.line_origin = self.start_xy.copy()     # the approach holds the line through here
        self.edge_xy = None                         # where the edge is, in the world, for the result check
        self.hop = None                             # a HopPlan when this jump is a standing hop
        self.hop_phase_start = 0
        if self.plan is not None:
            self.edge_xy = self.start_xy + self.heading * float(self.plan["edge"].dist)
            self.remaining = float(self.plan["edge"].dist)
            seconds = float(np.clip((self.remaining + 0.5) / (APPROACH_SPEED / 2), APPROACH_MIN_S, APPROACH_MAX_S))
            self.approach_budget = math.ceil(seconds / self.dt)
            self.max_steps = self.approach_budget + jump_budget
            self.phase = "approach"
            runup = self.remaining - float(self.plan["trigger"])
            if runup < RUNUP_MIN and float(np.linalg.norm(env.data.qvel[:2])) < HOP_SPEED_MAX:
                self.hop = self._plan_hop()
                if self.hop is not None:
                    self.phase = "creep"
                    self.max_steps = self.approach_budget + math.ceil(1.0 / self.dt) + math.ceil(1.6 / self.dt)

    # ---- the standing hop -------------------------------------------
    def _plan_hop(self):
        """An aimed hop onto the edge's surface, or None when the planner cannot promise one."""
        from radial_sphere.terrain_probe import hop_target
        from skills.mid_level.hop_planner import STAND_EDGE, load_hop_calibration, plan_standing_hop
        target = hop_target(self.plan)
        if target is None:
            return None
        table = hop_calibration_path(self.env)
        if table is None:
            return None
        cal = load_hop_calibration(table)
        return plan_standing_hop(0.0, (-1.0, target["near"] - STAND_EDGE), target, calibration=cal)

    def _hop_step(self, step):
        """Creep to the stand point, settle, then run the aimed hop."""
        progress = float((self.env.data.qpos[:2] - self.start_xy) @ self.heading)
        if self.phase == "creep":
            err = float(self.hop.x0) - progress
            if abs(err) <= HOP_STAND_TOL or err < -0.3 or step >= self.approach_budget:
                self.phase, self.hop_phase_start = "settle", step
            else:
                # Short pushes with stops between: the gait cannot go slow.
                cycle = step % 26
                if cycle < 6:
                    return skill_targets(self.env, "move" if err > 0 else "reverse", step,
                                         d_hat=self.heading, speed=APPROACH_SPEED)
                return skill_targets(self.env, "stop", step)
        if self.phase == "settle":
            if self._settled() or step - self.hop_phase_start > math.ceil(1.0 / self.dt):
                # Hand over to the jump_to schedule with the planned command.
                self.name = "jump_to"
                self.params = {"vx_target": float(self.hop.vx_cmd), "vz_target": float(self.hop.vz_cmd),
                               "wall_lock": True}
                self.jump_start = step
                self.skip_sprint = False
                self.phase = "crouch"
                return None
            return skill_targets(self.env, "stop", step)
        return None

    # ---- the approach ------------------------------------------------
    def approaching(self, step) -> bool:
        return self.plan is not None and self.phase in ("approach", "creep", "settle")

    def _distance_to_edge(self, step) -> float:
        """Distance to the edge along the current heading.

        The approach follows the route like `move` does (the heading is read
        from the waypoints every step: a frozen heading drifted the ball into
        a post on a diagonal route), so the odometry count is re-anchored on
        the probe every step; odometry alone fills in when the probe loses
        the edge for a moment.
        """
        if self.heading_fn is not None:
            h = np.asarray(self.heading_fn(), dtype=np.float64)[:2]
            if np.linalg.norm(h) > 1e-6:
                self.heading = h / np.linalg.norm(h)
        progress = float((self.env.data.qpos[:2] - self.start_xy) @ self.heading)
        estimate = float(self.plan["edge"].dist) - progress
        if self.probe is not None and step % REPROBE_EVERY == 0 and step > 0:
            fresh = self.probe(self.heading)
            if fresh is not None and abs(float(fresh["edge"].dist) - estimate) < 0.4:
                # Same edge, measured again: trust the measurement, and its
                # shape (a tread seen from far away reads as a platform).
                self.plan.update(fresh)
                self.start_xy = self.env.data.qpos[:2].copy()
                self.edge_xy = self.start_xy + self.heading * float(fresh["edge"].dist)
                estimate = float(fresh["edge"].dist)
        return estimate

    def _settled(self) -> bool:
        pos = self.env.data.qpos[:3]
        return (pos[2] <= self.ground_height(pos[:2]) + self.env.sphere_radius + .05
                and abs(float(self.env.data.qvel[2])) < .3)

    def _approach_step(self, step):
        """Roll on until the edge is at the firing distance, then hand over to the jump.

        An edge already nearer than the trigger is fired at once only when
        the ball is settled (a jump armed while still bouncing from the last
        landing hit the next crate every time); until then it rolls on, and
        if the edge gets under the rods first the option ends as "too_close".
        """
        self.remaining = self._distance_to_edge(step)
        from radial_sphere.terrain_probe import MIN_TARGET
        if self.remaining < MIN_TARGET:
            self.result = "too_close"
            return None
        if step >= self.approach_budget:
            self.result = "approach_timeout"
            return None
        if self.remaining <= float(self.plan["trigger"]) and (self._settled()
                                                              or self.remaining < float(self.plan["trigger"]) - 0.15):
            self.phase = "sprint"
            self.jump_start = step
            self.skip_sprint = float(np.linalg.norm(self.env.data.qvel[:2])) > RUNNING_SPEED
            return None
        normal = np.array([-self.heading[1], self.heading[0]])
        cross = float((self.env.data.qpos[:2] - self.line_origin) @ normal)
        return skill_targets(self.env, "move", step, d_hat=self.heading, speed=APPROACH_SPEED,
                             cross_track_error=cross)

    def _jump_phase(self, step):
        elapsed = (step - self.jump_start) * self.dt
        running = self.name == "jump_forward_while_moving"
        if running and self.skip_sprint:
            # Already rolling: the sprint phase would kick the ball off the
            # ground with the rods still out. Go straight to dip and launch.
            crouch_start, burn_start, burn_end = 0., .07, .20
        else:
            crouch_start, burn_start, burn_end = (.55, .62, .75) if running else (0., .20, .32)
        if elapsed < crouch_start:
            return "sprint"
        if elapsed < burn_start:
            return "dip" if running else "crouch"
        if (self.name == "jump_to" and elapsed > burn_start
                and self.env.data.qvel[2] >= self.params["vz_target"]):
            self.burn_finished = True
        if elapsed < burn_end and not self.burn_finished:
            return "launch" if running else "takeoff"
        pos = self.env.data.qpos[:3]
        near_ground = pos[2] <= self.ground_height(pos[:2]) + self.env.sphere_radius + .10
        if self.landing_started is None and self.env.data.qvel[2] <= 0 and near_ground:
            self.landing_started = elapsed
        return "landing" if self.landing_started is not None else "airborne"

    def targets(self, step):
        if self.hop is not None and self.phase in ("creep", "settle"):
            t = self._hop_step(step)
            if t is not None:
                return t
        if self.approaching(step):
            t = self._approach_step(step)
            if t is not None:
                return t
            if self.result in ("approach_timeout", "too_close"):
                # Never reached the edge, or it is already under the rods:
                # end the option on a brake instead of jumping into nothing.
                return skill_targets(self.env, "stop", step)
        call = dict(self.params)
        if self.is_jump:
            self.phase = self._jump_phase(step)
            call["phase"] = self.phase
        if self.name == "move":
            normal = np.array([-self.heading[1], self.heading[0]])
            call["cross_track_error"] = float((self.env.data.qpos[:2] - self.start_xy) @ normal)
        if self.name == "follow_path":
            mechanism = getattr(self.env.cfg.robot, "rod_mechanism", "single_stage")
            call.update(rod_mechanism=mechanism, curve_rod_mechanism=mechanism)
        return skill_targets(self.env, self.name, step, d_hat=self.heading, **call)

    def complete(self, steps_executed):
        """Done 0.2 s after touchdown, once the ball has stopped bouncing.

        The landing rollout can throw the ball back up; handing control to the
        next option mid-bounce makes that option fire in the air.
        """
        if self.result in ("approach_timeout", "too_close"):
            return True
        if self.approaching(steps_executed):
            return False
        if self.landing_started is None or (steps_executed - self.jump_start) * self.dt < self.landing_started + .20:
            return False
        pos = self.env.data.qpos[:3]
        near_ground = pos[2] <= self.ground_height(pos[:2]) + self.env.sphere_radius + .10
        done = near_ground and abs(float(self.env.data.qvel[2])) < .5
        if done and self.result is None:
            self.result = self._landing_result()
        return done

    def _landing_result(self) -> str:
        """"success" when the ball came down past the edge it aimed at, else "short"."""
        if self.edge_xy is None:
            return "success"
        past = float((self.env.data.qpos[:2] - self.edge_xy) @ self.heading)
        return "success" if past > 0.0 else "short"

    def finish(self, steps_executed):
        """Called once by the executor when the option stops; fixes ``result``."""
        if self.result is None and self.is_jump:
            if self.complete(steps_executed):
                self.result = self.result or self._landing_result()
            elif steps_executed >= self.max_steps:
                self.result = "timed_out"
            else:
                self.result = "interrupted"          # the episode ended mid-option
        return self.result
