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
)
JUMPS = frozenset(SKILL_NAMES[6:])
MAX_PARAMS = 3


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
    angle, speed, power = waypoint_heading, 1.1, 1.0
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
    if name in JUMPS - {"jump_to"}:
        params["power"] = power
    if name == "jump_to":
        params.update(vx_target=vx, vz_target=vz, wall_lock=True)
    heading = np.array([np.cos(angle), np.sin(angle)])
    return index, name, params, heading


class SkillOption:
    """State routing and bounded jump sequencing, interruptible each env step.

    Crouch/burn timings match skills.runner at its nominal 10 ms control
    interval. Flight/landing use local terrain height, so elevated platforms
    do not look like perpetual flight. Landing is latched to avoid chatter.
    """

    def __init__(self, env, name, params, heading, decision_every, ground_height):
        self.env, self.name = env, name
        self.params, self.heading = dict(params), heading
        self.start_xy = env.data.qpos[:2].copy()
        self.ground_height = ground_height
        self.dt = float(env.model.opt.timestep * env.action_repeat)
        self.is_jump = name in JUMPS
        self.max_steps = (math.ceil((2.4 if name == "jump_forward_while_moving" else 1.6) / self.dt)
                          if self.is_jump else decision_every)
        self.phase = "drive"
        self.landing_started = None
        self.burn_finished = False

    def _jump_phase(self, step):
        elapsed = step * self.dt
        running = self.name == "jump_forward_while_moving"
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
        return (self.landing_started is not None
                and steps_executed * self.dt >= self.landing_started + .20)
