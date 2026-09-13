"""The motors and the bus between them and the controller.

Everything here used to live inside `MujocoSteeringEnv`, which meant only a
trained steering policy ever met the hardware. A skill called against the base
env got MJCF friction and damping and nothing else: no rod velocity ceiling, no
stall force, no battery budget, no transport delay. Its numbers therefore said
what an ideal robot would do.

Pulling the models out here lets any caller opt in, and lets the base env apply
them once for everyone. It is off by default so no existing demo silently
changes; see ``sim2real.actuator_in_env`` in `configs/rl/config.yaml`.

Two units traps live in this file, both of which were real bugs:

* ``dt`` is the interval between *calls*, not the MuJoCo timestep. The base env
  advances ``action_repeat`` steps per call, so passing the raw timestep made
  every rod ``action_repeat`` times too slow.
* latency is quantised by that same interval. Commands reach the motors once
  per call, so a delay shorter than one call cannot be represented. The exact
  figure achieved is reported rather than the figure requested.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class RealisticActuatorModel:
    """A rod that cannot move instantly, infinitely hard, or for free.

    Four limits, applied in order: acceleration, velocity, stall force through
    a load derating, and finally a shared battery ceiling across all 60 motors.
    """

    def __init__(self, n_bars: int = 60, v_max: float = 0.28, a_max: float = 3.5,
                 f_max: float = 50.0, p_battery_max: float = 200.0,
                 dt: float = 0.01, min_ext: float = 0.025, max_ext: float = 0.160):
        self.n_bars, self.v_max, self.a_max = n_bars, v_max, a_max
        self.f_max, self.p_battery_max, self.dt = f_max, p_battery_max, dt
        self.min_ext, self.max_ext = min_ext, max_ext
        self.reset()

    def reset(self) -> None:
        self.current_pos = np.full(self.n_bars, self.min_ext, dtype=np.float32)
        self.current_vel = np.zeros(self.n_bars, dtype=np.float32)

    def apply_dynamics(self, target_pos: np.ndarray,
                       actual_forces: np.ndarray | None = None):
        """What the rods actually reach, given what was asked for."""
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(-1)
        desired_vel = (target_pos - self.current_pos) / self.dt

        effective_v_max = self.v_max
        if actual_forces is not None:
            # A loaded motor turns slower. Full stall force halves top speed.
            load = np.clip(np.abs(actual_forces) / max(self.f_max, 1e-9), 0.0, 1.0)
            effective_v_max = self.v_max * (1.0 - 0.5 * load)

        accel = np.clip((desired_vel - self.current_vel) / self.dt,
                        -self.a_max, self.a_max)
        vel = np.clip(self.current_vel + accel * self.dt,
                      -effective_v_max, effective_v_max)

        total_power = 0.0
        if actual_forces is not None:
            total_power = float(np.sum(np.abs(actual_forces * vel)))
            if total_power > self.p_battery_max:
                # One battery feeds every motor, so they all slow together.
                vel = vel * (self.p_battery_max / total_power)

        new_pos = np.clip(self.current_pos + vel * self.dt, self.min_ext, self.max_ext)
        self.current_vel = (new_pos - self.current_pos) / self.dt
        self.current_pos = new_pos.copy()
        return new_pos, {"total_power_w": total_power,
                         "max_actuator_vel": float(np.max(np.abs(self.current_vel)))}


class CommandLatency:
    """Hold each command back by the transport delay on the bus.

    The obvious implementation is wrong: appending to a `deque` and immediately
    popping its left returns the item just added, whatever ``maxlen`` says. That
    is what the old code did, so the modelled latency was zero. This reads the
    oldest entry *before* adding the newest.
    """

    def __init__(self, delay_steps: int):
        self.delay_steps = max(0, int(delay_steps))
        self._q: deque | None = deque(maxlen=self.delay_steps) if self.delay_steps else None

    def reset(self) -> None:
        if self._q is not None:
            self._q.clear()

    def __call__(self, command: np.ndarray) -> np.ndarray:
        if self._q is None:
            return command
        if len(self._q) < self._q.maxlen:
            # Priming. Nothing has had time to arrive late yet, so the robot
            # acts on the current command rather than on a blank.
            while len(self._q) < self._q.maxlen:
                self._q.append(np.asarray(command).copy())
            return command
        sent = self._q[0].copy()
        self._q.append(np.asarray(command).copy())
        return sent


class HardwareModel:
    """Latency then actuator dynamics, in the order the signal meets them."""

    def __init__(self, actuator: RealisticActuatorModel | None,
                 latency: CommandLatency | None, dt: float):
        self.actuator, self.latency, self.dt = actuator, latency, dt
        self.last_metrics: dict = {}

    @property
    def latency_ms(self) -> float:
        return 0.0 if self.latency is None else self.latency.delay_steps * self.dt * 1000.0

    @classmethod
    def from_config(cls, cfg, n_bars: int, dt: float, max_extend: float,
                    min_ext: float = 0.025) -> "HardwareModel | None":
        """Build from a `sim2real` block, or return None when it is off.

        ``dt`` must be the interval between calls to :meth:`apply`, not the
        MuJoCo timestep.
        """
        s2r = getattr(cfg, "sim2real", None)
        if s2r is None or not bool(getattr(s2r, "enabled", False)):
            return None
        d = dict(s2r)

        actuator = None
        if bool(d.get("enable_actuator_limits", True)):
            actuator = RealisticActuatorModel(
                n_bars=n_bars,
                v_max=float(d.get("actuator_v_max", 0.28)),
                a_max=float(d.get("actuator_a_max", 3.5)),
                f_max=float(d.get("actuator_force_limit", 50.0)),
                p_battery_max=float(d.get("battery_p_max", 200.0)),
                dt=dt, min_ext=min_ext, max_ext=float(max_extend))

        latency = None
        if bool(d.get("enable_latency", True)):
            ms = float(d.get("action_delay_ms", 25.0))
            latency = CommandLatency(int(np.clip(round(ms / max(dt * 1000.0, 1e-9)), 0, 64)))

        if actuator is None and latency is None:
            return None
        return cls(actuator, latency, dt)

    def reset(self) -> None:
        if self.actuator is not None:
            self.actuator.reset()
        if self.latency is not None:
            self.latency.reset()

    def apply(self, targets: np.ndarray,
              contact_forces: np.ndarray | None = None) -> np.ndarray:
        """Turn a commanded set of rod targets into what the motors reach."""
        out = np.asarray(targets, dtype=np.float32).reshape(-1)
        if self.latency is not None:
            out = self.latency(out)
        if self.actuator is not None:
            out, self.last_metrics = self.actuator.apply_dynamics(
                out, actual_forces=contact_forces)
        return out


__all__ = ["RealisticActuatorModel", "CommandLatency", "HardwareModel"]
