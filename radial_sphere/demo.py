"""Run a skill demo that is described by a yaml file rather than a script.

Twenty runners lived under ``scripts/skills/``. Sixteen of them called
``make_run_dir``, sixteen rendered frames, seventeen drew the same overlay,
and fourteen stepped a skill through ``execute_skill``. They differed in
data, not in logic: which scenario, which skill, how many steps, which
cameras, and what counts as success.

That data is now a yaml under ``demos/``, and this module is the one
runner. A demo that genuinely needs its own control flow, such as a course
state machine or a calibration sweep, stays a script.

The spec
--------
``scenario``  kind, config path and seed for `generate_scenario`.
``spawn``     optional starting pose, written straight into qpos.
``settle``    steps to run before recording, to let the robot sit down.
``skill``     the registry name, fixed ``args``, and ``feedback`` bindings
              that read live state each step.
``steps``     how long to run. ``stop_when`` finishes early once every bound
              in it holds; ``stop_when_any`` takes a list and finishes as soon
              as any one entry holds.
``video``     one or more panes, each a camera and an overlay.
``expect``    bounds on the metrics, checked at the end.

Live state
----------
Both ``feedback`` bindings and overlay text see the same names:

    step time ball_x ball_y ball_z vx vy vz speed distance_x path_length

Overlay lines are format strings over that namespace, so
``"x = {ball_x:.2f} m"`` works without any code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Bounds a demo may place on a metric.
_COMPARATORS = {
    "min": lambda value, bound: value >= bound,
    "max": lambda value, bound: value <= bound,
}


@dataclass
class DemoResult:
    """What one demo run produced."""

    name: str
    metrics: dict[str, float]
    failures: list[str] = field(default_factory=list)
    run_dir: Path | None = None
    video: Path | None = None
    steps_run: int = 0

    @property
    def passed(self) -> bool:
        return not self.failures


def _state(env, step: int, dt: float, distance_x: float, path_length: float,
           goal=None) -> dict[str, float]:
    pos = env.data.qpos[0:3]
    vel = env.data.qvel[0:3]
    goal_distance = (float(np.linalg.norm(np.asarray(goal, dtype=float)[:2] - pos[:2]))
                     if goal is not None else float("inf"))
    return {
        "goal_distance": goal_distance,
        "goal_x": float(np.asarray(goal, dtype=float)[0]) if goal is not None else 0.0,
        "step": step,
        "time": step * dt,
        "ball_x": float(pos[0]), "ball_y": float(pos[1]), "ball_z": float(pos[2]),
        "vx": float(vel[0]), "vy": float(vel[1]), "vz": float(vel[2]),
        "speed": float(np.linalg.norm(vel[:2])),
        "ball_xy": [float(pos[0]), float(pos[1])],
        "lin_vel": [float(vel[0]), float(vel[1]), float(vel[2])],
        "distance_x": distance_x,
        "path_length": path_length,
    }


def _scenario_args(spec, scenario) -> dict[str, Any]:
    """Arguments taken from the scenario, such as a cone layout.

    A cone course cannot state its cones in the demo yaml: they come from the
    scenario generator. This maps a skill argument to a scenario attribute.
    """
    out = {}
    for key, attribute in (spec.get("args_from_scenario", {}) or {}).items():
        value = getattr(scenario, str(attribute), None)
        if value is None:
            raise SystemExit(f"scenario has no {attribute!r} for skill argument {key!r}")
        out[key] = np.asarray(value, dtype=float)
    return out


def _skill_kwargs(spec, state: dict[str, float], *, feedback: bool = True) -> dict[str, Any]:
    """Fixed arguments, plus any bound to live state this step.

    The settle phase passes ``feedback=False``. Closing the loop while the
    robot is still dropping onto its rods feeds it a transient it should not
    react to, and the run diverges from there.
    """
    from omegaconf import OmegaConf

    args = OmegaConf.to_container(spec.get("args", {}) or {}, resolve=True)
    if not feedback:
        return args
    for key, source in (spec.get("feedback", {}) or {}).items():
        if source not in state:
            raise SystemExit(
                f"feedback {key}={source!r} is not a live state name; "
                f"available: {', '.join(sorted(state))}")
        args[key] = state[source]
    return args


def render_pane(env, pane, state: dict[str, float] | None = None) -> np.ndarray:
    """One annotated view. `camera` names an env camera; `tracking` builds one."""
    import mujoco

    from .overlay import annotate

    if "tracking" in pane and pane.tracking is not None:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = env.core_body_id
        cam.distance = float(pane.tracking.get("distance", 2.0))
        cam.elevation = float(pane.tracking.get("elevation", -15.0))
        cam.azimuth = float(pane.tracking.get("azimuth", 90.0))
        if env.renderer is None:
            env.render(camera_name=str(pane.get("camera", "chase")))
        env.renderer.update_scene(env.data, camera=cam)
        frame = env.renderer.render()
    else:
        frame = env.render(camera_name=str(pane.get("camera", "chase")))

    title = str(pane.get("title", "")).format(**state)
    lines = [str(line).format(**state) for line in (pane.get("lines") or [])]
    return annotate(frame, title, lines, margin=14) if title else frame


def compose(frames: list[np.ndarray]) -> np.ndarray:
    """Lay panes side by side, matching heights."""
    if len(frames) == 1:
        return frames[0]
    height = min(f.shape[0] for f in frames)
    out = []
    for f in frames:
        if f.shape[0] != height:
            import cv2
            width = int(f.shape[1] * height / f.shape[0])
            f = cv2.resize(f, (width, height))
        out.append(f)
    return np.concatenate(out, axis=1)


def check_expectations(metrics: dict[str, float], expect) -> list[str]:
    failures = []
    for key, bounds in (expect or {}).items():
        if key not in metrics:
            failures.append(f"{key}: no such metric; have {', '.join(sorted(metrics))}")
            continue
        for word, bound in bounds.items():
            if word not in _COMPARATORS:
                failures.append(f"{key}: unknown bound {word!r}; use min or max")
            elif not _COMPARATORS[word](metrics[key], float(bound)):
                failures.append(f"{key} = {metrics[key]:.3f}, expected {word} {float(bound):.3f}")
    return failures


# ---------------------------------------------------------------------------
# Toolkit for scripts that keep their own control flow
#
# A phase machine or a course cannot become a yaml: its control flow is the
# point. But sixteen runners each built a run dir, opened a video writer,
# rendered panes, drew the same overlay and accumulated the same metrics.
# That part is shared, and these two classes are it. `run_demo` above is
# written on them, so the declarative path and the scripted path cannot drift.
# ---------------------------------------------------------------------------


class Recorder:
    """Run directory, video writer and frame capture for one run.

    ``every`` is in caller steps, so a script keeps its own loop and simply
    calls :meth:`capture` each iteration.

        rec = Recorder(env, "wall_run", tag=mode, enabled=args.video)
        ...
        rec.capture(step, panes, state)
        rec.close()
    """

    def __init__(self, env, name: str, tag: str | None = None, *,
                 enabled: bool = True, fps: int = 25, every: int = 4,
                 out_name: str | None = None):
        from .run_id import build_run_id
        from .snapshot import make_run_dir

        self.env = env
        self.name = name
        self.every = max(int(every), 1)
        self.enabled = bool(enabled)
        self.run_dir = self.video = self._writer = None
        self.n_frames = 0
        if not self.enabled:
            return
        import imageio

        self.run_dir = make_run_dir(build_run_id(name, tag=tag))
        self.video = Path(self.run_dir) / "renders" / f"{out_name or name}.mp4"
        self.video.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(self.video), fps=int(fps),
                                          codec="libx264", pixelformat="yuv420p")

    def due(self, step: int) -> bool:
        return self._writer is not None and step % self.every == 0

    @property
    def path(self):
        """Alias for `video`, matching `radial_sphere.render.VideoRecorder`."""
        return self.video

    def add(self, frame) -> None:
        """Write one already-composed frame. A no-op when recording is off."""
        if self._writer is not None:
            self._writer.append_data(np.asarray(frame))
            self.n_frames += 1

    def capture(self, step: int, panes, state: dict | None = None) -> None:
        """Render, annotate and write the panes, if this step is due."""
        if not self.due(step):
            return
        self.add(compose([render_pane(self.env, p, state) for p in panes]))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Tracker:
    """Live state and the standard metrics, accumulated one step at a time.

    Call :meth:`update` once per loop iteration, before stepping. It returns
    the live state dict that overlay text and feedback bindings both read.
    """

    def __init__(self, env, goal=None):
        self.env = env
        self.goal = goal if goal is not None else getattr(env.scenario, "goal", None)
        self.dt = float(env.model.opt.timestep * env.action_repeat)
        self.distance_x = 0.0
        self.path_length = 0.0
        self.core_impacts = 0
        self.steps = 0
        self._prev = env.data.qpos[0:2].copy()
        self._speeds: list[float] = []
        self._heights: list[float] = []

    def update(self, step: int | None = None) -> dict:
        env = self.env
        now = env.data.qpos[0:2].copy()
        moved = now - self._prev
        if moved[0] > 0:
            self.distance_x += float(moved[0])
        self.path_length += float(np.linalg.norm(moved))
        self._prev = now
        self.steps = int(step) if step is not None else self.steps + 1

        state = _state(env, self.steps, self.dt, self.distance_x, self.path_length,
                       self.goal)
        self._speeds.append(state["speed"])
        self._heights.append(state["ball_z"])
        self.core_impacts += int(any(
            env.core_geom_id in (env.data.contact[i].geom1, env.data.contact[i].geom2)
            for i in range(env.data.ncon)))
        return state

    @property
    def metrics(self) -> dict[str, float]:
        pos = self.env.data.qpos[0:3]
        return {
            "final_x": float(pos[0]), "final_y": float(pos[1]), "final_z": float(pos[2]),
            "distance_x": self.distance_x, "path_length": self.path_length,
            "mean_speed": float(np.mean(self._speeds)) if self._speeds else 0.0,
            "max_speed": float(np.max(self._speeds)) if self._speeds else 0.0,
            "min_z": float(np.min(self._heights)) if self._heights else 0.0,
            "max_z": float(np.max(self._heights)) if self._heights else 0.0,
            "core_impacts": float(self.core_impacts),
            "goal_distance": (float(np.linalg.norm(np.asarray(self.goal, dtype=float)[:2]
                                                   - pos[:2]))
                              if self.goal is not None else float("inf")),
            "steps_run": float(self.steps), "duration_s": self.steps * self.dt,
        }

    def check(self, expect) -> list[str]:
        return check_expectations(self.metrics, expect)

    def report(self, prefix: str = "    ") -> None:
        for key, value in self.metrics.items():
            print(f"{prefix}{key:14s} {value:9.3f}")


def run_demo(spec, *, video: bool | None = None, quiet: bool = False) -> DemoResult:
    """Run one demo spec and return its metrics and any failed expectations."""
    import mujoco

    from .config import load_config
    from .mujoco_env import MujocoRadialSphereEnv
    from .scenario import generate_scenario
    from skills import execute_skill

    name = str(spec.name)
    sc = spec.scenario
    cfg = load_config(sc.get("config"))
    if spec.get("camera_enabled") is False:
        cfg.camera.enabled = False
    seed = int(sc.get("seed", 42))
    scenario = generate_scenario(str(sc.kind), cfg, seed=seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=200_000)
    env.reset(seed=seed)

    spawn = spec.get("spawn")
    if spawn is not None:
        for axis, index in (("x", 0), ("y", 1), ("z", 2)):
            if axis in spawn:
                env.data.qpos[index] = float(spawn[axis])
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

    dt = float(env.model.opt.timestep * env.action_repeat)
    goal = getattr(scenario, "goal", None)
    skill = spec.skill
    skill_name = str(skill.name)

    from_scenario = _scenario_args(skill, scenario)
    settle_args = {**_skill_kwargs(skill, {}, feedback=False), **from_scenario}
    for _ in range(int(spec.get("settle", 0))):
        env.step(execute_skill(skill_name, env.data.qpos[3:7].copy(), env.dirs_body,
                               env.max_extend, **settle_args))

    record = spec.get("video", {}) or {}
    want_video = bool(record.get("enabled", True)) if video is None else video
    panes = record.get("panes") or [{"camera": "chase"}]
    recorder = Recorder(env, f"demo_{name}", tag=skill_name, enabled=want_video,
                        fps=int(record.get("fps", 25)), every=int(record.get("every", 4)),
                        out_name=name)
    tracker = Tracker(env, goal=goal)

    stop_when = spec.get("stop_when") or {}
    stop_any = list(spec.get("stop_when_any") or [])
    total = int(spec.steps)

    for step in range(1, total + 1):
        state = tracker.update(step)
        env.step(execute_skill(skill_name, env.data.qpos[3:7].copy(), env.dirs_body,
                               env.max_extend,
                               **{**_skill_kwargs(skill, state), **from_scenario}))
        recorder.capture(step, panes, state)

        if not quiet and step % max(total // 10, 1) == 0:
            print(f"  step {step:5d}/{total}  x={state['ball_x']:+.2f}  "
                  f"z={state['ball_z']:.3f}  speed={state['speed']:.2f}")

        # stop_when needs every bound to hold; stop_when_any needs one entry.
        if stop_when and not check_expectations(state, stop_when):
            if not quiet:
                print(f"  stop_when met at step {step}")
            break
        if stop_any and any(not check_expectations(state, one) for one in stop_any):
            if not quiet:
                print(f"  stop_when_any met at step {step}")
            break

    recorder.close()
    metrics = tracker.metrics
    run_dir, out_video, steps_run = recorder.run_dir, recorder.video, tracker.steps
    env.close()
    return DemoResult(name=name, metrics=metrics, failures=check_expectations(metrics, spec.get("expect")),
                      run_dir=run_dir, video=out_video, steps_run=steps_run)
