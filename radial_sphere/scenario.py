"""Scenarios: what task the agent is asked to do.

A :class:`Scenario` is a small, serialisable spec describing one task instance:
where the sphere starts, where the goal is, the waypoints the controller/obs
track, and the breadcrumb markers to render.  Two kinds are supported:

* ``path``      — path navigation: follow a sinusoidal path to its end.
* ``goal``      — goal finding: reach a single goal point (random within an
                  arena); the waypoints are just a straight line spawn → goal.
* ``roundtrip`` — out-and-back: sine out, turnaround, return lane back beside
                  the spawn — the ball comes back toward the chase camera.

Generators:

    from radial_sphere import generate_scenario
    sc = generate_scenario("goal", cfg, seed=0)
    sc.save("scenario.json"); Scenario.load("scenario.json")
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import sample_path, sample_roundtrip

KINDS = ("path", "goal", "roundtrip")


def _arc_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


@dataclass
class Scenario:
    """One task instance for the agent (see module docstring)."""

    kind: str                # "path" | "goal"
    name: str
    spawn_xy: np.ndarray     # (2,) start position
    goal: np.ndarray         # (2,) target (== path_pts[-1])
    path_pts: np.ndarray     # (N, 2) waypoints the controller/obs track
    markers: np.ndarray      # (M, 2) breadcrumb markers to render (may be empty)
    path_length: float       # arc length, for normalising goal-distance in obs

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "spawn_xy": np.asarray(self.spawn_xy, dtype=float).tolist(),
            "goal": np.asarray(self.goal, dtype=float).tolist(),
            "path_pts": np.asarray(self.path_pts, dtype=float).tolist(),
            "markers": np.asarray(self.markers, dtype=float).reshape(-1, 2).tolist(),
            "path_length": float(self.path_length),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            kind=d["kind"],
            name=d["name"],
            spawn_xy=np.asarray(d["spawn_xy"], dtype=np.float32),
            goal=np.asarray(d["goal"], dtype=np.float32),
            path_pts=np.asarray(d["path_pts"], dtype=np.float32),
            markers=np.asarray(d["markers"], dtype=np.float32).reshape(-1, 2),
            path_length=float(d["path_length"]),
        )

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path) -> "Scenario":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def path_scenario(cfg, *, rng=None, name: str = "path") -> Scenario:
    """Path navigation: the sinusoidal path from the config ``path`` section."""
    p = cfg.path
    pts = sample_path(240, p.length, p.amplitude, p.waves).astype(np.float32)
    markers = sample_path(40, p.length, p.amplitude, p.waves).astype(np.float32)
    return Scenario(
        kind="path", name=name,
        spawn_xy=pts[0].copy(), goal=pts[-1].copy(),
        path_pts=pts, markers=markers, path_length=_arc_length(pts),
    )


def goal_scenario(cfg, *, rng=None, name: str = "goal") -> Scenario:
    """Goal finding: a random goal in the arena; waypoints are a straight line."""
    rng = rng if rng is not None else np.random.default_rng()
    sc = getattr(cfg, "scenario", None)
    goal_cfg = getattr(sc, "goal", None) if sc is not None else None
    x_range = tuple(getattr(goal_cfg, "x_range", (2.0, 5.0))) if goal_cfg else (2.0, 5.0)
    y_range = tuple(getattr(goal_cfg, "y_range", (-1.5, 1.5))) if goal_cfg else (-1.5, 1.5)

    spawn = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([rng.uniform(*x_range), rng.uniform(*y_range)], dtype=np.float32)
    # Straight-line waypoints so the look-ahead controller heads to the goal.
    pts = np.linspace(spawn, goal, 60).astype(np.float32)
    # No breadcrumbs for goal finding — only the goal marker is shown.
    markers = np.empty((0, 2), dtype=np.float32)
    return Scenario(
        kind="goal", name=name,
        spawn_xy=spawn, goal=goal,
        path_pts=pts, markers=markers, path_length=float(np.linalg.norm(goal - spawn)),
    )


def roundtrip_scenario(cfg, *, rng=None, name: str = "roundtrip") -> Scenario:
    """Out-and-back: follow the sine out, turn around, return beside the spawn."""
    p = cfg.path
    lane = float(getattr(p, "lane_offset", 3.0 * float(p.amplitude)))
    pts = sample_roundtrip(480, p.length, p.amplitude, p.waves, lane).astype(np.float32)
    markers = pts[::8].copy()
    return Scenario(
        kind="roundtrip", name=name,
        spawn_xy=pts[0].copy(), goal=pts[-1].copy(),
        path_pts=pts, markers=markers, path_length=_arc_length(pts),
    )


_GENERATORS = {"path": path_scenario, "goal": goal_scenario,
               "roundtrip": roundtrip_scenario}


def generate_scenario(kind: str, cfg, *, seed=None, name: str | None = None) -> Scenario:
    """Generate a scenario of the given ``kind`` (``"path"`` | ``"goal"``)."""
    if kind not in _GENERATORS:
        raise ValueError(f"unknown scenario kind {kind!r}; expected one of {KINDS}")
    rng = np.random.default_rng(seed)
    return _GENERATORS[kind](cfg, rng=rng, name=name or kind)
