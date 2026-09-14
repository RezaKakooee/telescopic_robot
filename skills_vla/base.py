"""Base definitions and contract for VLA-compatible skills.

Each skill in `skills_vla` is an independent module adhering to the `VLASkill`
protocol, with normalized parameter scaling in [-1, 1] and egocentric reference
for vision-language-action policies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from radial_sphere.gait import MIN_OFFSET


@dataclass(frozen=True)
class RobotState:
    """Proprioceptive robot state passed to all VLA skills."""

    quat: np.ndarray                        # (4,) body orientation, wxyz
    dirs_body: np.ndarray                   # (n_bars, 3) rod directions, body frame
    max_extend: float                       # stroke ceiling, metres
    lin_vel: np.ndarray | None = None       # (2,) or (3,) world velocity
    core_z: float | None = None             # world height of the core
    core_vz: float | None = None            # vertical speed of the core
    contact_forces: np.ndarray | None = None       # (n_bars,) newtons
    terrain_clearances: np.ndarray | None = None   # (n_bars,) signed metres
    surface_normal: np.ndarray | None = None       # (3,) normal vector
    min_offset: float = MIN_OFFSET          # baseline retracted rod length
    profile: str = "ideal"                  # "ideal" or "hardware"
    rod_mechanism: str = "multi_stage"      # selects the `skills` speed calibration
    ground_z: float = 0.0                   # local terrain height under the core (jump phases)

    @property
    def n_bars(self) -> int:
        return len(self.dirs_body)

    def floor_normal(self) -> np.ndarray:
        """The surface normal being ridden, defaulting to flat ground."""
        if self.surface_normal is None:
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
        n = np.asarray(self.surface_normal, dtype=np.float64)
        mag = float(np.linalg.norm(n))
        return n / mag if mag > 1e-9 else np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass(frozen=True)
class ParamSpec:
    """One tunable skill parameter bounded in [low, high].

    VLA networks predict normalized unit values in [-1, 1].
    `.scale(u)` maps [-1, 1] to [low, high].
    `.unscale(val)` maps [low, high] back to [-1, 1].
    """

    name: str
    low: float
    high: float
    default: float
    doc: str = ""

    def scale(self, unit_val: float) -> float:
        """Convert normalized [-1, 1] value to physical [low, high]."""
        u = float(np.clip(unit_val, -1.0, 1.0))
        return self.low + 0.5 * (u + 1.0) * (self.high - self.low)

    def unscale(self, phys_val: float) -> float:
        """Convert physical [low, high] value to normalized [-1, 1]."""
        if abs(self.high - self.low) < 1e-9:
            return 0.0
        c = float(np.clip(phys_val, self.low, self.high))
        return float(np.clip(2.0 * (c - self.low) / (self.high - self.low) - 1.0, -1.0, 1.0))


@dataclass
class SkillResult:
    """Output from executing a single control substep of a skill."""

    targets: np.ndarray         # (n_bars,) float32 rod extension targets
    done: bool = False          # True if atomic multi-step sequence is complete
    info: dict[str, Any] = None # diagnostic metadata (phase, power, etc.)


class VLASkill(ABC):
    """Abstract Base Class for all VLA skills."""

    name: str = "base"
    summary: str = "base skill"
    params: tuple[ParamSpec, ...] = ()
    is_multi_step: bool = False

    def scale_params(self, normalized_vector: np.ndarray | list[float]) -> dict[str, float]:
        """Convert an array of [-1, 1] action values into physical kwargs."""
        vec = np.asarray(normalized_vector, dtype=np.float64).reshape(-1)
        scaled = {}
        for i, p in enumerate(self.params):
            val = vec[i] if i < len(vec) else 0.0
            scaled[p.name] = p.scale(val)
        return scaled

    @abstractmethod
    def act(
        self,
        state: RobotState,
        camera_heading: float = 0.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute single-step rod extension targets (shape: (n_bars,))."""
        pass

    def step(
        self,
        state: RobotState,
        substep: int,
        camera_heading: float = 0.0,
        **kwargs: Any,
    ) -> SkillResult:
        """Execute one control step of a multi-step sequence."""
        targets = self.act(state, camera_heading=camera_heading, **kwargs)
        return SkillResult(targets=targets, done=True)
