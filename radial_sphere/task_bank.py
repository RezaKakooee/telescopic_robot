"""A fixed, reviewable set of maze tasks to train and evaluate against.

A task is two integers. `generate_scenario` is deterministic in the layout
seed, which draws the walls, and in the episode seed, which picks start and
goal. So the bank stores the recipe rather than the geometry: a few hundred
integer pairs instead of megabytes of wall coordinates, exactly reproducible
on any machine.

Why a bank and not a fresh random maze per episode:

* **A held-out split.** Layouts reserved from training let the eval afterwards
  say whether the policy navigates or memorised. A brand-new maze every reset
  has no such thing to hold back.
* **It is reviewable.** `tasks.json` can be read, sorted and filtered before a
  run starts, and the same 500 tasks come back next month.
* **Rebuilds are visible.** Switching task recompiles the MuJoCo scene, which
  measured 0.26 s against 0.01 s for a plain reset. A bank makes it a choice
  how often that is paid; `resample_every` spends one rebuild across several
  episodes.

Build one with `scripts/rl/build_maze_bank.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

#: Where `scripts/rl/build_maze_bank.py` writes by default.
DEFAULT_BANK = Path("storage_local/maze_task_bank/tasks.json")


@dataclass(frozen=True)
class Task:
    """One maze: which walls, and which corner to which corner."""

    layout_seed: int
    endpoint_seed: int
    route_m: float
    split: str

    @classmethod
    def from_record(cls, rec: dict) -> "Task":
        return cls(int(rec["layout_seed"]), int(rec["endpoint_seed"]),
                   float(rec.get("route_m", 0.0)), str(rec.get("split", "train")))


class TaskBank:
    """The tasks from one `tasks.json`, filtered to a split."""

    def __init__(self, path=DEFAULT_BANK, split: str | None = "train"):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"no task bank at {self.path}; build one with "
                "scripts/rl/build_maze_bank.py")
        raw = json.load(open(self.path))
        self.config = raw.get("config")
        tasks = [Task.from_record(r) for r in raw["tasks"]]
        self.tasks = [t for t in tasks if split is None or t.split == split]
        if not self.tasks:
            raise ValueError(f"{self.path} has no tasks in split {split!r}")
        self.split = split

    def __len__(self) -> int:
        return len(self.tasks)

    def sample(self, rng: np.random.Generator) -> Task:
        return self.tasks[int(rng.integers(len(self.tasks)))]

    def routes(self) -> np.ndarray:
        return np.array([t.route_m for t in self.tasks], dtype=float)

    def __repr__(self) -> str:
        r = self.routes()
        return (f"TaskBank({len(self)} tasks, split={self.split!r}, "
                f"routes {r.min():.1f}-{r.max():.1f} m)")


class BankedMazeEnv(gym.Wrapper):
    """Draw each episode's maze from a bank instead of at random.

    The wrapped env must have been built with ``randomize=True``, because that
    is what makes `MujocoRadialSphereEnv.reset` regenerate the scenario and
    rebuild the scene. This wrapper only decides *which* scenario that will be:
    it writes the task's layout seed onto the config the env reads, then resets
    with the task's endpoint seed.

    ``resample_every`` keeps the same task for that many episodes. Each switch
    costs a scene rebuild, so 1 is the most varied and the most expensive.
    """

    def __init__(self, env, bank: TaskBank, seed: int = 0, resample_every: int = 1):
        super().__init__(env)
        self.bank = bank
        self.resample_every = max(1, int(resample_every))
        self._rng = np.random.default_rng(seed)
        self._episodes = 0
        self._task: Task | None = None

    # The maze config lives on the innermost env, under whatever wrappers the
    # steering layer adds.
    def _maze_cfg(self):
        inner = self.env
        while not hasattr(inner, "cfg") and hasattr(inner, "env"):
            inner = inner.env
        return inner.cfg.scenario.maze

    @property
    def task(self) -> Task | None:
        return self._task

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self._task is None or self._episodes % self.resample_every == 0:
            self._task = self.bank.sample(self._rng)
        self._episodes += 1
        self._maze_cfg().layout_seed = self._task.layout_seed
        obs, info = self.env.reset(seed=self._task.endpoint_seed, options=options)
        info = dict(info)
        info["task_layout_seed"] = self._task.layout_seed
        info["task_endpoint_seed"] = self._task.endpoint_seed
        info["task_route_m"] = self._task.route_m
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._task is not None:
            info = dict(info)
            info["task_layout_seed"] = self._task.layout_seed
            info["task_route_m"] = self._task.route_m
        return obs, reward, terminated, truncated, info
