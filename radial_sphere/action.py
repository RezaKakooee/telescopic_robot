"""Action component: agent action → per-bar extension targets.

The agent emits one normalised value per telescoping bar in ``[-1, 1]``, which
maps linearly to a physical extension in ``[0, max_extend]`` metres.  ``decode``
turns an action into the ``dof_pos_target`` dict the RoboVerse handler expects;
``encode`` is its inverse (used to feed a scripted controller's metre-valued
targets back through the same normalised action interface).
"""
from __future__ import annotations

import numpy as np

from ._gym import spaces


class ActionModel:
    def __init__(self, cfg):
        self.n_bars = int(cfg.robot.n_bars)
        self.max_extend = float(cfg.robot.max_extend)
        self.slide_names = [f"slide_{k}" for k in range(self.n_bars)]

    def space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self.n_bars,), dtype=np.float32)

    # ------------------------------------------------------------------
    def decode(self, action) -> dict:
        """Action in ``[-1, 1]^n`` → ``{slide_k: extension_metres}`` dict."""
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        targets = (a + 1.0) * 0.5 * self.max_extend
        return {self.slide_names[k]: float(targets[k]) for k in range(self.n_bars)}

    def encode(self, targets) -> np.ndarray:
        """Per-bar extension targets (metres) → normalised action in ``[-1, 1]``."""
        t = np.asarray(targets, dtype=np.float32).reshape(-1)
        return np.clip(t / self.max_extend * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)
