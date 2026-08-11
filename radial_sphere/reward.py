"""Reward component.

Dense progress shaping toward the end of the path plus a sparse success bonus::

    reward = progress_coef * (prev_dist - dist) + (success if reached else 0)

where ``dist`` is the distance from the sphere to the path's end point and
``reached`` means that distance dropped below ``goal_eps``.
"""
from __future__ import annotations


class RewardModel:
    def __init__(self, cfg):
        self.progress_coef = float(cfg.reward.progress_coef)
        self.success = float(cfg.reward.success)
        self.goal_eps = float(cfg.reward.goal_eps)

    def compute(self, dist: float, prev_dist: float):
        """Return ``(reward, reached)`` for one transition."""
        progress = prev_dist - dist
        reached = dist < self.goal_eps
        reward = self.progress_coef * progress + (self.success if reached else 0.0)
        return reward, reached
