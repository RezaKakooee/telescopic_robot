"""Reward component.

Dense progress shaping toward the end of the path plus a sparse success bonus::

    reward = progress_coef * (prev_dist - dist) + (success if reached else 0)

where ``dist`` is the navigation distance to the goal and ``reached`` is
supplied by the environment only when the robot physically contacts the goal.
"""
from __future__ import annotations


class RewardModel:
    def __init__(self, cfg):
        self.progress_coef = float(cfg.reward.progress_coef)
        self.success = float(cfg.reward.success)
        self.collision_cost = float(getattr(cfg.reward, "collision_cost", 0.0))

    def compute(self, dist: float, prev_dist: float, *, reached: bool = False,
                wall_contact: bool = False):
        """Return ``(reward, reached)`` for one transition."""
        progress = prev_dist - dist
        reached = bool(reached)
        reward = self.progress_coef * progress + (self.success if reached else 0.0)
        if wall_contact and self.collision_cost > 0.0:
            reward -= self.collision_cost
        return reward, reached

