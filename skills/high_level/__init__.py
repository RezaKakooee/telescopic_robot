"""High-level skills: emit skill commands, never rod targets.

For planning and RL policies that choose which skill to run and with what
arguments, rather than computing rod extensions themselves.

The contract, so this folder does not become a place for leftovers:

* Input  : task state, not rod geometry.
* Output : a skill name plus its arguments, for `execute_skill`.
* Imports: may use mid_level and low_level. Neither may import this.

`goal_seeking.go_to_goal` is the first one. It reads a map and a goal and
answers with a skill name, so it never sees `dirs_body` or `max_extend`.
Nothing here belongs in `SKILL_REGISTRY`: every name in that registry
returns rod targets, and a planner returns a command instead.
"""

from .goal_seeking import (
    DEFAULT_CLEARANCE,
    MAX_DETOURS,
    SkillCommand,
    climbable_height,
    go_to_goal,
    plan_route,
)

__all__ = ["SkillCommand", "go_to_goal", "plan_route", "climbable_height",
           "DEFAULT_CLEARANCE", "MAX_DETOURS"]
