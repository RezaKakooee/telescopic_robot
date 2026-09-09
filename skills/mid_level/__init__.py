"""Mid-level skills: choose a low-level skill each step, and delegate.

`follow_path` and `stay_in_boundary` pick between stop, move, curve and
turn from the live state. `climb_stairs` sequences jumping, falling and
locomotion through its phases. That decision is what makes them mid-level.

Mid-level may import low-level. The reverse is forbidden, and
`tests/test_skill_levels.py` enforces it.
"""
