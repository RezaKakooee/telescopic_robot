"""Low-level skills: state in, rod targets out.

Each function here computes one behaviour and never chooses between
behaviours. Several call `move` internally to shape a heading, but they
always call it: nothing here branches on state to pick a different skill.

A low-level skill never imports a scenario, a camera or a run directory.
It is a pure function of orientation, rod directions and a command.
"""
