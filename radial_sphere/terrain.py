"""Named shapes for the terrain features a scenario carries.

``Scenario`` holds each feature as a list of bare tuples, and the shapes are
not obvious from a call site. ``stones`` is six numbers, or seven when the
seventh is a boolean asking for a circular scatter. ``gaps`` is five, with the
fifth optional. ``motordromes`` is eleven, four of them optional, two of those
guarded on truthiness rather than length. Nothing checked any of it, and a
reader had to go find the consumer to learn the order.

Every record here is a ``NamedTuple``, so it stays a tuple: existing code that
writes ``float(row[3])`` or ``len(row) > 4`` keeps working unchanged, while new
code can write ``row.half_y``. ``rows()`` accepts whatever a scenario happens
to hold - plain tuples, lists, numpy rows, or records - and returns records
with the computed defaults already filled in.

Lengths are half-extents unless a field says otherwise, matching MuJoCo's box
convention. All distances are metres, all angles degrees.
"""
from __future__ import annotations

from typing import NamedTuple


class Gap(NamedTuple):
    """A recessed pit cut into the floor."""

    x: float
    y: float
    half_x: float
    half_y: float
    depth: float = 0.10


class SandPatch(NamedTuple):
    """A high-friction slab lying flush on the floor."""

    x: float
    y: float
    half_x: float
    half_y: float


class StoneField(NamedTuple):
    """A scatter of procedural boulders inside one rectangle or disc."""

    x: float
    y: float
    half_x: float
    half_y: float
    count: int = 20
    max_size: float = 0.045
    #: Scatter inside the inscribed disc instead of the rectangle.
    circular: bool = False


class Step(NamedTuple):
    """A timber ledge standing on the floor. ``height`` is full, not half."""

    x: float
    y: float
    half_x: float
    half_y: float
    height: float


class Ramp(NamedTuple):
    """An incline or a flat plateau when ``pitch_deg`` is zero."""

    x: float
    y: float
    length: float
    width: float
    height_change: float
    pitch_deg: float
    yaw_deg: float


class Staircase(NamedTuple):
    """A flight of steps starting at ``(x, y)`` and running along ``yaw_deg``."""

    x: float
    y: float
    n_steps: int
    rise: float
    run: float
    width: float
    yaw_deg: float
    descending: bool = False


class Pipe(NamedTuple):
    """A hollow horizontal tube. ``outer_radius`` defaults just outside the bore."""

    x: float
    y: float
    length: float
    inner_radius: float
    outer_radius: float | None = None
    yaw_deg: float = 0.0

    def _resolve(self) -> "Pipe":
        if self.outer_radius is None:
            return self._replace(outer_radius=self.inner_radius + 0.015)
        return self


class VerticalCylinder(NamedTuple):
    """A hollow upright shaft."""

    x: float
    y: float
    height: float
    inner_radius: float
    outer_radius: float | None = None

    def _resolve(self) -> "VerticalCylinder":
        if self.outer_radius is None:
            return self._replace(outer_radius=self.inner_radius + 0.02)
        return self


class Motordrome(NamedTuple):
    """A wall-of-death bowl: flat floor, sloped apron, vertical wall.

    The last four fields keep the consumer's original guards. ``facets`` and
    ``plank_overlap`` fall back when falsy, not merely when absent, so zero is
    read as "use the default" exactly as before.
    """

    x: float
    y: float
    floor_radius: float
    wall_radius: float
    apron_height: float
    total_height: float
    friction: float = 1.35
    profile: str | None = None
    facets: int | None = None
    plank_overlap: float | None = None
    plank_pad: float | None = None
    #: Where the scripted ride ends. Read by the wall-of-death runner, not by
    #: the geometry builder.
    ride_end: float | None = None


class Cone(NamedTuple):
    """A traffic cone."""

    x: float
    y: float
    radius: float = 0.12


class Yardline(NamedTuple):
    """A painted stripe on the floor. ``rgba`` is a MuJoCo colour string."""

    x: float
    y: float
    half_x: float
    half_y: float
    rgba: str
    yaw_deg: float | None = None


def rows(record_cls, raw) -> tuple:
    """Normalise a scenario's raw feature list into `record_cls` records.

    Accepts ``None``, and rows given as tuples, lists, numpy rows or records.
    Computed defaults, such as a pipe's outer radius, are filled in here so
    every consumer sees the same resolved values.
    """
    if raw is None:
        return ()
    out = []
    for row in raw:
        record = row if isinstance(row, record_cls) else record_cls(*row)
        resolve = getattr(record, "_resolve", None)
        out.append(resolve() if resolve is not None else record)
    return tuple(out)
