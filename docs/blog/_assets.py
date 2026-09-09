"""Where a blog render writes its output.

Every render script wrote to ``docs/blog/assets`` through the same hardcoded
line, so there was no way to regenerate the set without overwriting the
published files, and no way to run two renders at once safely.

``BLOG_ASSETS_DIR`` overrides the destination. Nothing else changes: the
file names, the csv and the json summary are the same, so a regenerated set
can be compared against the published one side by side.

    BLOG_ASSETS_DIR=/tmp/regen python docs/blog/render_wall_push.py
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / "assets"


def assets_dir() -> Path:
    """The directory this render should write into, created if needed."""
    out = Path(os.environ.get("BLOG_ASSETS_DIR") or DEFAULT)
    out.mkdir(parents=True, exist_ok=True)
    return out
