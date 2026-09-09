"""Draw a text panel onto a rendered frame.

Presentation only, no physics. A combined skill video is hard to read without
labels: from a standstill, "move forward" and "reverse" look the same, and
"go slow" only means something next to "go fast". Stamping the active skill
and the live speed onto each frame makes the difference visible.
"""
from __future__ import annotations

import numpy as np

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]
_font_cache: dict[int, object] = {}


def _font(size: int):
    if size not in _font_cache:
        from PIL import ImageFont
        for path in _FONT_PATHS:
            try:
                _font_cache[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def annotate(frame: np.ndarray, title: str, lines: list[str] | None = None,
             *, margin: int = 16, alpha: int = 165) -> np.ndarray:
    """Return *frame* with a title and optional detail lines drawn top-left.

    The text sits on a translucent dark panel so it stays readable over both
    the light and dark squares of the floor grid.
    """
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).convert("RGBA")
    panel = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)

    title_font = _font(max(18, img.height // 18))
    line_font = _font(max(13, img.height // 28))
    lines = lines or []

    # Panel size from the widest rendered line.
    widths = [draw.textbbox((0, 0), title, font=title_font)[2]]
    widths += [draw.textbbox((0, 0), t, font=line_font)[2] for t in lines]
    line_h = draw.textbbox((0, 0), "Ag", font=line_font)[3]
    title_h = draw.textbbox((0, 0), "Ag", font=title_font)[3]
    box_w = max(widths) + 2 * margin
    box_h = title_h + len(lines) * (line_h + 4) + 2 * margin

    draw.rectangle([0, 0, box_w, box_h], fill=(12, 14, 18, alpha))
    draw.text((margin, margin - 2), title, font=title_font, fill=(255, 255, 255, 255))
    y = margin + title_h + 4
    for text in lines:
        draw.text((margin, y), text, font=line_font, fill=(190, 214, 240, 255))
        y += line_h + 4

    return np.asarray(Image.alpha_composite(img, panel).convert("RGB"))
