"""Rendering: pull an RGB frame from the handler's camera state.

Unlike ant_swarm (which rasterises a 2-D scene in pure NumPy), the radial-sphere
env renders through the RoboVerse MuJoCo handler's chase camera.  This module
isolates the "TensorState → uint8 image" conversion so the env just delegates::

    from radial_sphere.render import Renderer
    renderer = Renderer(cfg)
    frame = renderer.render(state)   # np.uint8 (H, W, 3) or None
"""
from __future__ import annotations

import numpy as np


class Renderer:
    def __init__(self, cfg, camera_name: str = "chase"):
        self.camera_name = camera_name

    def render(self, state) -> np.ndarray | None:
        """Return env 0's camera RGB as ``(H, W, 3)`` uint8, or ``None``.

        ``None`` is returned before the first ``reset()`` or if the named camera
        is absent (e.g. a headless build without cameras configured).
        """
        if state is None or self.camera_name not in state.cameras:
            return None
        rgb = state.cameras[self.camera_name].rgb[0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8) if rgb.max() <= 1.0 \
                else rgb.astype(np.uint8)
        return rgb
