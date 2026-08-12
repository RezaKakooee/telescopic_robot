"""Rendering: pull an RGB frame from the handler's camera state.

Unlike ant_swarm (which rasterises a 2-D scene in pure NumPy), the radial-sphere
env renders through the RoboVerse MuJoCo handler's chase camera.  This module
isolates the "TensorState → uint8 image" conversion so the env just delegates::

    from radial_sphere.render import Renderer
    renderer = Renderer(cfg)
    frame = renderer.render(state)   # np.uint8 (H, W, 3) or None

It also provides :class:`VideoRecorder`, which streams frames straight to an
MP4 file.  metasim's ``ObsSaver`` buffers every frame in RAM until ``save()``
— at dense capture rates a long episode runs into per-user memory caps on the
cluster (OOM kill loses the whole video); streaming keeps memory constant.
"""
from __future__ import annotations

from pathlib import Path

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


class VideoRecorder:
    """Stream RGB frames to an MP4 with constant memory.

        rec = VideoRecorder(run_dir / "renders" / "ep_001.mp4", fps=24)
        rec.add(env.render())   # None frames are ignored
        rec.close()             # finalises the file; safe to call twice
    """

    def __init__(self, path, fps: int = 24):
        self.path = Path(path)
        self.fps = int(fps)
        self._writer = None
        self.n_frames = 0

    def add(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        if self._writer is None:
            import imageio.v2 as iio
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = iio.get_writer(self.path, fps=self.fps)
        self._writer.append_data(frame)
        self.n_frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
