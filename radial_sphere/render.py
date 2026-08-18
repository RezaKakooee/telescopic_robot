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

    def render(self, state, camera_name: str | None = None) -> np.ndarray | None:
        """Return env 0's camera RGB as ``(H, W, 3)`` uint8, or ``None``.

        ``None`` is returned before the first ``reset()`` or if the named camera
        is absent (e.g. a headless build without cameras configured).
        """
        cam_name = camera_name or self.camera_name
        if state is None or cam_name not in state.cameras:
            return None
        rgb = state.cameras[cam_name].rgb[0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8) if rgb.max() <= 1.0 \
                else rgb.astype(np.uint8)
        return rgb

    def render_all(self, state) -> dict[str, np.ndarray]:
        """Return all available camera frames as a dict of ``{camera_name: frame}``."""
        if state is None or not getattr(state, "cameras", None):
            return {}
        frames = {}
        for cam_name in state.cameras:
            rgb = self.render(state, camera_name=cam_name)
            if rgb is not None:
                frames[cam_name] = rgb
        return frames


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


class MultiVideoRecorder:
    """Stream multiple synchronized camera frames to separate and combined MP4s."""

    def __init__(self, renders_dir, ep: int = 1, fps: int = 24,
                 camera_names: list[str] | None = None,
                 save_dual: bool = True):
        self.renders_dir = Path(renders_dir)
        self.ep = int(ep)
        self.fps = int(fps)
        self.camera_names = camera_names
        self.save_dual = save_dual
        self._recorders: dict[str, VideoRecorder] = {}
        self._dual_recorder: VideoRecorder | None = None
        self.n_frames = 0

    def add(self, frames: dict[str, np.ndarray] | None) -> None:
        if not frames:
            return
        for cam_name, frame in frames.items():
            if frame is None:
                continue
            if self.camera_names is not None and cam_name not in self.camera_names:
                continue
            if cam_name not in self._recorders:
                path = self.renders_dir / f"ep_{self.ep:03d}_{cam_name}.mp4"
                self._recorders[cam_name] = VideoRecorder(path, fps=self.fps)
            self._recorders[cam_name].add(frame)

        # Primary ep_XXX.mp4
        if "ep_default" not in self._recorders:
            default_path = self.renders_dir / f"ep_{self.ep:03d}.mp4"
            self._recorders["ep_default"] = VideoRecorder(default_path, fps=self.fps)

        pref_order = ["bird_fixed", "bird", "chase", "isometric"]
        default_frame = None
        for k in pref_order:
            if k in frames and frames[k] is not None:
                default_frame = frames[k]
                break
        if default_frame is None:
            default_frame = next((v for v in frames.values() if v is not None), None)
        if default_frame is not None:
            self._recorders["ep_default"].add(default_frame)

        # Dual side-by-side
        if self.save_dual and len(frames) >= 2:
            cam_list = [k for k in ["bird_fixed", "bird", "chase", "isometric"]
                        if k in frames and frames[k] is not None]
            if len(cam_list) >= 2:
                f1, f2 = frames[cam_list[0]], frames[cam_list[1]]
                if f1.shape[0] == f2.shape[0]:
                    dual_frame = np.concatenate([f1, f2], axis=1)
                else:
                    dual_frame = f1
                if self._dual_recorder is None:
                    self._dual_recorder = VideoRecorder(
                        self.renders_dir / f"ep_{self.ep:03d}_dual.mp4", fps=self.fps
                    )
                self._dual_recorder.add(dual_frame)

        self.n_frames += 1

    def close(self) -> None:
        for r in self._recorders.values():
            r.close()
        if self._dual_recorder is not None:
            self._dual_recorder.close()
            self._dual_recorder = None

    @property
    def paths(self) -> list[Path]:
        p = [r.path for r in self._recorders.values()]
        if self._dual_recorder is not None:
            p.append(self._dual_recorder.path)
        return p
