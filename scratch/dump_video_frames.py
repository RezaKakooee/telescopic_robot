"""Inspect frames of 3_mechanisms_transparent_comparison.mp4 directly."""
import os
os.environ["MUJOCO_GL"] = "egl"
import imageio.v2 as imageio
from pathlib import Path

reader = imageio.get_reader("storage_local/mechanism_comparison/3_mechanisms_transparent_comparison.mp4")
out_dir = Path("storage_local/mechanism_comparison/video_frames")
out_dir.mkdir(parents=True, exist_ok=True)

# Save frames around takeoff (frames 45, 48, 50, 52)
for idx, frame in enumerate(reader):
    if idx in [45, 48, 50, 52]:
        imageio.imwrite(str(out_dir / f"frame_{idx:03d}.png"), frame)
        print(f"Saved frame {idx}")

print("Done extracting frames.")
