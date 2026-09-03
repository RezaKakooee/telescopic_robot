"""Test stairs and hurdle jumping with tuned explosive actuators."""
import os
os.environ["MUJOCO_GL"] = "egl"
from scripts.skills import run_stairs

# Run stairs with video=False for quick verification
res = run_stairs.run(seed=42, record_video=False)
print("Stairs Result:", res)
