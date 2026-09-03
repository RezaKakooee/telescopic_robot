import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from scripts.skills.run_training_cones import run

print("Testing slalom parameters for 0 contacts and smooth weave:")
for speed in [1.2, 1.4, 1.6]:
    for lat in [0.75, 0.80, 0.85, 0.90]:
        for look in [0.65, 0.75, 0.85]:
            res = run(speed=speed, lateral_offset=lat, lookahead=look, record_video=False)
            print(f"speed={speed:.1f}, lat={lat:.2f}, look={look:.2f} -> cleared={res['n_cleared']}/10, contacts={res['contacts']}, min_dist={res['min_clearance']:.3f}m")
