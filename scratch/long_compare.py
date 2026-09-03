"""Is the ride steady over the long run, before and after smoothing?"""
import os; os.environ["MUJOCO_GL"]="egl"
import pathlib, tempfile, re, numpy as np
import scripts.skills.run_motordrome_wall_of_death as R
BASE = pathlib.Path("configs/rl/motordrome.yaml").read_text()
def cfg_with(**over):
    txt = BASE
    for k, v in over.items():
        if re.search(rf"^  {k}:", txt, re.M):
            txt = re.sub(rf"^  {k}:.*$", f"  {k}: {v}", txt, flags=re.M)
        else:
            txt = txt.replace("scenario:\n", f"scenario:\n  {k}: {v}\n", 1)
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", dir="configs/rl", delete=False)
    f.write(txt); f.close(); return f.name
OLD = dict(wall_radius=3.20, bowl_segments=22, bowl_facets=32, bowl_blend=0.0,
           bowl_lip_bank=0, bowl_plank_lap=1.06, bowl_plank_pad=0.02,
           bowl_ride_end=3.20, cylinder_height=3.20, bowl_bank_weight=0.0)
for name, over in (("old arena (hard join)", OLD), ("new arena (smooth join)", {})):
    path = cfg_with(**over)
    r = R.run(seconds=150, record_video=False, config=path)
    print(f"{name:24} peak {r['peak_z']:.2f}  last-8s {r['z_mean']:.2f}  laps {r['laps']:.1f}", flush=True)
    pathlib.Path(path).unlink()
