"""Least smoothing that keeps the steady ride. 150 s each."""
import os; os.environ["MUJOCO_GL"]="egl"
import pathlib, tempfile, re
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
# everything below keeps the lip and the soft-min; only the plank count varies
CASES = [
    ("32 planks, 26 rings, old lap/pad",
     dict(bowl_facets=32, bowl_segments=26, bowl_plank_lap=1.06, bowl_plank_pad=0.02)),
    ("48 planks, 26 rings, old lap/pad",
     dict(bowl_facets=48, bowl_segments=26, bowl_plank_lap=1.06, bowl_plank_pad=0.02)),
    ("32 planks, 30 rings, thin lap/pad",
     dict(bowl_facets=32, bowl_segments=30, bowl_plank_lap=1.01, bowl_plank_pad=0.004)),
]
for name, over in CASES:
    path = cfg_with(**over)
    r = R.run(seconds=150, record_video=False, config=path)
    print(f"{name:36} peak {r['peak_z']:.2f}  last-8s {r['z_mean']:.2f}  laps {r['laps']:.1f}", flush=True)
    pathlib.Path(path).unlink()
