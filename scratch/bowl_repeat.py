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
WORKING = dict(wall_radius=3.20, bowl_segments=22, bowl_facets=32,
               bowl_blend=0.0, bowl_lip_bank=0, bowl_plank_lap=1.06,
               bowl_ride_end=3.20, cylinder_height=3.20, bowl_bank_weight=0.0)
SEEDS = (42, 7, 13, 91)
print(f"{'case':26} " + " ".join(f"{s:>6}" for s in SEEDS) + f" {'mean':>6} {'min':>6}")
for name, extra in [("A 32 facets, 22 rings", {}),
                    ("B 48 facets, 22 rings", dict(bowl_facets=48)),
                    ("C 48 facets, 40 rings", dict(bowl_facets=48, bowl_segments=40, bowl_bank_weight=0.55))]:
    path = cfg_with(**{**WORKING, **extra}); vals=[]
    for s in SEEDS:
        vals.append(R.run(seconds=70, record_video=False, config=path, seed=s)["z_mean"])
    print(f"{name:26} " + " ".join(f"{v:6.2f}" for v in vals) +
          f" {np.mean(vals):6.2f} {min(vals):6.2f}", flush=True)
    pathlib.Path(path).unlink()
