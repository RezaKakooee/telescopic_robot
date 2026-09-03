"""One change at a time, from the arena that worked, to find what broke it."""
import os; os.environ["MUJOCO_GL"]="egl"
import pathlib, tempfile
import scripts.skills.run_motordrome_wall_of_death as R
from radial_sphere.config import load_config
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import Bowl

BASE = pathlib.Path("configs/rl/motordrome.yaml").read_text()
def cfg_with(**over):
    txt = BASE
    for k, v in over.items():
        import re
        if re.search(rf"^  {k}:", txt, re.M):
            txt = re.sub(rf"^  {k}:.*$", f"  {k}: {v}", txt, flags=re.M)
        else:
            txt = txt.replace("scenario:\n", f"scenario:\n  {k}: {v}\n", 1)
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", dir="configs/rl", delete=False)
    f.write(txt); f.close(); return f.name

WORKING = dict(wall_radius=3.20, bowl_segments=22, bowl_facets=32,
               bowl_blend=0.0, bowl_lip_bank=0, bowl_plank_lap=1.06,
               bowl_ride_end=3.20, cylinder_height=3.20, bowl_bank_weight=0.0)
CASES = [
    ("A baseline (the arena that worked)", {}),
    ("B + 48 facets",                      dict(bowl_facets=48)),
    ("C + 40 rings, even in radius",       dict(bowl_segments=40)),
    ("D + rings crowded into the curve",   dict(bowl_segments=40, bowl_bank_weight=0.55)),
    ("E + soft-min joins",                 dict(bowl_blend=0.35)),
    ("F + planks meeting end to end",      dict(bowl_plank_lap=1.001)),
]
print(f"{'case':38} {'depth':>6} {'peak':>6} {'z_mean':>7} {'v_mean':>7} {'laps':>5}")
for name, extra in CASES:
    path = cfg_with(**{**WORKING, **extra})
    sc = generate_scenario("motordrome", load_config(path), seed=1)
    depth = Bowl.from_motordrome(sc.motordromes[0]).rim_z
    r = R.run(seconds=70, record_video=False, config=path)
    print(f"{name:38} {depth:6.2f} {r['peak_z']:6.2f} {r['z_mean']:7.2f} "
          f"{r['peak_v']:7.2f} {r['laps']:5.1f}", flush=True)
    pathlib.Path(path).unlink()
