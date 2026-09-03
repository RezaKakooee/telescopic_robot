import os; os.environ["MUJOCO_GL"]="egl"
import pathlib, tempfile, re
import scripts.skills.run_motordrome_wall_of_death as R
from radial_sphere.config import load_config
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import Bowl
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
# target: smooth join to the wall, rounder barrel, rings crowded into the curl
T = dict(wall_radius=3.50, bowl_ride_end=3.10, bowl_lip_bank=78, bowl_blend=0.35,
         cylinder_height=3.90, bowl_facets=48, bowl_plank_lap=1.01,
         bowl_plank_pad=0.004, bowl_bank_weight=0.55)
print(f"{'rings':>6} {'weight':>7} {'depth':>6} | {'peak':>6} {'z_mean':>7} {'laps':>5}")
for ns, bw in ((22,0.55),(26,0.55),(30,0.90),(26,1.40)):
    path = cfg_with(bowl_segments=ns, **{**T, "bowl_bank_weight": bw})
    depth = Bowl.from_motordrome(generate_scenario("motordrome", load_config(path), seed=1).motordromes[0]).rim_z
    r = R.run(seconds=70, record_video=False, config=path)
    print(f"{ns:6d} {bw:7.2f} {depth:6.2f} | {r['peak_z']:6.2f} {r['z_mean']:7.2f} {r['laps']:5.1f}", flush=True)
    pathlib.Path(path).unlink()
