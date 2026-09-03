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
W = dict(wall_radius=3.20, bowl_blend=0.0, bowl_lip_bank=0,
         bowl_ride_end=3.20, cylinder_height=3.20, bowl_bank_weight=0.0)
print(f"{'facets':>7} {'rings':>6} {'lap':>6} {'pad_mm':>7} | {'peak':>6} {'z_mean':>7} {'laps':>5}")
for nf, ns, lap, pad in [(32,22,1.06,20),(48,22,1.01,4),(48,40,1.01,4),
                         (64,40,1.005,2),(48,40,1.005,2)]:
    path = cfg_with(bowl_facets=nf, bowl_segments=ns, bowl_plank_lap=lap,
                    bowl_plank_pad=pad/1000.0, **W)
    r = R.run(seconds=70, record_video=False, config=path)
    print(f"{nf:7d} {ns:6d} {lap:6.3f} {pad:7d} | {r['peak_z']:6.2f} {r['z_mean']:7.2f} {r['laps']:5.1f}", flush=True)
    pathlib.Path(path).unlink()
