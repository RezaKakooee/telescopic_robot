import os; os.environ["MUJOCO_GL"]="egl"
from omegaconf import OmegaConf
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import Bowl
import scripts.skills.run_motordrome_wall_of_death as R
import tempfile, yaml, pathlib
base=pathlib.Path("configs/rl/motordrome.yaml").read_text()
print(f"{'rim':>5} {'rimv':>5} {'depth':>6} | {'peak_z':>7} {'z_mean':>7} {'v_mean':>7} {'laps':>5}")
for rim, rimv in ((3.20,5.00),(3.80,5.20),(4.40,5.35)):
    txt=base.replace("wall_radius: 3.20","wall_radius: %.2f"%rim).replace("bowl_rim_speed: 5.00","bowl_rim_speed: %.2f"%rimv)
    f=tempfile.NamedTemporaryFile("w",suffix=".yaml",dir="configs/rl",delete=False)
    f.write(txt); f.close()
    cfg=load_config(f.name); sc=generate_scenario("motordrome",cfg,seed=1)
    b=Bowl.from_motordrome(sc.motordromes[0])
    r=R.run(seconds=70, record_video=False, config=f.name)
    print(f"{rim:5.2f} {rimv:5.2f} {b.rim_z:6.2f} | {r['peak_z']:7.2f} {r['z_mean']:7.2f} {r['peak_v']:7.2f} {r['laps']:5.1f}")
    pathlib.Path(f.name).unlink()
