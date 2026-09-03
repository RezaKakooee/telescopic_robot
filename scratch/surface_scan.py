import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import Bowl
cfg=load_config("configs/rl/motordrome.yaml")
sc=generate_scenario("motordrome",cfg,seed=1); md=sc.motordromes[0]
bowl=Bowl.from_motordrome(md)
env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=5); env.reset(seed=1)
m,d=env.model,env.data
d.qpos[:3]=[50,50,50]; mujoco.mj_forward(m,d)     # park the robot far away
gid=np.zeros(1,dtype=np.int32); down=np.array([0.,0.,-1.])
print("Scan the built surface with downward rays. mid-plank vs plank seam.")
print(f"{'r':>5} {'design':>7} {'seam z':>7} {'mid z':>7} {'err':>7}")
worst=0
for rr in np.arange(1.05,3.25,0.10):
    zs=[]
    for lbl,ang in (("seam",0.0),("mid",np.pi/32)):
        p=np.array([rr*np.cos(ang), rr*np.sin(ang), 4.0])
        dist=mujoco.mj_ray(m,d,p,down,None,1,-1,gid)
        zs.append(4.0-dist if dist>=0 else np.nan)
    des=bowl.height_at(rr)
    err=max(abs(zs[0]-des),abs(zs[1]-des))
    worst=max(worst,err)
    print(f"{rr:5.2f} {des:7.3f} {zs[0]:7.3f} {zs[1]:7.3f} {err:7.3f}")
print(f"\nworst height error {worst*1000:.0f} mm")
env.close()
