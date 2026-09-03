import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco, time
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat
cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
sc=generate_scenario("motordrome",cfg,seed=1)
env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
m,d=env.model,env.data
# which bodies belong to the robot?
names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(m.nbody)]
robot=[i for i,n in enumerate(names) if n and (n=="core" or n.startswith("inner_"))]
print("nbody",m.nbody,"robot bodies",len(robot),"| core id",env.core_body_id,"| sample",names[:4])
d.qpos[0]=1.536; d.qpos[1]=0; d.qpos[2]=1.60; d.qpos[3:7]=[1,0,0,0]; d.qvel[:]=0
mujoco.mj_forward(m,d)
core=np.array(d.qpos[:3]); dw=env.dirs_body@quat_to_rotmat(d.qpos[3:7]).T
gid=np.zeros(1,dtype=np.int32)
t0=time.time()
hits=[]
for i,u in enumerate(dw):
    dist=mujoco.mj_ray(m,d,core,np.ascontiguousarray(u),None,1,env.core_body_id,gid)
    hits.append((dist,int(gid[0])))
dt=time.time()-t0
gn=lambda g: mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) if g>=0 else "-"
print(f"60 rays in {dt*1000:.2f} ms  ({dt*1000*3000/1000:.1f} s per 3000-step run)")
rad=[(i,h[0],gn(h[1])) for i,h in enumerate(hits) if h[0]>=0]
print(f"hits: {len(rad)}/60")
for i,dist,nm in rad[:6]: print(f"  rod {i:2d} dist={dist:.3f} geom={nm}")
bad=[nm for _,_,nm in rad if nm and (nm.startswith("foot_") or nm.startswith("sleeve") or nm=="core_geom")]
print("robot self-hits:", len(bad))
