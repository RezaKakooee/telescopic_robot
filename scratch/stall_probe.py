import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco, collections
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import Bowl, advance_radius, wall_of_death, surface_frame
cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
sc=generate_scenario("motordrome",cfg,seed=1); md=sc.motordromes[0]
bowl=Bowl.from_motordrome(md)
env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
gname={i:(mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,i) or "") for i in range(env.model.ngeom)}
env.data.qpos[0]=0.55; env.data.qpos[1]=0; env.data.qpos[2]=0.22; env.data.qvel[:]=0
mujoco.mj_forward(env.model,env.data)
r_cmd=0.55
print(f"{'t':>5} {'r':>5} {'rcmd':>5} {'z':>5} {'v':>5} {'bank':>5} {'steer':>6} {'n_xy':>5} {'n_z':>5} {'contacts'}")
for i in range(2500):
    pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
    n,_=surface_frame(env.model,env.data,pos)
    tg,info=wall_of_death(env.data.qpos[3:7].copy(),env.dirs_body,env.max_extend,pos,vel,
                          r_cmd=r_cmd,bowl=bowl,normal=n,wall_radius=md[3],ccw=True,steer_gain=0.30)
    r_cmd=advance_radius(r_cmd,info["r"],info["speed"],bowl,open_rate=0.08)
    env.step(tg)
    if i%250==0:
        c=collections.Counter()
        for k in range(env.data.ncon):
            con=env.data.contact[k]
            nm=gname[con.geom1] if not gname[con.geom1].startswith("foot") else gname[con.geom2]
            c["_".join(nm.split("_")[:3]) if nm else "?"]+=1
        nn=n if n is not None else np.zeros(3)
        print(f"{i/100:5.1f} {info['r']:5.2f} {r_cmd:5.2f} {info['z']:5.2f} {info['speed']:5.2f} "
              f"{info['bank_deg']:5.1f} {info['steer']:+6.2f} {np.hypot(nn[0],nn[1]):5.2f} {nn[2]:5.2f} "
              f"{dict(c)}")
env.close()
