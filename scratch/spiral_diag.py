import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import Bowl, advance_radius, wall_of_death, surface_frame
from skills.locomotion import move, surface_drive
cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
sc=generate_scenario("motordrome",cfg,seed=1); md=sc.motordromes[0]
bowl=Bowl.from_motordrome(md)

def go(mode, steps=900):
    env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
    env.data.qpos[0]=0.55; env.data.qpos[1]=0; env.data.qpos[2]=0.22; env.data.qvel[:]=0
    mujoco.mj_forward(env.model,env.data)
    r_cmd=0.55; out=[]
    for i in range(steps):
        pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
        q=env.data.qpos[3:7].copy()
        n,_=surface_frame(env.model,env.data,pos)
        th=np.arctan2(pos[1],pos[0]); t=np.array([-np.sin(th),np.cos(th),0.])
        if mode=="straight":                       # plain move, no circling
            tg=move(q,env.dirs_body,env.max_extend,np.array([0.,1.]),back_gain=4.0)
        elif mode=="tangent":                      # surface_drive along the tangent, no steer
            nn=n if n is not None else np.array([0.,0.,-1.])
            tg=surface_drive(q,env.dirs_body,env.max_extend,nn,t,speed=2.8)
        elif mode=="tangent_flat":                 # same but assume a flat floor
            tg=surface_drive(q,env.dirs_body,env.max_extend,np.array([0.,0.,-1.]),t,speed=2.8)
        else:
            tg,info=wall_of_death(q,env.dirs_body,env.max_extend,pos,vel,r_cmd=r_cmd,bowl=bowl,
                                  normal=n,wall_radius=md[3],ccw=True)
            r_cmd=advance_radius(r_cmd,info["r"],info["speed"],bowl)
        env.step(tg)
        nz = n[2] if n is not None else np.nan
        nxy = float(np.hypot(n[0],n[1])) if n is not None else np.nan
        out.append((float(np.linalg.norm(vel[:2])), float(np.hypot(pos[0],pos[1])),
                    float(pos[2]), nxy, nz, env.data.ncon))
    env.close(); return out

for mode in ("straight","tangent_flat","tangent","spiral"):
    o=go(mode)
    print(f"{mode:14s}", end="")
    for k in (100,300,600,899):
        v,r,z,nxy,nz,nc = o[k]
        print(f" | t={k/100:.0f}s v={v:4.2f} r={r:4.2f}", end="")
    nn=[x[3] for x in o if not np.isnan(x[3])]
    print(f" | mean |n_xy|={np.mean(nn):.2f}" if nn else " | no contact")
