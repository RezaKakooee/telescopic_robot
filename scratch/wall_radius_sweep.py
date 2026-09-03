"""A vertical wall needs v = sqrt(g*r/mu). Smaller cylinder, slower ride.
Does a narrow vertical cylinder hold where a 1.8 m one does not?"""
import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat
from skills.wall_of_death import surface_frame, reach_caps, FOOT_BASE
from skills.locomotion import surface_drive

def hold(wall_r, vmult, ext=0.09, steps=800, z0=1.6, cap_on=True):
    cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
    cfg.scenario.wall_radius=float(wall_r)
    cfg.scenario.floor_radius=float(max(0.25,wall_r-0.45))
    cfg.scenario.apron_height=0.45; cfg.scenario.cylinder_height=3.0
    sc=generate_scenario("motordrome",cfg,seed=1); mu=sc.motordromes[0][6]
    env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
    E=env.max_extend; ROLL=FOOT_BASE+ext; cx=wall_r-ROLL
    v_need=np.sqrt(9.81*cx/mu); v0=vmult*v_need
    env.data.qpos[0]=cx; env.data.qpos[1]=0; env.data.qpos[2]=z0
    env.data.qpos[3:7]=[1,0,0,0]; env.data.qvel[:]=0; env.data.qvel[1]=v0
    env.data.qpos[7:7+60]=ext; env.data.qvel[5]=-v0/ROLL
    mujoco.mj_forward(env.model,env.data)
    zs=[];vs=[]
    for i in range(steps):
        pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
        n,d=surface_frame(env.model,env.data,pos)
        th=np.arctan2(pos[1],pos[0]); rh=np.array([np.cos(th),np.sin(th),0.])
        if n is None: n=rh
        t=np.array([-np.sin(th),np.cos(th),0.])
        cap=None
        if cap_on:
            dw=env.dirs_body@quat_to_rotmat(env.data.qpos[3:7]).T
            cap=reach_caps(pos,dw,E,wall_radius=wall_r,margin=0.008)
        tg=surface_drive(env.data.qpos[3:7].copy(),env.dirs_body,E,n,t,speed=2.8,reach_cap=cap)
        env.step(tg); zs.append(float(pos[2])); vs.append(float(np.linalg.norm(vel[:2])))
        if pos[2]<0.50: break
    env.close()
    return dict(v_need=v_need,v0=v0,n=len(zs),dz=zs[-1]-z0,v_end=vs[-1])

print("Vertical wall, radius sweep. Placed at z=1.6, 8 s. dz near 0 = it holds.\n")
print(f"{'wall_r':>7} {'v_need':>7} {'x need':>7} {'v0':>5} {'steps':>6} {'dz':>7} {'v_end':>6}")
for wr in (0.7, 0.9, 1.1, 1.4, 1.8):
    for vm in (1.3, 2.0):
        r=hold(wr,vm)
        print(f"{wr:7.2f} {r['v_need']:7.2f} {vm:7.1f} {r['v0']:5.2f} {r['n']:6d} {r['dz']:+7.2f} {r['v_end']:6.2f}")
