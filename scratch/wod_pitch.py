"""Does driving UP the wall hold the ball better than driving along it?

Friction is the only thing that can push a body up a vertical wall, and the
friction that holds a free ball also spins it, so the ball rolls down. The
cure is to roll it up on purpose: aim the wave along +z, not along the tangent.
"""
import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import wall_of_death, surface_frame, FOOT_BASE
cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
sc=generate_scenario("motordrome",cfg,seed=1); md=sc.motordromes[0]
wall_r,mu=md[3],md[6]

def trial(v0, pitch, press=0.0, tm=0.008, sp=2.8, z0=1.60, steps=800, ext=0.35):
    env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
    E=env.max_extend; R=FOOT_BASE+ext*E; r0=wall_r-R
    env.data.qpos[0]=r0; env.data.qpos[1]=0; env.data.qpos[2]=z0
    env.data.qpos[3:7]=[1,0,0,0]; env.data.qvel[:]=0
    env.data.qvel[1]=v0; env.data.qvel[5]=-v0/R
    mujoco.mj_forward(env.model,env.data)
    zs=[];vt=[];nc=[]
    for i in range(steps):
        pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
        n,d=surface_frame(env.model,env.data,pos)
        tg,info=wall_of_death(env.data.qpos[3:7].copy(),env.dirs_body,E,pos,vel,normal=n,
                              surface_dist=d,wall_radius=wall_r,wall_friction=mu,ccw=True,
                              press=press,speed=sp,touch_margin=tm,force_pitch=pitch)
        env.step(tg); zs.append(info["z"]); vt.append(info["v_t"]); nc.append(env.data.ncon)
        if info["z"]<0.45 or info["z"]>2.9: break
    env.close()
    return dict(n=len(zs),z_end=zs[-1],dz=zs[-1]-z0,vt_end=vt[-1],ncon=float(np.mean(nc)))

print("Aim the wave up the wall. pitch = vertical / tangential in the drive direction.")
print("dz > 0 means it climbed.\\n")
print(f"{'v0':>4} {'pitch':>7} {'press':>6} {'steps':>6} {'z_end':>6} {'dz':>7} {'vt_end':>7} {'ncon':>5}")
for v0 in (3.4, 4.2):
    for pitch in (0.0, 0.5, 1.0, 2.0, 5.0, 30.0):
        r=trial(v0,pitch)
        print(f"{v0:4.1f} {pitch:7.1f} {0.0:6.2f} {r['n']:6d} {r['z_end']:6.2f} {r['dz']:+7.2f} {r['vt_end']:7.2f} {r['ncon']:5.1f}")
