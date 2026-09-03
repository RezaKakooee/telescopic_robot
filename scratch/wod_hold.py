"""Hold test with rods stopped at the surface."""
import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco, itertools
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import wall_of_death, surface_frame, FOOT_BASE

def trial(v0, z0=1.60, climb=0.0, press=0.0, tm=0.008, sp=2.8, steps=800, ext=0.35, seed=1):
    cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
    sc=generate_scenario("motordrome",cfg,seed=seed); md=sc.motordromes[0]
    wall_r,mu=md[3],md[6]
    env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=seed)
    E=env.max_extend; R=FOOT_BASE+ext*E; r0=wall_r-R
    env.data.qpos[0]=r0; env.data.qpos[1]=0; env.data.qpos[2]=z0
    env.data.qpos[3:7]=[1,0,0,0]; env.data.qvel[:]=0
    env.data.qvel[1]=v0; env.data.qvel[5]=-v0/R
    mujoco.mj_forward(env.model,env.data)
    zs=[];vs=[];nc=[]
    for i in range(steps):
        pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
        n,d=surface_frame(env.model,env.data,pos)
        tg,info=wall_of_death(env.data.qpos[3:7].copy(),env.dirs_body,E,pos,vel,
                              normal=n,surface_dist=d,wall_radius=wall_r,wall_friction=mu,
                              ccw=True,climb_rate=climb,press=press,speed=sp,touch_margin=tm)
        env.step(tg); zs.append(info["z"]); vs.append(info["speed"]); nc.append(env.data.ncon)
        if info["z"]<0.45: break
    env.close()
    return dict(n=len(zs),z_end=zs[-1],drop=z0-zs[-1],v_end=vs[-1],
                v_mean=float(np.mean(vs)),ncon=float(np.mean(nc)))

print("Hold test with rods capped at the wall. z0=1.60, level, 8 s.")
print("drop < 0.2 m = holding.\n")
print(f"{'v0':>4} {'press':>6} {'margin':>7} {'gain(v)':>8} {'steps':>6} {'z_end':>6} {'drop':>6} {'v_end':>6} {'ncon':>5}")
for v0 in (3.4, 4.2):
    for pr in (0.30, 0.60, 0.85, 1.00):
        for tm in (0.008, 0.025):
            r=trial(v0, press=pr, tm=tm, sp=2.8)
            print(f"{v0:4.1f} {pr:6.2f} {tm*1000:7.0f} {2.8:8.1f} {r['n']:6d} {r['z_end']:6.2f} "
                  f"{r['drop']:6.2f} {r['v_end']:6.2f} {r['ncon']:5.1f}")
