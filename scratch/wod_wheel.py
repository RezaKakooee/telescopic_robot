"""Is a uniformly-extended ball a wheel? A 60-rod shell at fixed stroke is a
60-sided polygon: rolling on it ripples by R*(1-cos(13 deg)) = 7 mm. That
should roll on the wall instead of skidding on it."""
import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
FOOT=0.173
cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
sc=generate_scenario("motordrome",cfg,seed=1); md=sc.motordromes[0]
wall_r,mu=md[3],md[6]

def trial(v0, ext, z0=2.0, steps=800):
    env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
    E=env.max_extend; R=FOOT+ext; r0=wall_r-R
    env.data.qpos[0]=r0; env.data.qpos[1]=0; env.data.qpos[2]=z0
    env.data.qpos[3:7]=[1,0,0,0]; env.data.qvel[:]=0
    env.data.qpos[7:7+60]=ext                       # start already extended
    env.data.qvel[1]=v0; env.data.qvel[5]=-v0/R
    mujoco.mj_forward(env.model,env.data)
    tg=np.full(60, ext, dtype=np.float32)
    zs=[];vs=[];rs=[];nc=[]
    for i in range(steps):
        env.step(tg)
        p=env.data.qpos[:3]; v=env.data.qvel[:3]
        zs.append(float(p[2])); vs.append(float(np.linalg.norm(v[:2])))
        rs.append(float(np.hypot(p[0],p[1]))); nc.append(env.data.ncon)
        if p[2]<0.5: break
    env.close()
    return dict(n=len(zs),z_end=zs[-1],drop=z0-zs[-1],v_end=vs[-1],
                r_end=rs[-1],ncon=float(np.mean(nc)))

print("Uniform-extension wheel test. Placed on the wall, level, up to 8 s.")
print("Rolls if speed survives; holds if drop is small.\n")
print(f"{'v0':>4} {'stroke':>7} {'Rwheel':>7} {'need_v':>7} {'steps':>6} {'z_end':>6} {'drop':>6} {'v_end':>6} {'r_end':>6} {'ncon':>5}")
for ext in (0.26, 0.18, 0.10):
    R=FOOT+ext
    need=np.sqrt(9.81*(wall_r-R)/mu)
    for v0 in (3.0, 4.0, 5.0):
        r=trial(v0,ext)
        print(f"{v0:4.1f} {ext:7.2f} {R:7.3f} {need:7.2f} {r['n']:6d} {r['z_end']:6.2f} "
              f"{r['drop']:6.2f} {r['v_end']:6.2f} {r['r_end']:6.2f} {r['ncon']:5.1f}")
