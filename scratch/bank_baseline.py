import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat
from skills.wall_of_death import surface_frame, reach_caps, FOOT_BASE
from skills.locomotion import surface_drive
WALL_R=1.80; EXT=0.09; ROLL=FOOT_BASE+EXT

def arena(bank_deg, top=3.0):
    cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
    b=np.radians(bank_deg)
    apron_h=min(top-0.3,(WALL_R-0.35)*np.tan(b)); floor_r=WALL_R-apron_h/np.tan(b)
    cfg.scenario.floor_radius=float(floor_r); cfg.scenario.wall_radius=WALL_R
    cfg.scenario.apron_height=float(apron_h); cfg.scenario.cylinder_height=float(top)
    return cfg, generate_scenario("motordrome",cfg,seed=1), floor_r, apron_h

def hold(bank_deg, mode, steps=800, z_frac=0.55):
    cfg,sc,floor_r,apron_h=arena(bank_deg); b=np.radians(bank_deg)
    env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
    E=env.max_extend
    z_s=apron_h*z_frac; r_s=floor_r+z_s/np.tan(b)
    n0=np.array([np.sin(b),0.,-np.cos(b)])
    cx=r_s-ROLL*np.sin(b); cz=z_s+ROLL*np.cos(b)
    v0=np.sqrt(9.81*cx*np.tan(b))
    env.data.qpos[0]=cx; env.data.qpos[1]=0; env.data.qpos[2]=cz
    env.data.qpos[3:7]=[1,0,0,0]; env.data.qvel[:]=0; env.data.qvel[1]=v0
    env.data.qpos[7:7+60]=EXT
    env.data.qvel[3:6]=np.cross(n0,np.array([0.,1.,0.]))*(v0/ROLL)
    mujoco.mj_forward(env.model,env.data)
    fixed=np.full(60,EXT,dtype=np.float32); zs=[];vs=[]
    for i in range(steps):
        pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
        if mode=="wheel":
            tg=fixed
        else:
            n,d=surface_frame(env.model,env.data,pos)
            if n is None: n=n0; d=None
            th=np.arctan2(pos[1],pos[0]); t=np.array([-np.sin(th),np.cos(th),0.])
            cap=None
            if mode=="cyl":
                dw=env.dirs_body@quat_to_rotmat(env.data.qpos[3:7]).T
                cap=reach_caps(pos,dw,E,wall_radius=WALL_R,margin=0.008)
            tg=surface_drive(env.data.qpos[3:7].copy(),env.dirs_body,E,n,t,speed=2.8,reach_cap=cap)
        env.step(tg); zs.append(float(pos[2])); vs.append(float(np.linalg.norm(vel[:2])))
        if pos[2]<0.05: break
    env.close()
    return cz, zs[-1]-cz, vs[-1], v0, len(zs)

print("Baselines on a cone. wheel = rods locked at 0.09, no control at all.")
print("drive = the gait, no cap. cyl = the gait with the cylinder cap only.\n")
print(f"{'bank':>5} {'v0':>5} | " + " | ".join(f"{m:>18}" for m in ("wheel","drive","cyl")))
print(f"{'':>5} {'':>5} | " + " | ".join(f"{'dz':>7}{'v_end':>7}{'n':>4}" for _ in range(3)))
for bk in (30,45,60,75,88):
    row=[]
    for m in ("wheel","drive","cyl"):
        z0,dz,ve,v0,n=hold(bk,m); row.append(f"{dz:+7.2f}{ve:7.2f}{n:4d}")
    print(f"{bk:5d} {v0:5.2f} | " + " | ".join(row))
