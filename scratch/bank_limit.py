"""How steep a bank can this robot hold a level orbit on?

For each bank angle the ball is placed on the surface at the speed where the
bank alone supplies the centripetal force, v = sqrt(g r tan b). Then it drives
tangentially for 8 s. If it keeps its height, that bank is ridable.
"""
import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import surface_frame, reach_caps, FOOT_BASE
from skills.locomotion import surface_drive

WALL_R = 1.80
ROLL   = FOOT_BASE + 0.35*0.26      # ~0.264 m riding envelope

def arena(bank_deg, top=3.0):
    cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
    b=np.radians(bank_deg)
    apron_h = min(top-0.3, (WALL_R-0.35)*np.tan(b))     # cone reaches as high as it can
    floor_r = WALL_R - apron_h/np.tan(b)
    cfg.scenario.floor_radius=float(floor_r); cfg.scenario.wall_radius=WALL_R
    cfg.scenario.apron_height=float(apron_h); cfg.scenario.cylinder_height=float(top)
    sc=generate_scenario("motordrome",cfg,seed=1)
    return cfg, sc, floor_r, apron_h

def hold(bank_deg, z_frac=0.55, steps=800, vscale=1.0):
    cfg,sc,floor_r,apron_h = arena(bank_deg)
    mu = sc.motordromes[0][6]
    b=np.radians(bank_deg)
    env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
    E=env.max_extend
    z_s = apron_h*z_frac
    r_s = floor_r + z_s/np.tan(b)
    n_hat = np.array([np.sin(b),0.0,-np.cos(b)])            # ball -> surface
    cx = r_s - ROLL*np.sin(b); cz = z_s + ROLL*np.cos(b)
    v0 = vscale*np.sqrt(9.81*cx*np.tan(b))
    env.data.qpos[0]=cx; env.data.qpos[1]=0; env.data.qpos[2]=cz
    env.data.qpos[3:7]=[1,0,0,0]; env.data.qvel[:]=0
    env.data.qvel[1]=v0
    # rolling: spin axis is the surface normal crossed with travel
    ax=np.cross(n_hat,np.array([0.,1.,0.]))
    env.data.qvel[3:6]=ax*(v0/ROLL)
    mujoco.mj_forward(env.model,env.data)
    zs=[];vs=[]
    for i in range(steps):
        pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
        n,d=surface_frame(env.model,env.data,pos)
        if n is None: n=n_hat; d=None
        th=np.arctan2(pos[1],pos[0]); t_hat=np.array([-np.sin(th),np.cos(th),0.])
        from radial_sphere.geometry import quat_to_rotmat
        dw=env.dirs_body@quat_to_rotmat(env.data.qpos[3:7]).T
        caps=reach_caps(pos,dw,E,wall_radius=WALL_R,
                        plane_normal=n if d is not None else None,plane_dist=d,margin=0.008)
        tg=surface_drive(env.data.qpos[3:7].copy(),env.dirs_body,E,n,t_hat,speed=2.8,reach_cap=caps)
        env.step(tg); zs.append(float(pos[2])); vs.append(float(np.linalg.norm(vel[:2])))
        if pos[2] < 0.05: break
    env.close()
    return dict(z0=cz, v0=v0, n=len(zs), z_end=zs[-1], dz=zs[-1]-cz,
                v_end=vs[-1], apron_h=apron_h, floor_r=floor_r, mu=mu)

print("Steepest ridable bank. Ball placed at the bank's own equilibrium speed.")
print("dz near 0 with speed kept = it rides.\n")
print(f"{'bank':>5} {'floor_r':>8} {'apron_h':>8} {'v0':>5} {'steps':>6} {'z0':>5} {'z_end':>6} {'dz':>7} {'v_end':>6}")
for bk in (30, 40, 50, 60, 70, 80, 88):
    r=hold(bk)
    print(f"{bk:5d} {r['floor_r']:8.2f} {r['apron_h']:8.2f} {r['v0']:5.2f} {r['n']:6d} "
          f"{r['z0']:5.2f} {r['z_end']:6.2f} {r['dz']:+7.2f} {r['v_end']:6.2f}")
