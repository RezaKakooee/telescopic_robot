import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat
from skills.wall_of_death import wall_of_death, surface_frame, reach_caps, FOOT_BASE
cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
sc=generate_scenario("motordrome",cfg,seed=1); md=sc.motordromes[0]
wall_r,mu=md[3],md[6]
env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
gname={i:(mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,i) or "") for i in range(env.model.ngeom)}
E=env.max_extend; R=FOOT_BASE+0.35*E; r0=wall_r-R; v0=4.2
env.data.qpos[0]=r0; env.data.qpos[1]=0; env.data.qpos[2]=2.0
env.data.qpos[3:7]=[1,0,0,0]; env.data.qvel[:]=0
env.data.qvel[1]=v0; env.data.qvel[5]=-v0/R
mujoco.mj_forward(env.model,env.data)
print(f"placed r={r0:.3f} z=2.0 vt={v0}  wall={wall_r}")
print(f"{'i':>3} {'r':>6} {'z':>5} {'vr':>6} {'vz':>6} {'vt':>5} {'nw':>3} {'Fn':>6} {'Fz':>6} "
      f"{'maxrad':>7} {'radcap':>7} {'radtgt':>7} {'radq':>6}")
f6=np.zeros(6)
for i in range(60):
    pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
    n,d=surface_frame(env.model,env.data,pos)
    tg,info=wall_of_death(env.data.qpos[3:7].copy(),env.dirs_body,E,pos,vel,normal=n,
                          surface_dist=d,wall_radius=wall_r,wall_friction=mu,ccw=True,
                          climb_rate=0.0,press=0.0,speed=2.8,touch_margin=0.008)
    th=np.arctan2(pos[1],pos[0]); rh=np.array([np.cos(th),np.sin(th),0.]); th_h=np.array([-np.sin(th),np.cos(th),0.])
    dw=env.dirs_body@quat_to_rotmat(env.data.qpos[3:7]).T
    caps=reach_caps(pos,dw,E,wall_radius=wall_r,plane_normal=n if d is not None else None,plane_dist=d,margin=0.008)
    un=dw@rh; k=int(np.argmax(un))                      # most radial rod
    feet=pos[None,:]+dw*(FOOT_BASE+np.asarray(env.data.qpos[7:7+60]))[:,None]
    maxrad=float(np.hypot(feet[:,0],feet[:,1]).max())
    nw=0;Fn=0.;Fz=0.
    for c in range(env.data.ncon):
        con=env.data.contact[c]
        if "md_wall" not in gname[con.geom1]+gname[con.geom2]: continue
        mujoco.mj_contactForce(env.model,env.data,c,f6)
        fw=np.array(con.frame).reshape(3,3).T@f6[:3]
        Fn+=abs(float(np.dot(fw,rh))); Fz+=float(fw[2]); nw+=1
    if i<30 or i%5==0:
        print(f"{i:3d} {info['r']:6.3f} {pos[2]:5.2f} {np.dot(vel,rh):6.2f} {vel[2]:6.2f} "
              f"{np.dot(vel,th_h):5.2f} {nw:3d} {Fn:6.1f} {Fz:6.1f} {maxrad:7.3f} "
              f"{caps[k]:7.3f} {tg[k]:7.3f} {env.data.qpos[7+k]:6.3f}")
    env.step(tg)
env.close()
