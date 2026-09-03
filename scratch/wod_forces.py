import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import wall_of_death, surface_normal
FOOT=0.173
cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
sc=generate_scenario("motordrome",cfg,seed=1); md=sc.motordromes[0]
wall_r,mu=md[3],md[6]
env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
E=env.max_extend
m_total = env.model.body_subtreemass[env.core_body_id]
print(f"robot mass = {m_total:.3f} kg; weight = {m_total*9.81:.2f} N")
# name the wall geoms so we can tell wall contacts from apron
gname={i:mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(env.model.ngeom)}
R_roll=FOOT+0.35*E; r0=wall_r-R_roll
env.data.qpos[0]=r0; env.data.qpos[1]=0; env.data.qpos[2]=1.60
env.data.qpos[3:7]=[1,0,0,0]; env.data.qvel[:]=0.0
env.data.qvel[1]=4.2; env.data.qvel[5]=-4.2/R_roll
mujoco.mj_forward(env.model,env.data)
print(f"placed r={r0:.3f}\n")
print(f"{'i':>4} {'z':>6} {'r':>5} {'vz':>6} {'vt':>5} {'ncon':>4} {'Fn_wall':>8} {'need':>6} {'Ffric':>7} {'wallcon':>7} {'maxrad':>7}")
f6=np.zeros(6)
for i in range(220):
    pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
    n=surface_normal(env.model,env.data,pos)
    tg,info=wall_of_death(env.data.qpos[3:7].copy(),env.dirs_body,E,pos,vel,normal=n,
                          wall_radius=wall_r,wall_friction=mu,ccw=True,climb_rate=0.0,press=0.0)
    # measure contact forces on wall planks
    Fn=0.0; Ft=0.0; nw=0
    th=np.arctan2(pos[1],pos[0]); r_hat=np.array([np.cos(th),np.sin(th),0.])
    for c in range(env.data.ncon):
        con=env.data.contact[c]
        nm1,nm2=gname.get(con.geom1,""),gname.get(con.geom2,"")
        if not (("md_wall" in (nm1 or "")) or ("md_wall" in (nm2 or ""))): continue
        mujoco.mj_contactForce(env.model,env.data,c,f6)
        frame=np.array(con.frame).reshape(3,3)
        fw=frame.T@f6[:3]
        Fn+=abs(float(np.dot(fw,r_hat))); Ft+=float(np.linalg.norm(fw-np.dot(fw,r_hat)*r_hat)); nw+=1
    # farthest rod reach in the radial direction
    Rm=np.array(env.data.qpos[3:7]); 
    from radial_sphere.geometry import quat_to_rotmat
    dw=env.dirs_body@quat_to_rotmat(Rm).T
    L=FOOT+np.asarray(tg)
    feet=pos[None,:]+dw*L[:,None]
    maxrad=float(np.hypot(feet[:,0],feet[:,1]).max())
    if i%15==0:
        t_hat=np.array([-np.sin(th),np.cos(th),0.])
        print(f"{i:4d} {pos[2]:6.2f} {info['r']:5.2f} {vel[2]:6.2f} {np.dot(vel,t_hat):5.2f} "
              f"{env.data.ncon:4d} {Fn:8.2f} {m_total*info['v_t']**2/max(info['r'],.01):6.2f} "
              f"{Ft:7.2f} {nw:7d} {maxrad:7.3f}")
    env.step(tg)
env.close()
