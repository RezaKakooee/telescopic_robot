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
gname=None
def run(v0, tm, sp, steps=300, label=""):
    global gname
    env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
    if gname is None:
        gname={i:(mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,i) or "") for i in range(env.model.ngeom)}
    E=env.max_extend; R=FOOT_BASE+0.35*E; r0=wall_r-R
    env.data.qpos[0]=r0; env.data.qpos[1]=0; env.data.qpos[2]=2.0
    env.data.qpos[3:7]=[1,0,0,0]; env.data.qvel[:]=0
    env.data.qvel[1]=v0; env.data.qvel[5]=-v0/R
    mujoco.mj_forward(env.model,env.data)
    f6=np.zeros(6); touch=0; Fz_sum=0.0; Fn_sum=0.0; rs=[]; zs=[]
    for i in range(steps):
        pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
        n,d=surface_frame(env.model,env.data,pos)
        tg,info=wall_of_death(env.data.qpos[3:7].copy(),env.dirs_body,E,pos,vel,normal=n,
                              surface_dist=d,wall_radius=wall_r,wall_friction=mu,ccw=True,
                              climb_rate=0.0,press=0.0,speed=sp,touch_margin=tm)
        env.step(tg)
        th=np.arctan2(pos[1],pos[0]); r_hat=np.array([np.cos(th),np.sin(th),0.])
        nw=0; Fz=0.0; Fn=0.0
        for c in range(env.data.ncon):
            con=env.data.contact[c]
            if "md_wall" not in gname[con.geom1]+gname[con.geom2]: continue
            mujoco.mj_contactForce(env.model,env.data,c,f6)
            fw=np.array(con.frame).reshape(3,3).T@f6[:3]
            Fz+=float(fw[2]); Fn+=abs(float(np.dot(fw,r_hat))); nw+=1
        if nw: touch+=1
        Fz_sum+=Fz; Fn_sum+=Fn; rs.append(info["r"]); zs.append(pos[2])
    env.close()
    m=1.52
    print(f"{label:22s} duty={touch/steps:5.2f} meanFz={Fz_sum/steps:6.2f}N (need {m*9.81:.1f}) "
          f"meanFn={Fn_sum/steps:6.2f}N  r={np.mean(rs):.3f}  dz={zs[-1]-2.0:+.2f}")

print(f"wall r={wall_r} mu={mu}. Placed at z=2.0, 3 s.\n")
for v0 in (3.4, 4.2, 5.0, 6.0):
    run(v0, 0.008, 2.8, label=f"v0={v0}")
print()
for tm in (0.0, 0.015, 0.030, 0.060):
    run(4.2, tm, 2.8, label=f"v0=4.2 margin={tm*1000:.0f}mm")
print()
for sp in (0.4, 1.2, 2.0, 2.8):
    run(4.2, 0.008, sp, label=f"v0=4.2 gain@{sp}m/s")
