"""Try candidate bowls. Drive from the floor, measure the sustained height."""
import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import wall_of_death, surface_frame

def run(floor_r, wall_r, b0, b1, steps=3000, seed=1, climb=0.0, stroke=0.30, quiet=True):
    cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
    cfg.robot.max_extend=stroke
    cfg.scenario.floor_radius=floor_r; cfg.scenario.wall_radius=wall_r
    cfg.scenario.bowl_start_bank=b0; cfg.scenario.bowl_max_bank=b1; cfg.scenario.bowl_segments=14
    sc=generate_scenario("motordrome",cfg,seed=seed); md=sc.motordromes[0]
    rim_z=md[4]; mu=md[6]
    env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=seed)
    env.data.qpos[0]=max(0.12,floor_r*0.5); env.data.qpos[1]=0.0
    env.data.qpos[2]=0.20; env.data.qvel[:]=0.0
    mujoco.mj_forward(env.model,env.data)
    zs=[];vs=[];laps=0.0;pth=0.0
    for i in range(steps):
        pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
        n,d=surface_frame(env.model,env.data,pos)
        tg,info=wall_of_death(env.data.qpos[3:7].copy(),env.dirs_body,env.max_extend,pos,vel,
                              normal=n,surface_dist=None,wall_radius=wall_r,wall_friction=mu,
                              ccw=True,climb_rate=climb,press=0.0,force_pitch=0.0)
        env.step(tg)
        th=np.arctan2(pos[1],pos[0]); dth=(th-pth+np.pi)%(2*np.pi)-np.pi
        if i: laps+=dth/(2*np.pi)
        pth=th; zs.append(info["z"]); vs.append(info["speed"])
    env.close()
    tail=slice(-1000,None)
    return dict(rim_z=rim_z, max_z=max(zs), z_mean=float(np.mean(zs[tail])),
                z_min=float(np.min(zs[tail])), v_mean=float(np.mean(vs[tail])),
                laps=abs(laps))

print("Full bowls: flat at the centre, steepening to the rim. 30 s each.")
print("z_mean over the last 10 s is the sustained ride height.\n")
print(f"{'floor':>6} {'rim':>5} {'bank':>8} {'depth':>6} | {'max_z':>6} {'z_mean':>7} {'z_min':>6} {'v_mean':>7} {'laps':>5}")
for fr,wr,b0,b1 in [(0.25,1.60,0,60),(0.25,1.90,0,62),(0.25,2.20,0,60),
                    (0.25,1.90,10,66),(0.25,2.20,12,64),(0.25,2.60,8,58)]:
    r=run(fr,wr,b0,b1)
    print(f"{fr:6.2f} {wr:5.2f} {b0:3d}-{b1:<4d} {r['rim_z']:6.2f} | {r['max_z']:6.2f} "
          f"{r['z_mean']:7.2f} {r['z_min']:6.2f} {r['v_mean']:7.2f} {r['laps']:5.1f}")
