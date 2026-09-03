import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import Bowl, advance_radius, wall_of_death, surface_frame

def run(steps=4000, seed=1, verbose=False, open_rate=0.10, steer_gain=0.55,
        max_steer=0.35, lat=4.5, **over):
    cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
    for k,v in over.items(): setattr(cfg.scenario,k,v)
    sc=generate_scenario("motordrome",cfg,seed=seed); md=sc.motordromes[0]
    bowl=Bowl.from_motordrome(md, lateral_limit=lat)
    env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=seed)
    env.data.qpos[0]=0.55; env.data.qpos[1]=0.0; env.data.qpos[2]=0.22; env.data.qvel[:]=0
    mujoco.mj_forward(env.model,env.data)
    r_cmd=0.55; zs=[];vs=[];rc=[];rr=[];laps=0.0;pth=0.0
    for i in range(steps):
        pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
        n,_=surface_frame(env.model,env.data,pos)
        tg,info=wall_of_death(env.data.qpos[3:7].copy(),env.dirs_body,env.max_extend,pos,vel,
                              r_cmd=r_cmd,bowl=bowl,normal=n,wall_radius=md[3],ccw=True,
                              steer_gain=steer_gain,max_steer=max_steer)
        r_cmd=advance_radius(r_cmd,info["r"],info["speed"],bowl,open_rate=open_rate)
        env.step(tg)
        th=np.arctan2(pos[1],pos[0]); d=(th-pth+np.pi)%(2*np.pi)-np.pi
        if i: laps+=d/(2*np.pi)
        pth=th; zs.append(info["z"]); vs.append(info["speed"]); rc.append(r_cmd); rr.append(info["r"])
        if verbose and i%400==0:
            print(f"  {i:5d} r={info['r']:5.2f} r_cmd={r_cmd:5.2f} z={info['z']:5.2f} "
                  f"v={info['speed']:5.2f} hold={info['hold_speed']:5.2f} bank={info['bank_deg']:4.1f} "
                  f"steer={info['steer']:+5.2f}")
    env.close()
    t=slice(-800,None)
    return dict(depth=bowl.rim_z, max_z=max(zs), z_mean=float(np.mean(zs[t])),
                z_min=float(np.min(zs[t])), v_mean=float(np.mean(vs[t])),
                r_mean=float(np.mean(rr[t])), rcmd_end=rc[-1], laps=abs(laps))

print("Spiral out. 40 s. Last 8 s is the sustained ride.\n")
print(f"{'ramp':>5} {'open':>5} {'steer':>6} {'depth':>6} | {'max_z':>6} {'z_mean':>7} {'z_min':>6} {'v_mean':>7} {'r_mean':>7} {'laps':>5}")
for ramp in (6.0, 3.5):
    for op in (0.10, 0.04):
        r=run(open_rate=op, bowl_ramp_rate=ramp)
        print(f"{ramp:5.1f} {op:5.2f} {0.55:6.2f} {r['depth']:6.2f} | {r['max_z']:6.2f} {r['z_mean']:7.2f} "
              f"{r['z_min']:6.2f} {r['v_mean']:7.2f} {r['r_mean']:7.2f} {r['laps']:5.1f}")
