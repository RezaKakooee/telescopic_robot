import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco, imageio.v2 as iio
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import Bowl, advance_radius, surface_frame, wall_of_death
import scripts.skills.run_motordrome_wall_of_death as R
S="/tmp/claude-1000/-home-azureuser-telescopic-robot/4f64cce2-bbc9-4914-8de7-37c3bd38fa98/scratchpad"
cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
sc=generate_scenario("motordrome",cfg,seed=42); md=sc.motordromes[0]; bowl=Bowl.from_motordrome(md)
rim_r, wall_top = float(md[3]), float(md[5])
env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**7); env.reset(seed=42)
env.data.qpos[0]=0.55; env.data.qpos[1]=0; env.data.qpos[2]=0.22; env.data.qvel[:]=0
mujoco.mj_forward(env.model,env.data)
env.render(camera_name="fixed_angle_close_3d")
r_cmd=0.55; shots={}
for i in range(4200):
    pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
    n,_=surface_frame(env.model,env.data,pos)
    tg,info=wall_of_death(env.data.qpos[3:7].copy(),env.dirs_body,env.max_extend,pos,vel,
                          r_cmd=r_cmd,bowl=bowl,normal=n,wall_radius=rim_r,ccw=True,steer_gain=0.30)
    r_cmd=advance_radius(r_cmd,info["r"],info["speed"],bowl); env.step(tg)
    th=float(np.arctan2(pos[1],pos[0]))
    # early shot, then four points around one high lap
    if i==200: shots["early"]=(info,th,pos.copy())
    if i>=3600 and len(shots)<5:
        deg=(np.degrees(th))%360
        for k,tgt in (("far",150.),("left",240.),("near",330.),("right",60.)):
            if k not in shots and abs(((deg-tgt+180)%360)-180)<6:
                shots[k]=(info,th,pos.copy())
env.close()
env2=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10); env2.reset(seed=42)
print("captured:", list(shots))
