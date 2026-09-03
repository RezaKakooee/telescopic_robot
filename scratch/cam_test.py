import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco, imageio.v2 as iio
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import Bowl, advance_radius, surface_frame, wall_of_death
S="/tmp/claude-1000/-home-azureuser-telescopic-robot/4f64cce2-bbc9-4914-8de7-37c3bd38fa98/scratchpad"
cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
sc=generate_scenario("motordrome",cfg,seed=42); md=sc.motordromes[0]; bowl=Bowl.from_motordrome(md)
env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**7); env.reset(seed=42)
env.data.qpos[0]=0.55; env.data.qpos[1]=0; env.data.qpos[2]=0.22; env.data.qvel[:]=0
mujoco.mj_forward(env.model,env.data)
r_cmd=0.55
for i in range(3500):                     # settle into the high orbit
    pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
    n,_=surface_frame(env.model,env.data,pos)
    tg,info=wall_of_death(env.data.qpos[3:7].copy(),env.dirs_body,env.max_extend,pos,vel,
                          r_cmd=r_cmd,bowl=bowl,normal=n,wall_radius=float(md[3]),ccw=True,steer_gain=0.30)
    r_cmd=advance_radius(r_cmd,info["r"],info["speed"],bowl); env.step(tg)
print(f"settled r={info['r']:.2f} z={info['z']:.2f} v={info['speed']:.2f}")
env.render(camera_name="fixed_angle_close_3d")
th=float(np.arctan2(env.data.qpos[1],env.data.qpos[0]))
opts=[("track_d9_e30",dict(track=1,d=9.0,el=-30,az=th)),
      ("track_d7_e25",dict(track=1,d=7.0,el=-25,az=th)),
      ("track_d12_e40",dict(track=1,d=12.0,el=-40,az=th)),
      ("wide_d11_e28",dict(track=0,d=11.0,el=-28,az=th))]
for name,o in opts:
    cam=mujoco.MjvCamera()
    if o["track"]:
        cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid=env.core_body_id
    else:
        cam.type=mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:]=[0,0,0.8]
    cam.distance=o["d"]; cam.elevation=o["el"]; cam.azimuth=float(np.degrees(o["az"]))+50.0
    env.renderer.update_scene(env.data,camera=cam)
    iio.imwrite(f"{S}/cam_{name}.png", env.renderer.render()); print("wrote",name)
env.close()
