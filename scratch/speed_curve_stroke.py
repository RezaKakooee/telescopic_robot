import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move, SPEED_CURVE
cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
GAINS=[g for g,_ in SPEED_CURVE]; V16=[v for _,v in SPEED_CURVE]
print("Does the measured speed curve just scale with stroke?\n")
print(f"{'gain':>5} {'v@0.16 (ref)':>13} {'v@0.30':>8} {'ratio':>7}")
res={}
for gain in GAINS:
    row=[]
    for stroke in (0.30,):
        cfg.robot.max_extend=stroke
        sc=generate_scenario("path",cfg,seed=1)
        env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
        d=np.array([1.,0.]); vs=[]
        for i in range(650):
            env.step(move(env.data.qpos[3:7].copy(),env.dirs_body,env.max_extend,d,back_gain=gain))
            vs.append(float(np.linalg.norm(env.data.qvel[:2])))
        env.close(); row.append(float(np.mean(vs[-200:])))
    res[gain]=row[0]
for g,v16 in SPEED_CURVE:
    v30=res[g]
    print(f"{g:5.2f} {v16:13.2f} {v30:8.2f} {v30/max(v16,1e-9):7.2f}")
print(f"\nstroke ratio 0.30/0.16 = {0.30/0.16:.2f}")
