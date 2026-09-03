import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
cfg=load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg,False)
print(f"{'stroke':>7} {'foot_mu':>8} {'v_top':>7} {'v^2/g':>7}  (v^2/g caps r*tan(bank) at the ride point)")
for stroke in (0.26, 0.30, 0.34):
    for fmu in (1.6, 2.2):
        cfg.robot.max_extend=stroke; cfg.sim2real.rubber_friction_sliding=fmu
        sc=generate_scenario("path",cfg,seed=1)
        env=MujocoRadialSphereEnv(cfg,scenario=sc,randomize=False,max_steps=10**6); env.reset(seed=1)
        d=np.array([1.0,0.0]); vs=[]
        for i in range(700):
            tg=move(env.data.qpos[3:7].copy(),env.dirs_body,env.max_extend,d,back_gain=4.0)
            env.step(tg); vs.append(float(np.linalg.norm(env.data.qvel[:2])))
        env.close()
        v=float(np.mean(vs[-200:]))
        print(f"{stroke:7.2f} {fmu:8.1f} {v:7.2f} {v*v/9.81:7.2f}")
