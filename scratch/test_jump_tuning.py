"""Test tuning parameters to give multi_stage explosive, powerful jump."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill

def evaluate_jump(damping=0.4, frictionloss=0.1, armature=0.002, max_f=120.0, kp=1100.0, kv=22.0):
    cfg = load_config("configs/rl/config.yaml")
    cfg.robot.rod_mechanism = "multi_stage"
    cfg.camera.enabled = False
    
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=200)
    env.reset(seed=42)
    
    for k in range(env.n_bars):
        # slide_k
        jid = env.model.joint(f"slide_{k}").id
        env.model.dof_damping[jid] = damping
        env.model.dof_armature[jid] = armature
        env.model.dof_frictionloss[jid] = frictionloss
        
        # slide1_k
        try:
            j1id = env.model.joint(f"slide1_{k}").id
            env.model.dof_damping[j1id] = damping * 0.5
            env.model.dof_armature[j1id] = armature * 0.5
            env.model.dof_frictionloss[j1id] = frictionloss * 0.5
        except Exception:
            pass
            
        # actuator
        aid = env.model.actuator(f"slide_{k}").id
        env.model.actuator_forcerange[aid] = [-max_f, max_f]
        env.model.actuator_gainprm[aid, 0] = kp
        env.model.actuator_biasprm[aid, 1] = -kp
        env.model.actuator_biasprm[aid, 2] = -kv

    peak_z = env.data.qpos[2]
    peak_vz = env.data.qvel[2]
    
    for step in range(80):
        quat = env.data.qpos[3:7].copy()
        if step < 20: phase = "crouch"
        elif step < 32: phase = "takeoff"
        elif env.data.qpos[2] > 0.28: phase = "airborne"
        else: phase = "landing"
        
        targets = execute_skill("jump_up", quat, env.dirs_body, env.max_extend, phase=phase)
        env.step(targets)
        
        z = env.data.qpos[2]
        vz = env.data.qvel[2]
        if z > peak_z: peak_z = z
        if vz > peak_vz: peak_vz = vz
        
    env.close()
    return peak_z, peak_vz

for max_f in [80.0, 120.0, 150.0]:
    for kp in [900.0, 1200.0, 1500.0]:
        pz, pvz = evaluate_jump(damping=0.3, frictionloss=0.1, armature=0.002, max_f=max_f, kp=kp, kv=22.0)
        print(f"max_f={max_f:5.1f}N, kp={kp:5.0f} -> peak_z = {pz:.3f} m, peak_vz = {pvz:.2f} m/s")
