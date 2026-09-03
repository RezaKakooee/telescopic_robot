"""Diagnose why the jump became weak and test parameter sensitivity."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill

def test_jump(mechanism="multi_stage", sim2real_enabled=False, damping=0.0, armature=0.001, max_f=120.0):
    cfg = load_config("configs/rl/config.yaml")
    cfg.robot.rod_mechanism = mechanism
    cfg.camera.enabled = False
    cfg.sim2real.enabled = sim2real_enabled
    
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=200)
    env.reset(seed=42)
    
    # Overwrite joint parameters to test sensitivity
    for k in range(env.n_bars):
        # slide_k
        jid = env.model.joint(f"slide_{k}").id
        env.model.dof_damping[jid] = damping
        env.model.dof_armature[jid] = armature
        # actuator
        aid = env.model.actuator(f"slide_{k}").id
        env.model.actuator_forcerange[aid] = [-max_f, max_f]
        
        # stage1_k if exists
        try:
            j1id = env.model.joint(f"slide1_{k}").id
            env.model.dof_damping[j1id] = damping * 0.5
            env.model.dof_armature[j1id] = armature * 0.5
        except Exception:
            pass

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

print("=== JUMP PEAK ANALYSIS ===")
# 1. Baseline single_stage (original jump)
z_orig, vz_orig = test_jump("single_stage", sim2real_enabled=False, damping=0.0, armature=0.001, max_f=120.0)
print(f"Original single_stage (ideal):   peak_z = {z_orig:.3f} m, peak_vz = {vz_orig:.2f} m/s")

# 2. multi_stage with high damping / low force (what we currently have)
z_curr, vz_curr = test_jump("multi_stage", sim2real_enabled=False, damping=3.5, armature=0.018, max_f=55.0)
print(f"Current multi_stage (heavy/damp): peak_z = {z_curr:.3f} m, peak_vz = {vz_curr:.2f} m/s")

# 3. multi_stage with tuned high-power actuators (real brushless pulse)
z_tuned, vz_tuned = test_jump("multi_stage", sim2real_enabled=False, damping=0.5, armature=0.002, max_f=120.0)
print(f"Tuned multi_stage (high pulse):   peak_z = {z_tuned:.3f} m, peak_vz = {vz_tuned:.2f} m/s")

# 4. multi_stage with high force 150N (peak discharge LiPo)
z_burst, vz_burst = test_jump("multi_stage", sim2real_enabled=False, damping=0.2, armature=0.001, max_f=150.0)
print(f"Burst multi_stage (150N burst):   peak_z = {z_burst:.3f} m, peak_vz = {vz_burst:.2f} m/s")
