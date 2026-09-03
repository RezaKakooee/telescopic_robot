import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill

cfg = load_config('configs/rl/config.yaml')
cfg.robot.rod_mechanism = 'multi_stage'
cfg.camera.enabled = False
scenario = generate_scenario('goal', cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=200)
env.reset(seed=42)

peak_z = env.data.qpos[2]
peak_vz = env.data.qvel[2]

for step in range(80):
    quat = env.data.qpos[3:7].copy()
    if step < 20: phase = 'crouch'
    elif step < 32: phase = 'takeoff'
    elif env.data.qpos[2] > 0.28: phase = 'airborne'
    else: phase = 'landing'
    
    targets = execute_skill('jump_up', quat, env.dirs_body, env.max_extend, phase=phase)
    env.step(targets)
    
    z = env.data.qpos[2]
    vz = env.data.qvel[2]
    if z > peak_z: peak_z = z
    if vz > peak_vz: peak_vz = vz

print(f'NATIVE MULTI_STAGE JUMP: peak_z = {peak_z:.3f} m, peak_vz = {peak_vz:.2f} m/s')
env.close()
