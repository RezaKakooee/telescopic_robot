"""Render test of jumping skills with restored explosive power."""
import os
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill
from skills.overlay import annotate

out_dir = Path("storage_local/jump_test")
out_dir.mkdir(parents=True, exist_ok=True)

def render_jump(name, skill_fn, n_steps=80):
    cfg = load_config("configs/rl/config.yaml")
    cfg.robot.rod_mechanism = "multi_stage"
    cfg.camera.enabled = True
    
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=200)
    env.reset(seed=42)
    
    vid_path = out_dir / f"{name}.mp4"
    w = imageio.get_writer(str(vid_path), fps=25, quality=9)
    
    peak_z = env.data.qpos[2]
    
    for step in range(n_steps):
        quat = env.data.qpos[3:7].copy()
        targets = skill_fn(step, env, quat)
        env.step(targets)
        
        z = env.data.qpos[2]
        if z > peak_z: peak_z = z
        
        if step % 2 == 0:
            frame = env.render(camera_name="close")
            lines = [
                f"Peak Height z: {peak_z:.3f} m",
                f"Current z:     {z:.3f} m",
                f"Vertical vz:   {env.data.qvel[2]:+.2f} m/s",
                f"Step: {step}/{n_steps}",
            ]
            annotated = np.array(annotate(frame, f"Restored Explosive Jump: {name}", lines, margin=14), copy=True)
            w.append_data(annotated)
            
    w.close()
    env.close()
    print(f"[{name}] Saved: {vid_path} (peak_z = {peak_z:.3f} m)")
    return peak_z

def jump_up_fn(step, env, quat):
    if step < 20: phase = "crouch"
    elif step < 32: phase = "takeoff"
    elif env.data.qpos[2] > 0.28: phase = "airborne"
    else: phase = "landing"
    return execute_skill("jump_up", quat, env.dirs_body, env.max_extend, phase=phase)

def jump_fwd_fn(step, env, quat):
    if step < 20: phase = "crouch"
    elif step < 32: phase = "takeoff"
    elif env.data.qpos[2] > 0.28: phase = "airborne"
    else: phase = "landing"
    return execute_skill("jump_forward_while_stopped", quat, env.dirs_body, env.max_extend, d_hat=np.array([1.0, 0.0]), phase=phase)

pz1 = render_jump("09_jump_up", jump_up_fn, n_steps=80)
pz2 = render_jump("10_jump_forward_stopped", jump_fwd_fn, n_steps=90)
