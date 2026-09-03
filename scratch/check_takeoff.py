"""Inspect exact geom positions during takeoff in MujocoRadialSphereEnv for multi_stage and zip_chain."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.jumping import jump_up

for mech in ["multi_stage", "zip_chain"]:
    cfg = load_config("configs/rl/config.yaml")
    cfg.robot.rod_mechanism = mech
    cfg.robot.appearance_theme = "realistic"
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    # Takeoff targets
    quat = env.data.qpos[3:7].copy()
    targets = jump_up(quat, env.dirs_body, env.max_extend, phase="takeoff")
    for _ in range(30):
        env.step(targets)

    # Find a bar with max extension (ground_mask bar)
    ground_bars = np.where(targets == env.max_extend)[0]
    k = ground_bars[0]
    
    print(f"\n=== {mech.upper()} BAR {k} AT TAKEOFF (ctrl = {targets[k]:.3f}) ===")
    core_pos = env.data.xpos[env.core_body_id]
    u = env.dirs_body[k]
    
    # Sleeve geom
    s_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"sleeve_{k}")
    s_pos = env.data.geom_xpos[s_gid] - core_pos
    print(f"  sleeve_{k} dist: {np.linalg.norm(s_pos):.4f} m")

    # Stage1 geom
    st1_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"stage1_geom_{k}")
    if st1_gid >= 0:
        st1_pos = env.data.geom_xpos[st1_gid] - core_pos
        print(f"  stage1_geom_{k} dist: {np.linalg.norm(st1_pos):.4f} m")

    # Inner geom
    in_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"inner_geom_{k}")
    in_pos = env.data.geom_xpos[in_gid] - core_pos
    print(f"  inner_geom_{k} dist: {np.linalg.norm(in_pos):.4f} m")

    # Foot geom
    f_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{k}")
    f_pos = env.data.geom_xpos[f_gid] - core_pos
    print(f"  foot_{k} dist: {np.linalg.norm(f_pos):.4f} m")
    
    # Joint values
    j_slide = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"slide_{k}")
    j_qpos_adr = env.model.jnt_qposadr[j_slide]
    print(f"  slide_{k} qpos: {env.data.qpos[j_qpos_adr]:.4f} m")
    if mech in ["multi_stage", "zip_chain"]:
        j_slide1 = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"slide1_{k}")
        j1_qpos_adr = env.model.jnt_qposadr[j_slide1]
        print(f"  slide1_{k} qpos: {env.data.qpos[j1_qpos_adr]:.4f} m")

    env.close()
