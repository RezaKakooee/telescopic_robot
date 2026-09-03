"""Verify that MujocoRadialSphereEnv initializes and steps smoothly under all 3 rod mechanisms."""
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

def test_mechanism(mech_name: str):
    cfg = load_config("configs/rl/config.yaml")
    cfg.robot.rod_mechanism = mech_name
    cfg.robot.appearance_theme = "realistic"
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100)
    obs, info = env.reset(seed=42)
    
    # Step with a push wave
    targets = np.full(env.n_bars, 0.08, dtype=np.float32)
    obs, rew, done, trunc, info = env.step(targets)
    
    print(f"[{mech_name.upper()}] Verification PASSED! nu={env.model.nu}, nq={env.model.nq}, obs_dim={obs.shape}, foot_count={len(env.foot_geom_ids)}")
    env.close()

if __name__ == "__main__":
    for m in ["single_stage", "multi_stage", "zip_chain"]:
        test_mechanism(m)
    print("\nALL 3 ROD MECHANISMS VERIFIED SUCCESSFULLY!")
