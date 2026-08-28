"""Test and verify non-stalling locomotion on Slopes, Stairs, Glass Pipe, and Gauntlet."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.controller import bar_targets, desired_direction
from radial_sphere.scenario import generate_scenario
import mujoco

def test_scenario(kind: str):
    print(f"\n--- Testing Scenario: {kind.upper()} ---")
    cfg = load_config("configs/rl/terrain_slopes_and_ramps.yaml")
    scenario = generate_scenario(kind, cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=1200)

    obs, info = env.reset(seed=42)
    ctrl = env.cfg.controller

    for step in range(1, 1001):
        ball_pos = env.data.qpos[0:3]
        ball_vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]

        # Pitch angle along travel direction
        R_mat = np.zeros((3, 3))
        mujoco.mju_quat2Mat(R_mat.reshape(-1), quat)
        pitch_angle = float(np.arcsin(np.clip(R_mat[0, 2], -1.0, 1.0)))

        d_hat, drive = desired_direction(ball_pos[:2], env.path_pts, lookahead=1.0)
        targets = bar_targets(
            quat,
            env.dirs_body,
            env.max_extend,
            d_hat,
            drive=drive,
            min_offset=0.0,
            back_gain=2.5,
            enable_curb_vaulting=True,
            curb_boost_gain=3.0,
            enable_incline_assist=True,
            incline_pitch=pitch_angle,
            incline_boost_gain=2.0,
            enable_pipe_bracing=(kind == "glass_pipe" or kind == "extreme_gauntlet"),
            pipe_bracing_gain=0.42,
            enable_underbelly_contact=True,
            underbelly_stance_gain=0.40,
            underbelly_threshold_z=-0.20,
            enable_active_suspension=True,
            core_z=float(ball_pos[2]),
            core_vz=float(ball_vel[2]),
            target_ride_height=0.28,
            suspension_kp=0.75,
            suspension_kd=0.15,
        )

        obs, rew, term, trunc, info = env.step(targets)

        if step % 50 == 0 or term or trunc or info["distance"] < 0.45:
            print(f"Step {step:4d}: Ball pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.2f}), vx={ball_vel[0]:.2f}m/s, Dist to Goal={info['distance']:.2f}m")

        if term or trunc or info["distance"] < 0.45:
            print(f"✅ SUCCESS! Reached goal in {step} steps (dist={info['distance']:.3f}m)!")
            env.close()
            return True

    env.close()
    print(f"❌ Timed out before goal. Final pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f})")
    return False

if __name__ == "__main__":
    for sc in ["slopes", "stairs", "glass_pipe", "extreme_gauntlet"]:
        test_scenario(sc)
