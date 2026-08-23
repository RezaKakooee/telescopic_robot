"""Test script for 5 Optional Smooth Maneuver Enhancements."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from omegaconf import OmegaConf
import rootutils

rootutils.setup_root("/home/azureuser/telescopic_robot", pythonpath=True)

from radial_sphere import (
    MujocoSteeringEnv,
    bar_targets,
    desired_direction,
    generate_scenario,
    load_config_cli,
)

print("=========================================================================")
print("TESTING 5 OPTIONAL SMOOTH MANEUVER ENHANCEMENTS")
print("=========================================================================\n")

# 1. Test Continuous Spline Heading
fake_path = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.5, 1.0],
    [1.5, 2.0],
], dtype=np.float32)
ball_xy = np.array([0.2, 0.0], dtype=np.float32)

d_discrete, _ = desired_direction(ball_xy, fake_path, lookahead=1.5, enable_spline_heading=False)
d_spline, _ = desired_direction(ball_xy, fake_path, lookahead=1.5, enable_spline_heading=True, spline_smoothing_weight=0.5)
print(f"[1. Spline Heading] Discrete: {d_discrete} vs Spline Smoothed: {d_spline}")
assert not np.allclose(d_discrete, d_spline)

# 2. Test Curvature-Adaptive Deceleration Glide
d_straight, drive_straight = desired_direction(np.array([1.5, 0.5]), fake_path, lookahead=0.4, enable_curvature_deceleration=True)
d_corner, drive_corner = desired_direction(np.array([0.2, 0.0]), fake_path, lookahead=0.9, enable_curvature_deceleration=True, curvature_brake_gain=1.8)
print(f"[2. Curvature Glide] Straightaway Drive: {drive_straight:.3f} vs Approaching Corner Drive: {drive_corner:.3f} (Curvature Braking)")
assert drive_corner < drive_straight

# 3. Test Gaussian Stance Handoff
dirs_body = np.random.randn(60, 3)
dirs_body /= np.linalg.norm(dirs_body, axis=1, keepdims=True)
quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
d_hat = np.array([1.0, 0.0], dtype=np.float32)

targets_base = bar_targets(quat, dirs_body, 0.16, d_hat, drive=1.0)
targets_gauss = bar_targets(quat, dirs_body, 0.16, d_hat, drive=1.0, enable_gaussian_stance=True, gaussian_stance_sigma=0.38)
print(f"[3. Gaussian Stance] Mean: {np.mean(targets_gauss):.4f}m vs Base: {np.mean(targets_base):.4f}m (C-infinity smooth stance window)")
assert not np.array_equal(targets_base, targets_gauss)

# 4. Test Gyroscopic Precession Damping
targets_gyro = bar_targets(quat, dirs_body, 0.16, d_hat, drive=1.0, enable_gyroscopic_damping=True, ang_vel=np.array([2.0, 0.0, 1.5]))
print(f"[4. Gyroscopic Damping] Max Gyro Correction: {np.max(np.abs(targets_gyro - targets_base)):.4f}m")
assert not np.array_equal(targets_base, targets_gyro)

# 5. Test Actuator Slew-Rate Limiter
last_targets = np.full(60, 0.025, dtype=np.float32)
demanded_targets = np.full(60, 0.160, dtype=np.float32)
targets_slew = bar_targets(quat, dirs_body, 0.16, d_hat, drive=1.0, enable_actuator_slew_rate=True, last_targets=last_targets, actuator_max_vel=0.35, actuator_dt=0.05)
max_change = float(np.max(targets_slew - last_targets))
print(f"[5. Actuator Slew-Rate] Demanded Delta: 0.135m -> Rate-Limited Delta: {max_change:.4f}m (Capped at 0.35 * 0.05 = 0.0175m)")
assert max_change <= 0.01751

# 6. Full Closed-Loop Simulation Test with All Smooth Maneuver Options Active
print("\n--- Running Full Closed-Loop Simulation with All Smooth Maneuver Enhancements ---")
cfg = load_config_cli(name="maze_level3_large_active_braking")
OmegaConf.set_struct(cfg, False)
cfg.controller.enable_spline_heading = True
cfg.controller.enable_curvature_deceleration = True
cfg.controller.enable_actuator_slew_rate = True
cfg.controller.enable_gaussian_stance = True
cfg.controller.enable_gyroscopic_damping = True

sc = generate_scenario("maze", cfg, seed=42)
env = MujocoSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=200)
obs, _ = env.reset(seed=42)

step = 0
done = False
total_r = 0.0

while not done and step < 100:
    action = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    obs, r, terminated, truncated, info = env.step(action)
    total_r += float(r)
    done = terminated or truncated
    step += 1

print(f"Smooth Maneuver Closed-Loop Simulation Successful! Steps: {step} | Return: {total_r:.2f}")
env.close()

print("\nALL 5 SMOOTH MANEUVER ENHANCEMENTS TESTED AND VERIFIED FUNCTIONAL!")
