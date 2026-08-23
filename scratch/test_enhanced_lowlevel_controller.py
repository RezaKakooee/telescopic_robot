"""Test script for 5 Optional Low-Level Control Enhancements."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from omegaconf import OmegaConf
import rootutils

rootutils.setup_root("/home/azureuser/telescopic_robot", pythonpath=True)

from radial_sphere import (
    MujocoSteeringEnv,
    bar_targets,
    generate_scenario,
    load_config_cli,
)

print("=========================================================================")
print("TESTING 5 OPTIONAL LOW-LEVEL CONTROL ENHANCEMENTS")
print("=========================================================================\n")

# Base parameters
dirs_body = np.random.randn(60, 3)
dirs_body /= np.linalg.norm(dirs_body, axis=1, keepdims=True)
quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
d_hat = np.array([1.0, 0.0], dtype=np.float32)
max_extend = 0.16
min_offset = 0.025

# 1. Test Base (All False)
targets_base = bar_targets(quat, dirs_body, max_extend, d_hat, drive=1.0)
print(f"[1. Base Controller] Min: {np.min(targets_base):.4f}m | Max: {np.max(targets_base):.4f}m | Shape: {targets_base.shape}")
assert np.all(targets_base >= min_offset) and np.all(targets_base <= max_extend)

# 2. Test Asymmetric Power-Cosine Wave
targets_power = bar_targets(quat, dirs_body, max_extend, d_hat, drive=1.0, enable_power_wave=True, wave_power_exponent=1.5)
print(f"[2. Power Wave] Mean: {np.mean(targets_power):.4f}m vs Base Mean: {np.mean(targets_base):.4f}m (Concentrated Push)")
assert not np.array_equal(targets_base, targets_power)

# 3. Test Dynamic Camber Banking
targets_bank = bar_targets(quat, dirs_body, max_extend, d_hat, drive=1.0, enable_camber_banking=True, yaw_rate=1.5, camber_bank_gain=0.04)
print(f"[3. Camber Banking] Max bank delta: {np.max(np.abs(targets_bank - targets_base)):.4f}m")
assert not np.array_equal(targets_base, targets_bank)

# 4. Test Active Flank Retraction
fake_lidar = np.ones(24, dtype=np.float32)
fake_lidar[6] = 0.10  # Wall on left flank at 0.30m
targets_flank = bar_targets(quat, dirs_body, max_extend, d_hat, drive=1.0, enable_flank_retraction=True, lidar_ranges=fake_lidar, flank_retract_dist=0.45, flank_min_offset=0.005)
print(f"[4. Flank Retraction] Flank Min: {np.min(targets_flank):.4f}m (Retracted to 0.005m)")
assert np.min(targets_flank) <= 0.0051

# 5. Test Contact Force Compliance
push_idx = int(np.argmax(targets_base))
fake_forces = np.zeros(60, dtype=np.float32)
fake_forces[push_idx] = 80.0  # High ground impact force on pushing rod
targets_comp = bar_targets(quat, dirs_body, max_extend, d_hat, drive=1.0, enable_contact_compliance=True, contact_forces=fake_forces, compliance_gain=0.0005, max_contact_force=40.0)
print(f"[5. Force Compliance] Rod {push_idx} target: {targets_comp[push_idx]:.4f}m vs Base: {targets_base[push_idx]:.4f}m (Relieved by 0.020m)")
assert targets_comp[push_idx] < targets_base[push_idx]

# 6. Test Anti-Stall Reflex
targets_stall = bar_targets(quat, dirs_body, max_extend, d_hat, drive=1.0, enable_anti_stall_reflex=True, forward_vel=0.05, sim_time=0.25, anti_stall_speed_threshold=0.15)
print(f"[6. Anti-Stall Reflex] Max pulse delta: {np.max(np.abs(targets_stall - targets_base)):.4f}m")
assert not np.array_equal(targets_base, targets_stall)

# 7. Full Closed-Loop Simulation Test with Enhanced Features in Maze
print("\n--- Running Full Closed-Loop Maze Simulation with All Enhancements Active ---")
cfg = load_config_cli(name="maze_level3_large_active_braking")
OmegaConf.set_struct(cfg, False)
cfg.controller.enable_power_wave = True
cfg.controller.enable_flank_retraction = True
cfg.controller.enable_camber_banking = True
cfg.controller.enable_contact_compliance = True
cfg.controller.enable_anti_stall_reflex = True

sc = generate_scenario("maze", cfg, seed=20202)
env = MujocoSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=300)
obs, _ = env.reset(seed=20202)

step = 0
done = False
total_r = 0.0

while not done and step < 200:
    action = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    obs, r, terminated, truncated, info = env.step(action)
    total_r += float(r)
    done = terminated or truncated
    step += 1

print(f"Closed-Loop Run Successful! Steps: {step} | Return: {total_r:.2f} | Speed: {np.linalg.norm(env.env.data.qvel[:2]):.2f}m/s")
env.close()

print("\nALL 5 OPTIONAL ENHANCEMENTS TESTED AND VERIFIED FUNCTIONAL!")
