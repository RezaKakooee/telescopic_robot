import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config('configs/rl/config.yaml')
cfg.robot.rod_mechanism = 'multi_stage'
cfg.camera.enabled = False
scenario = generate_scenario('goal', cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100)
env.reset(seed=42)

# Step with simulated command
targets = np.full(env.n_bars, 0.05, dtype=np.float32)
obs, rew, done, trunc, info = env.step(targets)

acc = env.data.sensor('imu_acc').data
gyro = env.data.sensor('imu_gyro').data
frc_0 = env.data.sensor('frc_0').data
pos_0 = env.data.sensor('pos_0').data
touch_0 = env.data.sensor('touch_0').data

print("High-Standard Menagerie Sensor Verification:")
print(f"  Total Bodies:               {env.model.nbody}")
print(f"  Total Joints:               {env.model.njnt} (nq={env.model.nq})")
print(f"  Total Actuators:            {env.model.nu}")
print(f"  Total Equality Constraints: {env.model.neq}")
print(f"  Total Sensors:              {env.model.nsensor}")
print(f"  IMU Accel:                  {acc}")
print(f"  IMU Gyro:                   {gyro}")
print(f"  Bar 0 Joint Pos:            {pos_0[0]:.4f} m")
print(f"  Bar 0 Actuator Force:       {frc_0[0]:.2f} N")
print(f"  Bar 0 Foot Touch Force:     {touch_0[0]:.2f} N")
print("VERIFICATION COMPLETED SUCCESSFULLY!")
env.close()
