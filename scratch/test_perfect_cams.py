"""Test perfect unobstructed cameras for vertical chimney climbing."""
import os
os.environ["MUJOCO_GL"] = "egl"
import imageio
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
env.reset(seed=42)

env.data.qpos[0] = 0.0
env.data.qpos[1] = 0.0
env.data.qpos[2] = 0.65
mujoco.mj_forward(env.model, env.data)

env.render(camera_name="fixed_angle_close_3d")

# Camera 1: Macro Close-up front view into the gap
cam1 = mujoco.MjvCamera()
cam1.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam1.trackbodyid = env.core_body_id
cam1.distance = 1.45
cam1.elevation = -6.0
cam1.azimuth = 180.0

env.renderer.update_scene(env.data, camera=cam1)
img1 = env.renderer.render()
imageio.imwrite("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/cam_macro.png", img1)

# Camera 2: Elevated 3D front view
cam2 = mujoco.MjvCamera()
cam2.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam2.trackbodyid = env.core_body_id
cam2.distance = 2.60
cam2.elevation = -22.0
cam2.azimuth = 180.0

env.renderer.update_scene(env.data, camera=cam2)
img2 = env.renderer.render()
imageio.imwrite("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/cam_elevated.png", img2)

env.close()
