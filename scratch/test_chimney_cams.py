"""Test front and rear opening cameras for vertical chimney."""
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
env.data.qpos[2] = 0.70
mujoco.mj_forward(env.model, env.data)

env.render(camera_name="fixed_angle_close_3d")

# Camera 1: Front opening view (facing the slot between Box 1 and Box 2)
cam1 = mujoco.MjvCamera()
cam1.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam1.trackbodyid = env.core_body_id
cam1.distance = 2.2
cam1.elevation = -12.0
cam1.azimuth = 180.0  # Front-on view into the fissure

env.renderer.update_scene(env.data, camera=cam1)
img1 = env.renderer.render()
imageio.imwrite("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/chimney_cam_front.png", img1)

# Camera 2: Isometric 3D angle from above looking down the slot
cam2 = mujoco.MjvCamera()
cam2.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam2.trackbodyid = env.core_body_id
cam2.distance = 2.8
cam2.elevation = -35.0
cam2.azimuth = 20.0  # Slight angle

env.renderer.update_scene(env.data, camera=cam2)
img2 = env.renderer.render()
imageio.imwrite("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/chimney_cam_iso.png", img2)

env.close()
