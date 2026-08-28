"""Test rear chase camera down the trench."""
import os
os.environ["MUJOCO_GL"] = "egl"
import imageio
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/gap_bridge.yaml")
scenario = generate_scenario("gap_bridge", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
env.reset(seed=42)

env.data.qpos[0] = 0.5
env.data.qpos[1] = 0.0
env.data.qpos[2] = 0.25 + 0.19
mujoco.mj_forward(env.model, env.data)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = env.core_body_id
cam.distance = 1.8
cam.elevation = -12.0
cam.azimuth = 0.0  # looking directly down the trench from behind

env.render(camera_name="fixed_angle_close_3d")
env.renderer.update_scene(env.data, camera=cam)

img = env.renderer.render()
imageio.imwrite("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/rear_trench_view.png", img)
env.close()
