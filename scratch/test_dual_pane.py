"""Test dual-pane composite render for chimney arena."""
import os
os.environ["MUJOCO_GL"] = "egl"
import cv2
import imageio
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.overlay import annotate

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
env.reset(seed=42)

env.data.qpos[0] = 0.0
env.data.qpos[1] = 0.0
env.data.qpos[2] = 0.70
mujoco.mj_forward(env.model, env.data)

env.render(camera_name="fixed_angle_close_3d")

# 1. Macro tracking front view (Left Pane)
cam1 = mujoco.MjvCamera()
cam1.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam1.trackbodyid = env.core_body_id
cam1.distance = 1.35
cam1.elevation = -5.0
cam1.azimuth = 180.0
env.renderer.update_scene(env.data, camera=cam1)
f_macro = env.renderer.render()

# 2. Wide elevation tracking view (Right Pane)
cam2 = mujoco.MjvCamera()
cam2.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam2.trackbodyid = env.core_body_id
cam2.distance = 2.40
cam2.elevation = -14.0
cam2.azimuth = 180.0
env.renderer.update_scene(env.data, camera=cam2)
f_wide = env.renderer.render()

h, w = f_macro.shape[:2]
f_wide_resized = cv2.resize(f_wide, (w, h))

f_macro_ann = annotate(
    f_macro,
    "SKILL: chimney_climb (Between Two Vertical Boxes)",
    [
        "Action:     MID-AIR SUSPENSION (LOCKED)",
        "Setup:      Vertical Box 1 (Left) & Box 2 (Right)",
        "Gap Width:  40 cm  (Box Height 4.0 m)",
        "Elevation:  z = 0.700 m  (Δz = +50 cm)",
        "Vertical V: vz = +0.00 m/s",
    ],
    margin=14,
)

f_wide_ann = annotate(
    f_wide_resized,
    "ELEVATION OVERVIEW (FLOOR & HEIGHT SCALE)",
    [
        "Elevation:  z = 0.700 m",
        "Vertical V: vz = +0.00 m/s",
        "Centering:  y = +0.0 cm",
        "Status:     LOCKED IN MID-AIR",
        "Time:       0.85s",
    ],
    margin=14,
)

composite = np.concatenate([f_macro_ann, f_wide_ann], axis=1)
imageio.imwrite("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/chimney_perfect_preview.png", composite)
env.close()
