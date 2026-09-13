"""Generate sample camera snapshots from RoboBallVisionWrapper to inspect perspectives."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import cv2
import numpy as np

from radial_sphere.config import load_config_cli
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from radial_sphere.vision_wrapper import RoboBallVisionWrapper

ARTIFACTS_DIR = Path("/home/azureuser/.gemini/antigravity-ide/brain/e14032e8-5276-4443-8925-16d7ecbcdbd7")


def sample_camera(camera_name: str, out_filename: str):
    cfg = load_config_cli(name="playground_parkour_skills")
    scenario = generate_scenario("playground", cfg, seed=100)
    base_env = SkillArbitrationEnv(cfg, scenario=scenario, seed=100)
    env = RoboBallVisionWrapper(
        base_env,
        camera_name=camera_name,
        image_size=(384, 384),
        channels_first=False,
    )
    obs, info = env.reset(seed=100)

    # Let the robot roll 15 steps into the course to show the corridor and obstacles
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    # Action for move
    action[0] = 1.0  # move skill
    for _ in range(3):
        obs, r, term, trunc, info = env.step(action)

    frame = obs["image"]  # (384, 384, 3) RGB
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    out_path = ARTIFACTS_DIR / out_filename
    cv2.imwrite(str(out_path), bgr)
    print(f"Saved {camera_name} snapshot to {out_path}")
    env.close()


def print_cam(env):
    data = env._underlying.data
    pos = data.qpos[:3]
    heading = env._cam_heading
    print(f"DEBUG: robot pos={pos[:2]}, cam_heading={np.degrees(heading):.1f} deg")


if __name__ == "__main__":
    sample_camera("front", "front_cam_sample.png")
    sample_camera("birdview", "birdview_cam_sample.png")
    sample_camera("chase", "chase_cam_sample.png")

