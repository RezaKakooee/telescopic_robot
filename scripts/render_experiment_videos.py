"""Render demonstration videos of the retrained policy navigating mazes.

Saves all MP4 videos directly to the experiment run directory:
storage_local/20260911_0011__local_455348__train_rl/renders/
"""
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

os.environ.setdefault("MUJOCO_GL", "egl")
import time
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from radial_sphere.config import load_config
from radial_sphere.mujoco_steering import MujocoSteeringEnv
from radial_sphere.scenario import generate_scenario

EXP_DIR = repo_root / "storage_local/20260911_0011__local_455348__train_rl"
CHECKPOINT_PATH = EXP_DIR / "checkpoints/final.zip"
NORM_PATH = EXP_DIR / "vecnormalize.pkl"
RENDERS_DIR = EXP_DIR / "renders"

MAZES_TO_RENDER = [
    {
        "id": "maze_0_training_fixed",
        "title": "Training Maze (Level 3, Fixed 7x6)",
        "badge": "TRAINING MAZE",
        "level": 3,
        "seed": 7,
        "cols": 7,
        "rows": 6,
        "max_steps": 550,
        "filename": "maze_0_training_level3.mp4",
    },
    {
        "id": "maze_5_large_45m_gauntlet",
        "title": "Maze 5: Large 7x6 45m Gauntlet (Level 3)",
        "badge": "UNSEEN ZERO-SHOT BENCHMARK",
        "level": 3,
        "seed": 50505,
        "cols": 7,
        "rows": 6,
        "max_steps": 650,
        "filename": "maze_5_unseen_gauntlet_level3.mp4",
    },
    {
        "id": "maze_2_multiloop_braid",
        "title": "Maze 2: High-Density Multi-Loop Braid (Level 2)",
        "badge": "UNSEEN ZERO-SHOT BENCHMARK",
        "level": 2,
        "seed": 20202,
        "cols": None,
        "rows": None,
        "max_steps": 480,
        "filename": "maze_2_unseen_multiloop_braid_level2.mp4",
    },
    {
        "id": "maze_6_dense_switchback",
        "title": "Maze 6: Dense S-Curve Switchback (Level 3)",
        "badge": "UNSEEN ZERO-SHOT BENCHMARK",
        "level": 3,
        "seed": 60606,
        "cols": None,
        "rows": None,
        "max_steps": 550,
        "filename": "maze_6_unseen_switchback_level3.mp4",
    },
]


def render_maze_video(m_info, model, norm_path, renders_dir):
    title = m_info["title"]
    badge_name = m_info["badge"]
    filename = m_info["filename"]
    level = m_info["level"]
    seed = m_info["seed"]
    max_steps = m_info["max_steps"]
    video_path = renders_dir / filename

    print(f"\n--- Rendering: {title} ---")
    cfg = load_config("configs/rl/maze_level3_large_active_braking.yaml")
    cfg.camera.enabled = False
    cfg.robot.rod_mechanism = "multi_stage"
    cfg.scenario.maze.level = level
    cfg.scenario.maze.layout_seed = seed
    if m_info["cols"] is not None:
        cfg.scenario.maze.cols = m_info["cols"]
        cfg.scenario.maze.rows = m_info["rows"]

    scenario = generate_scenario("maze", cfg, seed=seed)
    vec = DummyVecEnv([lambda: MujocoSteeringEnv(cfg, scenario=scenario, randomize=False, max_steps=max_steps)])
    if norm_path.exists():
        vec = VecNormalize.load(str(norm_path), vec)
        vec.training = False
        vec.norm_reward = False

    inner = vec.venv.envs[0] if hasattr(vec, "venv") else vec.envs[0]
    obs = vec.reset()

    # Resolution: 960x544 (Viewport 960x448, Header 96px)
    VIEW_W, VIEW_H = 960, 448
    inner.env.model.vis.global_.offwidth = VIEW_W
    inner.env.model.vis.global_.offheight = VIEW_H
    renderer = mujoco.Renderer(inner.env.model, height=VIEW_H, width=VIEW_W)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = inner.env.core_body_id
    camera.elevation = -30.0
    camera.distance = 3.3
    camera.azimuth = 50.0

    font_bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_title = ImageFont.truetype(font_bold_path, 17)
    font_badge = ImageFont.truetype(font_bold_path, 12)
    font_data = ImageFont.truetype(font_path, 14)
    font_small = ImageFont.truetype(font_path, 12)

    writer = imageio.get_writer(
        str(video_path),
        fps=25,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart", "-crf", "22", "-preset", "medium"],
    )

    goal_pos = np.asarray(scenario.goal[:2])
    done = False
    success = False
    wall_hits = 0
    step_cnt = 0

    t_start = time.time()
    for frame_idx in range(max_steps):
        if not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec.step(action)
            info = infos[0]
            step_cnt += 1
            if int(info.get("wall_contact", 0)) or int(info.get("n_wall_contacts", 0) > 0):
                wall_hits += 1

            pos = inner.env.data.qpos[:2]
            dist_to_goal = float(np.linalg.norm(pos - goal_pos))
            speed = float(np.linalg.norm(inner.env.data.qvel[:2]))

            if dist_to_goal < 0.45:
                done = True
                success = True
            elif dones[0]:
                done = True

        # Camera gentle rotation
        t_sim = step_cnt * 0.02
        camera.azimuth = 50.0 + 10.0 * np.sin(0.18 * t_sim)

        renderer.update_scene(inner.env.data, camera)
        viewport = renderer.render()

        # Canvas: 960 x 544
        canvas = Image.new("RGB", (960, 544), (15, 23, 42))
        canvas.paste(Image.fromarray(viewport), (0, 96))

        draw = ImageDraw.Draw(canvas)

        # Title & Model info
        draw.text((16, 12), f"RoboBall Multi-Stage RL: {title}", font=font_title, fill=(255, 255, 255))
        draw.text((16, 36), "Model: PPO Active-Braking (400k steps) | Native Multi-Stage Concentric Antenna Rods", font=font_small, fill=(148, 163, 184))

        # Status badge
        if success:
            badge_color = (34, 197, 94)  # Green
            status_str = "GOAL REACHED! (SUCCESS)"
        elif done:
            badge_color = (239, 68, 68)  # Red
            status_str = "TERMINATED"
        else:
            badge_color = (56, 189, 248)  # Blue
            status_str = "AGILE LOCOMOTION & ACTIVE BRAKING"

        # Badge rectangle
        draw.rounded_rectangle([(16, 60), (220, 86)], radius=5, fill=(30, 41, 59), outline=(71, 85, 105))
        draw.text((24, 66), badge_name, font=font_badge, fill=(250, 204, 21))

        # Status badge
        draw.rounded_rectangle([(230, 60), (510, 86)], radius=5, fill=badge_color, outline=badge_color)
        draw.text((240, 66), status_str, font=font_badge, fill=(15, 23, 42) if not success else (255, 255, 255))

        # Telemetry metrics
        collision_pct = (wall_hits / max(step_cnt, 1)) * 100.0
        draw.text(
            (530, 66),
            f"Speed: {speed:.2f} m/s  |  Dist: {dist_to_goal:.2f} m  |  Wall Hits: {wall_hits} ({collision_pct:.1f}%)",
            font=font_data,
            fill=(203, 213, 225),
        )

        # Time on top right
        draw.text((855, 14), f"{t_sim:5.1f} s", font=font_title, fill=(255, 255, 255))

        # Separator line
        draw.line([(0, 95), (960, 95)], fill=(51, 65, 85), width=2)

        writer.append_data(np.array(canvas))

        if done and frame_idx > step_cnt + 25:
            # Hold last frame for ~1s
            break

    writer.close()
    vec.close()
    elapsed = time.time() - t_start
    print(f"Finished {filename}: Steps={step_cnt}, Solved={success}, WallHits={wall_hits} ({elapsed:.1f}s)")
    return video_path


def main():
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading PPO model from {CHECKPOINT_PATH}...")
    model = PPO.load(str(CHECKPOINT_PATH), device="cpu")

    # Also copy the side-by-side comparison video if it exists
    src_comp = repo_root / "docs/blog/assets/maze-policy-transfer-comparison.mp4"
    if src_comp.exists():
        import shutil
        dest_comp = RENDERS_DIR / "maze_policy_transfer_comparison.mp4"
        shutil.copyfile(src_comp, dest_comp)
        print(f"Copied side-by-side video to {dest_comp}")

    for m_info in MAZES_TO_RENDER:
        render_maze_video(m_info, model, NORM_PATH, RENDERS_DIR)

    print(f"\nAll videos rendered successfully into: {RENDERS_DIR}")


if __name__ == "__main__":
    main()
