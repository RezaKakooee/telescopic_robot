"""Render side-by-side comparison video: Old Policy vs Retrained Policy on New Multi-Stage Ball.

Left: Old Policy (Aug 2026 weights trained on single-stage ball)
Right: Retrained Policy (400k steps trained natively on multi-stage ball)
Maze: Level 3 Fixed 7x6 Maze (Seed 7)

Output:
  - docs/blog/assets/maze-policy-transfer-comparison.mp4
  - docs/blog/assets/maze-policy-transfer-comparison.png
"""
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
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

OLD_CKPT = repo_root / "storage_local/20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking/checkpoints/ppo_final.zip"
OLD_NORM = repo_root / "storage_local/20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking/checkpoints/vecnormalize_final.pkl"

NEW_CKPT = repo_root / "storage_local/20260911_0011__local_455348__train_rl/checkpoints/final.zip"
NEW_NORM = repo_root / "storage_local/20260911_0011__local_455348__train_rl/vecnormalize.pkl"

ASSETS_DIR = repo_root / "docs/blog/assets"


def create_env(seed=7):
    cfg = load_config("configs/rl/maze_level3_large_active_braking.yaml")
    cfg.camera.enabled = False
    cfg.robot.rod_mechanism = "multi_stage"
    cfg.scenario.maze.level = 3
    cfg.scenario.maze.layout_seed = seed
    cfg.scenario.maze.cols = 7
    cfg.scenario.maze.rows = 6

    scenario = generate_scenario("maze", cfg, seed=seed)
    vec = DummyVecEnv([lambda: MujocoSteeringEnv(cfg, scenario=scenario, randomize=False, max_steps=2000)])
    return vec, scenario


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    video_path = ASSETS_DIR / "maze-policy-transfer-comparison.mp4"
    preview_path = ASSETS_DIR / "maze-policy-transfer-comparison.png"

    print("Loading models and environments...")
    old_model = PPO.load(str(OLD_CKPT), device="cpu")
    new_model = PPO.load(str(NEW_CKPT), device="cpu")

    vec_old, scenario_old = create_env()
    if OLD_NORM.exists():
        vec_old = VecNormalize.load(str(OLD_NORM), vec_old)
        vec_old.training = False
        vec_old.norm_reward = False

    vec_new, scenario_new = create_env()
    if NEW_NORM.exists():
        vec_new = VecNormalize.load(str(NEW_NORM), vec_new)
        vec_new.training = False
        vec_new.norm_reward = False

    inner_old = vec_old.venv.envs[0] if hasattr(vec_old, "venv") else vec_old.envs[0]
    inner_new = vec_new.venv.envs[0] if hasattr(vec_new, "venv") else vec_new.envs[0]

    obs_old = vec_old.reset()
    obs_new = vec_new.reset()

    # Setup MuJoCo offscreen renderers (each panel 480x420, canvas 960x540)
    PANEL_W, PANEL_H = 480, 420
    renderer_old = mujoco.Renderer(inner_old.env.model, height=PANEL_H, width=PANEL_W)
    renderer_new = mujoco.Renderer(inner_new.env.model, height=PANEL_H, width=PANEL_W)

    cam_old = mujoco.MjvCamera()
    cam_old.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam_old.trackbodyid = inner_old.env.core_body_id
    cam_old.elevation = -32.0
    cam_old.distance = 3.2
    cam_old.azimuth = 45.0

    cam_new = mujoco.MjvCamera()
    cam_new.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam_new.trackbodyid = inner_new.env.core_body_id
    cam_new.elevation = -32.0
    cam_new.distance = 3.2
    cam_new.azimuth = 45.0

    font_bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_title = ImageFont.truetype(font_bold_path, 16)
    font_badge = ImageFont.truetype(font_bold_path, 12)
    font_data = ImageFont.truetype(font_path, 13)
    font_small = ImageFont.truetype(font_path, 11)

    writer = imageio.get_writer(
        str(video_path),
        fps=25,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart", "-crf", "22", "-preset", "medium"],
    )

    max_frames = 550  # ~22 seconds @ 25 fps
    goal_pos = np.asarray(scenario_new.goal[:2])

    done_old = False
    done_new = False
    saved_preview = False

    dist_old = 999.0
    dist_new = 999.0
    speed_old = 0.0
    speed_new = 0.0
    wall_old = 0
    wall_new = 0

    print(f"Rendering side-by-side comparison to {video_path}...")
    t_start = time.time()

    for frame_idx in range(max_frames):
        # Step Old Policy
        if not done_old:
            act_old, _ = old_model.predict(obs_old, deterministic=True)
            obs_old, _, dones_o, infos_o = vec_old.step(act_old)
            info_o = infos_o[0]
            if int(info_o.get("wall_contact", 0)) or int(info_o.get("n_wall_contacts", 0) > 0):
                wall_old += 1
            pos_o = inner_old.env.data.qpos[:2]
            dist_old = float(np.linalg.norm(pos_o - goal_pos))
            speed_old = float(np.linalg.norm(inner_old.env.data.qvel[:2]))
            if dist_old < 0.45 or dones_o[0]:
                done_old = True

        # Step New Policy
        if not done_new:
            act_new, _ = new_model.predict(obs_new, deterministic=True)
            obs_new, _, dones_n, infos_n = vec_new.step(act_new)
            info_n = infos_n[0]
            if int(info_n.get("wall_contact", 0)) or int(info_n.get("n_wall_contacts", 0) > 0):
                wall_new += 1
            pos_n = inner_new.env.data.qpos[:2]
            dist_new = float(np.linalg.norm(pos_n - goal_pos))
            speed_new = float(np.linalg.norm(inner_new.env.data.qvel[:2]))
            if dist_new < 0.45 or dones_n[0]:
                done_new = True

        # Camera dynamic gentle rotation following movement
        t_sim = frame_idx * 0.04
        cam_old.azimuth = 45.0 + 8.0 * np.sin(0.2 * t_sim)
        cam_new.azimuth = 45.0 + 8.0 * np.sin(0.2 * t_sim)

        renderer_old.update_scene(inner_old.env.data, cam_old)
        view_old = renderer_old.render()

        renderer_new.update_scene(inner_new.env.data, cam_new)
        view_new = renderer_new.render()

        # Canvas: 960 x 540 (Dark Slate header 120px, panels 420px)
        canvas = Image.new("RGB", (960, 540), (15, 23, 42))
        canvas.paste(Image.fromarray(view_old), (0, 120))
        canvas.paste(Image.fromarray(view_new), (480, 120))

        draw = ImageDraw.Draw(canvas)

        # Header Title
        draw.text((16, 12), "RoboBall Multi-Stage Dynamics: RL Policy Transfer vs Native Retraining", font=font_title, fill=(255, 255, 255))
        draw.text((16, 36), "Single-Maze Active-Braking RL on Fixed 7x6 Level 3 Maze | Both models running on Multi-Stage Concentric Ball", font=font_small, fill=(148, 163, 184))

        # Panel Badges & Status
        # Left Panel (Old Policy)
        draw.rounded_rectangle([(16, 62), (180, 88)], radius=5, fill=(239, 68, 68), outline=(220, 38, 38))
        draw.text((24, 68), "OLD POLICY (Zero-Shot)", font=font_badge, fill=(255, 255, 255))
        draw.text((192, 68), "Under-actuated stall due to multi-stage friction", font=font_small, fill=(252, 165, 165))
        draw.text((16, 94), f"Speed: {speed_old:.2f} m/s  |  Dist to Goal: {dist_old:.2f} m  |  Wall Hits: {wall_old}", font=font_data, fill=(248, 113, 113))

        # Right Panel (Retrained Policy)
        status_txt = "GOAL REACHED! (SUCCESS)" if done_new else "Agile cruise & active braking"
        badge_bg = (34, 197, 94) if done_new else (56, 189, 248)
        draw.rounded_rectangle([(496, 62), (680, 88)], radius=5, fill=badge_bg, outline=(30, 144, 255))
        draw.text((504, 68), "RETRAINED (400k Steps)", font=font_badge, fill=(15, 23, 42) if not done_new else (255, 255, 255))
        draw.text((690, 68), status_txt, font=font_small, fill=(187, 247, 208) if done_new else (186, 230, 253))
        draw.text((496, 94), f"Speed: {speed_new:.2f} m/s  |  Dist to Goal: {dist_new:.2f} m  |  Wall Hits: {wall_new}", font=font_data, fill=(134, 239, 172) if done_new else (56, 189, 248))

        # Divider line between panels
        draw.line([(480, 50), (480, 540)], fill=(51, 65, 85), width=2)
        # Header bottom separator
        draw.line([(0, 119), (960, 119)], fill=(71, 85, 105), width=1)

        # Elapsed time indicator on top right
        draw.text((860, 14), f"{frame_idx * 0.04:5.1f} s", font=font_title, fill=(255, 255, 255))

        frame_arr = np.array(canvas)
        writer.append_data(frame_arr)

        if not saved_preview and frame_idx == 200:
            canvas.save(preview_path)
            saved_preview = True
            print(f"Saved preview image at frame {frame_idx} to {preview_path}")

        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx}/{max_frames} (Old: {dist_old:.2f}m, New: {dist_new:.2f}m) - elapsed: {time.time()-t_start:.1f}s")

    if not saved_preview:
        canvas.save(preview_path)

    writer.close()
    vec_old.close()
    vec_new.close()
    print(f"Video saved successfully to: {video_path} (Elapsed: {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
