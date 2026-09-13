"""Zero-Shot Baseline Evaluation of Flat-Obstacle VLA Policy on the 3D Playground Parkour Task.

Tests the existing 2D-steering VLA on the multi-station playground course
to quantify the domain gap before parkour adaptation.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill
from scripts.data.generate_obstacle_vla_dataset import render_vla_frame
from scripts.vla.train_vla import RoboballVLAPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_vla_baseline")


def draw_hud(
    frame: np.ndarray,
    step: int,
    dist_goal: float,
    progress_xy: tuple[float, float],
    pred_action: np.ndarray,
    status_text: str = "",
) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Top banner
    draw.rectangle([(0, 0), (img.width, 36)], fill=(15, 23, 42))
    text = (
        f"Step {step:03d} | Pos: ({progress_xy[0]:.2f}, {progress_xy[1]:.2f}) | "
        f"Goal: {dist_goal:.2f}m | "
        f"Act: [{pred_action[0]:+.2f}, {pred_action[1]:+.2f}]"
    )
    draw.text((8, 10), text, fill=(241, 245, 249))

    if status_text:
        col = (34, 197, 94) if "SUCCESS" in status_text else (239, 68, 68)
        draw.rectangle([(0, img.height - 30), (img.width, img.height)], fill=(15, 23, 42))
        draw.text((10, img.height - 24), status_text, fill=col)

    return np.asarray(img)


def evaluate_baseline_episode(
    policy: RoboballVLAPolicy,
    cfg,
    device: torch.device,
    max_steps: int = 500,
    seed: int = 42,
    record_frames: bool = True,
) -> dict:
    scenario = generate_scenario("playground", cfg, seed=seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=max_steps + 50)
    env.reset(seed=seed)

    # Initial settle
    for _ in range(20):
        env.step(np.full(len(env.dirs_body), 0.025, dtype=np.float32))

    goal = np.asarray(scenario.goal, dtype=np.float32)[:2]
    step = 0
    frames_recorded = []
    stuck_counter = 0
    last_pos = env.data.qpos[:2].copy()

    while step < max_steps:
        pos = env.data.qpos[:3].copy()
        quat = env.data.qpos[3:7].copy()
        vel = env.data.qvel[:3].copy()
        ball_xy = pos[:2]

        rel_goal = goal - ball_xy
        dist_goal = float(np.linalg.norm(rel_goal))

        frame = render_vla_frame(env, width=256, height=256)

        state_vec = np.concatenate([
            ball_xy.astype(np.float32),
            vel[:2].astype(np.float32),
            rel_goal.astype(np.float32),
            np.array([dist_goal], dtype=np.float32),
            quat.astype(np.float32),
        ])

        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        st_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred_act = policy(img_tensor, st_tensor).squeeze(0).cpu().numpy()

        v_world = pred_act[:2]
        v_norm = float(np.linalg.norm(v_world))
        if v_norm > 0.15:
            d_hat = v_world / v_norm
            speed = float(np.clip(v_norm, 0.6, 1.4))
        else:
            d_hat = rel_goal / max(dist_goal, 1e-4)
            speed = 0.8

        rod_targets = execute_skill(
            "move_forward",
            quat,
            env.dirs_body,
            env.max_extend,
            d_hat=d_hat,
            speed=speed,
            lin_vel=vel[:2],
        )

        env.step(rod_targets)

        now_pos = env.data.qpos[:2].copy()
        disp = float(np.linalg.norm(now_pos - last_pos))
        if disp < 0.01:
            stuck_counter += 1
        else:
            stuck_counter = 0
        last_pos = now_pos.copy()

        if record_frames:
            hud_frame = draw_hud(frame, step, dist_goal, (float(now_pos[0]), float(now_pos[1])), pred_act)
            frames_recorded.append(hud_frame)

        step += 1

        # Terminate early if stuck on an obstacle/hurdle for > 60 steps
        if stuck_counter > 60 or dist_goal < 0.35:
            break

    final_pos = env.data.qpos[:2].copy()
    final_dist = float(np.linalg.norm(final_pos - goal))
    success = bool(final_dist < 0.35)
    env.close()

    if record_frames and frames_recorded:
        badge = f"RESULT: {'SUCCESS' if success else f'FAILED (Stuck at x={final_pos[0]:.2f}, y={final_pos[1]:.2f})'}"
        frames_recorded.append(draw_hud(frames_recorded[-1], step, final_dist, (float(final_pos[0]), float(final_pos[1])), pred_act, badge))

    return {
        "steps": step,
        "final_x": float(final_pos[0]),
        "final_y": float(final_pos[1]),
        "final_dist_to_goal": final_dist,
        "success": success,
        "frames": frames_recorded,
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot baseline evaluation of flat VLA on playground.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="storage_local/20260913_1525__local__train_vla__obstacle_navigation/checkpoints/best_policy.pt",
        help="Path to trained VLA checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/evaluation/vla_zero_shot",
        help="Output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/code/config.yaml",
        help="Playground config path",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = RoboballVLAPolicy().to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    cfg = load_config(args.config)
    res = evaluate_baseline_episode(policy, cfg, device, max_steps=400)

    logger.info("=" * 60)
    logger.info("ZERO-SHOT PLAYGROUND BASELINE EVALUATION COMPLETE")
    logger.info(f"Final Position:   ({res['final_x']:.2f}, {res['final_y']:.2f})")
    logger.info(f"Final Dist Goal:  {res['final_dist_to_goal']:.2f}m")
    logger.info(f"Steps:            {res['steps']}")
    logger.info(f"Success:          {res['success']}")
    logger.info("=" * 60)

    summary_file = out_dir / "zero_shot_summary.json"
    with open(summary_file, "w") as f:
        json.dump({k: v for k, v in res.items() if k != "frames"}, f, indent=2)

    if res["frames"]:
        video_path = out_dir / "zero_shot_playground_rollout.mp4"
        imageio.mimsave(str(video_path), res["frames"], fps=25)
        logger.info(f"Saved video to: {video_path}")


if __name__ == "__main__":
    main()
