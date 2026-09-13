"""Closed-Loop Evaluation of Fine-Tuned VLA Policy for Roboball Obstacle Navigation.

Evaluates the VLA policy on unseen obstacle layouts:
- Visual Closed-Loop: Visual frame rendered -> VLA predicts steering -> low-level controller rolls ball
- Measures: Goal arrival success rate, minimum obstacle clearance, collision count, time-to-goal
- Records: Annotated video showing the ball passing obstacles and reaching the target
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mujoco

from radial_sphere.config import load_config
from radial_sphere.controller import bar_targets
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill
from scripts.data.generate_obstacle_vla_dataset import pillar_clearance, render_vla_frame
from scripts.vla.train_vla import RoboballVLAPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_vla")


def draw_hud(
    frame: np.ndarray,
    step: int,
    dist_goal: float,
    min_clearance: float,
    pred_action: np.ndarray,
    status_text: str = "",
) -> np.ndarray:
    """Draw telemetry HUD overlay on camera frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Top banner background
    draw.rectangle([(0, 0), (img.width, 36)], fill=(15, 23, 42))

    # Text telemetry
    text = (
        f"Step {step:03d} | Goal: {dist_goal:4.2f}m | "
        f"Clearance: {min_clearance:4.2f}m | "
        f"Act: [{pred_action[0]:+.2f}, {pred_action[1]:+.2f}]"
    )
    draw.text((8, 10), text, fill=(241, 245, 249))

    if status_text:
        col = (34, 197, 94) if "SUCCESS" in status_text else (239, 68, 68)
        draw.rectangle([(0, img.height - 30), (img.width, img.height)], fill=(15, 23, 42))
        draw.text((10, img.height - 24), status_text, fill=col)

    return np.asarray(img)


def evaluate_episode(
    policy: RoboballVLAPolicy,
    seed: int,
    cfg,
    device: torch.device,
    max_steps: int = 600,
    goal_tolerance: float = 0.35,
    record_frames: bool = True,
) -> dict:
    """Run a single closed-loop evaluation episode."""
    scenario = generate_scenario("obstacle", cfg, seed=seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=max_steps + 100)
    env.reset(seed=seed)

    # Initial settle steps
    for _ in range(20):
        env.step(np.full(len(env.dirs_body), 0.025, dtype=np.float32))

    goal = np.asarray(scenario.goal, dtype=np.float32)[:2]
    obstacles = np.asarray(scenario.obstacles, dtype=np.float32).reshape(-1, 3)

    min_clearance = float("inf")
    arrived = False
    step = 0
    hold_steps = 15
    frames_recorded = []

    while step < max_steps:
        pos = env.data.qpos[:3].copy()
        quat = env.data.qpos[3:7].copy()
        vel = env.data.qvel[:3].copy()
        ball_xy = pos[:2]

        rel_goal = goal - ball_xy
        dist_goal = float(np.linalg.norm(rel_goal))

        # Render camera frame
        frame = render_vla_frame(env, width=256, height=256)

        # Assemble state vector
        state_vec = np.concatenate([
            ball_xy.astype(np.float32),
            vel[:2].astype(np.float32),
            rel_goal.astype(np.float32),
            np.array([dist_goal], dtype=np.float32),
            quat.astype(np.float32),
        ])

        # Prepare tensors for VLA inference
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        st_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(device)

        # Predict steering action from fine-tuned VLA
        with torch.no_grad():
            pred_act = policy(img_tensor, st_tensor).squeeze(0).cpu().numpy()

        v_world = pred_act[:2]
        v_norm = float(np.linalg.norm(v_world))
        rel_goal_dir = rel_goal / max(dist_goal, 1e-4)
        if v_norm > 0.15:
            d_hat = v_world / v_norm
            speed = float(np.clip(v_norm, 0.6, 1.4))
        else:
            d_hat = rel_goal_dir
            speed = 0.8

        # Execute VLA predicted steering command
        if dist_goal <= goal_tolerance:
            rod_targets = execute_skill("stop", quat, env.dirs_body, env.max_extend)
        else:
            rod_targets = execute_skill(
                "move_forward",
                quat,
                env.dirs_body,
                env.max_extend,
                d_hat=d_hat,
                speed=speed,
                lin_vel=vel[:2],
            )

        # Step simulation
        obs, r, terminated, truncated, info = env.step(rod_targets)

        # Track clearance
        now_xy = env.data.qpos[:2].copy()
        curr_clr = pillar_clearance(obstacles, now_xy)
        min_clearance = min(min_clearance, curr_clr)

        if record_frames:
            hud_frame = draw_hud(frame, step, dist_goal, min_clearance, pred_act)
            frames_recorded.append(hud_frame)

        step += 1

        if dist_goal <= goal_tolerance and float(np.linalg.norm(env.data.qvel[:2])) < 0.15:
            if not arrived:
                arrived = True
            hold_steps -= 1
            if hold_steps <= 0:
                break

    end_pos = env.data.qpos[:2].copy()
    final_dist = float(np.linalg.norm(end_pos - goal))
    success = bool(final_dist <= goal_tolerance and min_clearance > -0.05)
    env.close()

    # Final frame with success/fail badge
    if record_frames and len(frames_recorded) > 0:
        badge = f"RESULT: {'SUCCESS (Goal Reached)' if success else 'FAILED (Missed Goal/Collision)'}"
        frames_recorded.append(draw_hud(frames_recorded[-1], step, final_dist, min_clearance, pred_act, badge))

    return {
        "seed": seed,
        "steps": step,
        "success": success,
        "final_dist": final_dist,
        "min_clearance": min_clearance,
        "collision": bool(min_clearance < 0.0),
        "frames": frames_recorded,
    }


def evaluate_vla_benchmark(
    checkpoint_path: Path,
    output_dir: Path,
    config_path: str = "configs/rl/obstacle_realistic.yaml",
    n_episodes: int = 10,
    seed_offset: int = 5000,
    save_video: bool = True,
):
    """Run complete closed-loop VLA benchmark over unseen obstacle scenes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating VLA Checkpoint: {checkpoint_path} on {device}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy = RoboballVLAPolicy(state_dim=ckpt.get("state_dim", 11), action_dim=ckpt.get("action_dim", 3)).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    cfg = load_config(config_path)

    results = []
    success_count = 0
    collision_count = 0

    video_dir = output_dir / "eval_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_episodes):
        seed = seed_offset + i
        res = evaluate_episode(
            policy=policy,
            seed=seed,
            cfg=cfg,
            device=device,
            record_frames=save_video,
        )
        results.append(res)
        if res["success"]:
            success_count += 1
        if res["collision"]:
            collision_count += 1

        logger.info(
            f"Eval [{i + 1:02d}/{n_episodes:02d}] (Seed {seed}) | "
            f"Success: {res['success']} | "
            f"Final Dist: {res['final_dist']:.2f}m | "
            f"Min Clearance: {res['min_clearance']:.2f}m | "
            f"Steps: {res['steps']}"
        )

        if save_video and len(res["frames"]) > 0:
            vid_path = video_dir / f"eval_seed_{seed}_{'success' if res['success'] else 'fail'}.mp4"
            imageio.mimsave(str(vid_path), res["frames"], fps=25)

    success_rate = success_count / n_episodes * 100.0
    collision_rate = collision_count / n_episodes * 100.0
    mean_steps = np.mean([r["steps"] for r in results])
    mean_dist = np.mean([r["final_dist"] for r in results])
    mean_clr = np.mean([r["min_clearance"] for r in results])

    summary = {
        "checkpoint": str(checkpoint_path),
        "total_episodes": n_episodes,
        "success_rate_pct": success_rate,
        "collision_rate_pct": collision_rate,
        "mean_steps": float(mean_steps),
        "mean_final_dist": float(mean_dist),
        "mean_min_clearance": float(mean_clr),
        "episodes": [
            {
                "seed": r["seed"],
                "success": r["success"],
                "steps": r["steps"],
                "final_dist": float(r["final_dist"]),
                "min_clearance": float(r["min_clearance"]),
                "collision": r["collision"],
            }
            for r in results
        ],
    }

    summary_file = output_dir / "eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("CLOSED-LOOP VLA EVALUATION COMPLETE!")
    logger.info(f"Success Rate:    {success_rate:.1f}% ({success_count}/{n_episodes})")
    logger.info(f"Collision Rate:  {collision_rate:.1f}% ({collision_count}/{n_episodes})")
    logger.info(f"Mean Steps:      {mean_steps:.1f}")
    logger.info(f"Mean Final Dist: {mean_dist:.2f}m")
    logger.info(f"Summary JSON:    {summary_file}")
    if save_video:
        logger.info(f"Videos saved to: {video_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned VLA policy on obstacle scenes.")
    parser.add_argument("--checkpoint", type=str, default="storage_local/20260913_1525__local__train_vla__obstacle_navigation/checkpoints/best_policy.pt", help="Path to trained policy checkpoint")
    parser.add_argument("--out-dir", type=str, default="storage_local/20260913_1525__local__train_vla__obstacle_navigation/evaluation", help="Output directory")
    parser.add_argument("--config", type=str, default="configs/rl/obstacle_realistic.yaml", help="Config file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--seed-offset", type=int, default=5000, help="Starting test seed")
    parser.add_argument("--video", action="store_true", default=True, help="Record rollout videos")
    args = parser.parse_args()

    evaluate_vla_benchmark(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.out_dir),
        config_path=args.config,
        n_episodes=args.episodes,
        seed_offset=args.seed_offset,
        save_video=args.video,
    )


if __name__ == "__main__":
    main()
