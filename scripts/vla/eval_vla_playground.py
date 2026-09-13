"""Closed-Loop Evaluation of Fine-Tuned Parkour VLA on the 3D Playground Parkour Task.

Evaluates the multimodal VLA policy on the multi-station parkour course:
- Perception: Visual RGB camera frame + robot state
- Policy: ParkourVLAPolicy predicts macro skill selection (10 classes)
- Execution: Low-level controller executes the predicted parkour skill
- Telemetry: Real-time HUD showing station, active skill, distance to goal
- Output: Benchmark metrics JSON + MP4 video rollouts
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mujoco

from radial_sphere.config import load_config_cli
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from scripts.data.generate_obstacle_vla_dataset import render_vla_frame
from scripts.vla.train_parkour_vla import ParkourVLAPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_parkour_vla")

SKILL_NAMES = (
    "move", "stop", "reverse", "follow_path", "straddle_gap",
    "traverse_rough_terrain", "jump_up", "jump_forward_while_stopped",
    "jump_forward_while_moving", "jump_to",
)


def draw_hud(
    frame: np.ndarray,
    step: int,
    pos: tuple[float, float],
    dist_goal: float,
    skill_name: str,
    status_text: str = "",
) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Top banner
    draw.rectangle([(0, 0), (img.width, 36)], fill=(15, 23, 42))
    text = (
        f"Step {step:02d} | Pos: ({pos[0]:4.1f}, {pos[1]:4.1f}) | "
        f"Goal: {dist_goal:4.1f}m | "
        f"Skill: {skill_name}"
    )
    draw.text((8, 10), text, fill=(241, 245, 249))

    if status_text:
        col = (34, 197, 94) if "SUCCESS" in status_text else (239, 68, 68)
        draw.rectangle([(0, img.height - 30), (img.width, img.height)], fill=(15, 23, 42))
        draw.text((10, img.height - 24), status_text, fill=col)

    return np.asarray(img)


def evaluate_parkour_episode(
    policy: ParkourVLAPolicy,
    cfg,
    device: torch.device,
    seed: int = 100,
    max_steps: int = 50,
    record_frames: bool = True,
    fps: int = 25,
) -> dict:
    scenario = generate_scenario("playground", cfg, seed=seed)
    env = SkillArbitrationEnv(cfg, scenario=scenario, seed=seed, max_steps=1500)
    obs, info = env.reset(seed=seed)

    goal = np.asarray(scenario.goal, dtype=np.float32)[:2]
    step = 0
    frames_recorded = []
    skill_history = []
    terminated = truncated = False

    current_skill = "start"
    next_frame_time = float(env.env.data.time)

    def record_step(current):
        nonlocal next_frame_time
        if not record_frames:
            return
        now = float(current.env.data.time)
        while now + 1e-9 >= next_frame_time:
            cpos = current.env.data.qpos[:2]
            cdist = float(np.linalg.norm(goal - cpos))
            cframe = render_vla_frame(current.env, width=384, height=384)
            hud = draw_hud(cframe, step, (float(cpos[0]), float(cpos[1])), cdist, current_skill)
            frames_recorded.append(hud)
            next_frame_time += 1.0 / fps

    if record_frames:
        env.on_control_step = record_step

    while not (terminated or truncated) and step < max_steps:
        pos = env.env.data.qpos[:3].copy()
        quat = env.env.data.qpos[3:7].copy()
        vel = env.env.data.qvel[:3].copy()
        ball_xy = pos[:2]

        rel_goal = goal - ball_xy
        dist_goal = float(np.linalg.norm(rel_goal))

        # Model takes 256x256 frame for visual encoding
        frame = render_vla_frame(env.env, width=256, height=256)

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

        skill_idx = int(np.argmax(pred_act[:len(SKILL_NAMES)]))
        current_skill = SKILL_NAMES[skill_idx]
        skill_history.append(current_skill)

        # Step environment (invokes on_control_step for all intermediate physics steps)
        obs, r, terminated, truncated, info = env.step(pred_act)
        step += 1

    env.on_control_step = None
    success = bool(info.get("success", False))
    final_pos = env.env.data.qpos[:2].copy()
    final_dist = float(np.linalg.norm(final_pos - goal))
    env.close()

    # Hold the final result frame for 30 frames (1.2s)
    if record_frames and frames_recorded:
        badge = f"RESULT: {'SUCCESS (Course Completed)' if success else f'FINAL DIST: {final_dist:.2f}m'}"
        final_hud = draw_hud(frames_recorded[-1], step, (float(final_pos[0]), float(final_pos[1])), final_dist, current_skill, badge)
        for _ in range(30):
            frames_recorded.append(final_hud)

    return {
        "seed": seed,
        "steps": step,
        "success": success,
        "final_pos": [float(final_pos[0]), float(final_pos[1])],
        "final_dist": final_dist,
        "skill_history": skill_history,
        "frames": frames_recorded,
    }


def evaluate_parkour_benchmark(
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    n_episodes: int = 5,
    seed_offset: int = 100,
    save_video: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating Parkour VLA Checkpoint: {checkpoint_path} on {device}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy = ParkourVLAPolicy(state_dim=ckpt.get("state_dim", 11), action_dim=ckpt.get("action_dim", 10)).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    cfg = load_config_cli(path=str(config_path))

    results = []
    success_count = 0
    video_dir = output_dir / "eval_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_episodes):
        seed = seed_offset + i
        res = evaluate_parkour_episode(
            policy=policy,
            cfg=cfg,
            device=device,
            seed=seed,
            record_frames=save_video,
        )
        results.append(res)
        if res["success"]:
            success_count += 1

        logger.info(
            f"Eval [{i + 1:02d}/{n_episodes:02d}] (Seed {seed}) | "
            f"Success: {res['success']} | "
            f"Final Dist: {res['final_dist']:.2f}m | "
            f"Steps: {res['steps']}"
        )

        if save_video and res["frames"]:
            vid_path = video_dir / f"parkour_vla_seed_{seed}_{'success' if res['success'] else 'fail'}.mp4"
            imageio.mimsave(str(vid_path), res["frames"], fps=25)
            # Also save slow-motion version at 15 fps
            vid_slow = video_dir / f"parkour_vla_seed_{seed}_slowmo.mp4"
            imageio.mimsave(str(vid_slow), res["frames"], fps=15)
            logger.info(f"Saved real-time video (25 fps, {len(res['frames'])} frames): {vid_path}")
            logger.info(f"Saved slow-mo video (15 fps): {vid_slow}")

    success_rate = (success_count / n_episodes) * 100.0
    mean_dist = float(np.mean([r["final_dist"] for r in results]))
    mean_steps = float(np.mean([r["steps"] for r in results]))

    summary = {
        "checkpoint": str(checkpoint_path),
        "total_episodes": n_episodes,
        "success_rate_pct": success_rate,
        "mean_steps": mean_steps,
        "mean_final_dist": mean_dist,
        "episodes": [
            {
                "seed": r["seed"],
                "success": r["success"],
                "steps": r["steps"],
                "final_pos": r["final_pos"],
                "final_dist": r["final_dist"],
                "skills_dispatched": r["skill_history"][:10],
            }
            for r in results
        ],
    }

    summary_file = output_dir / "parkour_vla_eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("PARKOUR VLA EVALUATION BENCHMARK COMPLETE!")
    logger.info(f"Success Rate:    {success_rate:.1f}% ({success_count}/{n_episodes})")
    logger.info(f"Mean Final Dist: {mean_dist:.2f}m")
    logger.info(f"Mean Steps:      {mean_steps:.1f}")
    logger.info(f"Summary JSON:    {summary_file}")
    if save_video:
        logger.info(f"Videos saved to: {video_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Parkour VLA policy on 3D playground course.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/checkpoints/best_parkour_policy.pt",
        help="Path to trained Parkour VLA checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/code/config.yaml",
        help="Path to playground config",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/evaluation/vla",
        help="Output evaluation directory",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed-offset", type=int, default=100, help="Starting seed")
    parser.add_argument("--video", action="store_true", default=True, help="Save MP4 rollout videos")
    args = parser.parse_args()

    evaluate_parkour_benchmark(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        output_dir=Path(args.out_dir),
        n_episodes=args.episodes,
        seed_offset=args.seed_offset,
        save_video=args.video,
    )


if __name__ == "__main__":
    main()
