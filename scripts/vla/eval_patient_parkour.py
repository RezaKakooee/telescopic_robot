"""Evaluate Patient Context-Aware Parkour Skill Arbitration (Option B).

Implements context-aware, patient high-level arbitration:
- Uses existing handcrafted skills ('move', 'jump_forward_while_moving', 'traverse_rough_terrain').
- Navigates smoothly on flat corridors and straightaways without jumping.
- Triggers targeted leaps strictly at physical obstacles:
    1. Station 2: 0.18m Hurdle
    2. Station 3: Platform step-up onto Box 1
    3. Station 3: Deep Valleys (Chasms 1 & 2)
    4. Station 4: 3-Step Stairs to Terrace Deck
- Navigates compliant rough terrain across the cobblestone bed.
- Records 25 FPS real-time and 15 FPS slow-motion videos with continuous physics capture and HUD telemetry.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import imageio
import numpy as np

from radial_sphere.config import load_config_cli
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from radial_sphere.handcrafted_skill_backend import SKILL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_patient_parkour")


def select_patient_skill(pos: np.ndarray, guidance: np.ndarray) -> tuple[str, str]:
    """Determine skill and descriptive navigational status from spatial context."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

    # 1. Leg 1: (0, 0) -> (9.0, 0.0) heading East (+X)
    if y < 1.0 and x < 8.0:
        if 3.4 <= x <= 4.6:
            return "jump_forward_while_moving", "Station 2: Hurdle Leap (0.18m)"
        return "move", "Station 1: Patient Street Cruising"

    # Turn 1: Corner 1 near (9.0, 0.0) turning North (+Y)
    if x >= 8.0 and y < 2.0:
        return "move", "Turn 1: Rounding Corner North"

    # 2. Leg 2: x ~ 9.0, moving North (+Y)
    if x > 7.5 and 2.0 <= y < 8.5:
        if 1.9 <= y <= 2.3:
            return "jump_forward_while_moving", "Station 3: Platform Step-up (Box 1)"
        if 3.0 <= y <= 3.6:
            return "jump_forward_while_moving", "Station 3: Valley 1 Chasm Leap"
        if 4.7 <= y <= 5.3:
            return "jump_forward_while_moving", "Station 3: Valley 2 Chasm Leap"
        return "move", "Station 3: Cruising Parkour Deck"

    # Turn 2: Corner 2 near (9.0, 9.0) turning West (-X)
    if x > 7.2 and y >= 8.0:
        return "move", "Turn 2: Rounding Corner West"

    # 3. Leg 3: y ~ 9.0, moving West (-X)
    if y > 7.5 and 1.5 < x <= 7.2:
        if 6.3 <= x <= 7.2:
            return "jump_forward_while_moving", "Station 4: Stair Ascent (3 Steps)"
        return "move", "Station 4: Terrace Deck & Ramp"

    # Turn 3: Corner 3 near (0.0, 9.0) turning North (+Y)
    if x <= 1.5 and y < 10.5:
        return "move", "Turn 3: Rounding Corner North"

    # 4. Leg 4: x around 0.0, moving North (+Y)
    if 10.5 <= y < 12.4:
        return "traverse_rough_terrain", "Station 5: Rough Cobblestones"
    if 12.4 <= y <= 13.0:
        return "jump_forward_while_moving", "Station 5: Curb Vault into Conduit"
    if 13.0 < y < 14.4:
        return "move", "Station 6: Low-Clearance Conduit"
    if 14.4 <= y <= 15.2:
        return "jump_forward_while_moving", "Station 6: Ring Barrier Vault to Goal"

    # Final straightaway to Station 7
    return "move", "Station 7: Terminal Goal Approach"


def render_vla_frame(env, width: int = 384, height: int = 384) -> np.ndarray:
    """Render 3D tracking chase camera up-close."""
    import mujoco
    core_env = env.env if hasattr(env, "env") else env
    if core_env.renderer is None or core_env.renderer.height != height or core_env.renderer.width != width:
        core_env.renderer = mujoco.Renderer(core_env.model, height=height, width=width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = core_env.core_body_id
    cam.distance = 3.2
    cam.elevation = -26.0
    cam.azimuth = 135.0
    core_env.renderer.update_scene(core_env.data, camera=cam)
    return core_env.renderer.render()



def draw_hud(
    frame: np.ndarray,
    step: int,
    pos: tuple[float, float],
    goal_dist: float,
    skill_name: str,
    status_desc: str,
    badge: str | None = None,
) -> np.ndarray:
    img = frame.copy()
    h, w, _ = img.shape

    # Top overlay bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 54), (15, 20, 30), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    # Telemetry line 1
    t1 = f"Step {step:03d} | Pos: ({pos[0]:.1f}, {pos[1]:.1f}) | Goal Dist: {goal_dist:.2f}m"
    cv2.putText(img, t1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Telemetry line 2 (Skill & Context)
    is_jump = "jump" in skill_name
    skill_color = (80, 200, 255) if is_jump else ((80, 255, 140) if skill_name == "move" else (255, 200, 80))
    t2 = f"Skill: {skill_name} [{status_desc}]"
    cv2.putText(img, t2, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, skill_color, 1, cv2.LINE_AA)

    # Optional result badge
    if badge:
        b_overlay = img.copy()
        cv2.rectangle(b_overlay, (0, h - 38), (w, h), (10, 40, 15) if "SUCCESS" in badge else (40, 10, 10), -1)
        cv2.addWeighted(b_overlay, 0.90, img, 0.10, 0, img)
        cv2.putText(img, badge, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def evaluate_patient_episode(
    cfg,
    seed: int = 100,
    max_macro_steps: int = 320,
    record_frames: bool = True,
    fps: int = 25,
) -> dict:
    scenario = generate_scenario("playground", cfg, seed=seed)
    env = SkillArbitrationEnv(cfg, scenario=scenario, seed=seed, max_steps=1500)
    obs, info = env.reset(seed=seed)

    goal = np.asarray(env.scenario.goal, dtype=np.float32)[:2]
    frames_recorded = []
    skill_history = []
    step = 0
    terminated = truncated = False
    next_frame_time = 0.0
    current_skill = "move"
    current_status = "Station 1: Patient Street Cruising"

    def record_step(current):
        nonlocal next_frame_time
        if not record_frames:
            return
        now = float(current.env.data.time)
        while now + 1e-9 >= next_frame_time:
            cpos = current.env.data.qpos[:2]
            cdist = float(np.linalg.norm(goal - cpos))
            cframe = render_vla_frame(current.env, width=384, height=384)
            hud = draw_hud(cframe, step, (float(cpos[0]), float(cpos[1])), cdist, current_skill, current_status)
            frames_recorded.append(hud)
            next_frame_time += 1.0 / fps

    if record_frames:
        env.on_control_step = record_step

    counts = {"move": 0, "jump": 0, "rough": 0, "other": 0}

    while not (terminated or truncated) and step < max_macro_steps:
        pos = env.env.data.qpos[:3].copy()
        guidance = env.waypoint_tracker.get_guidance(pos)
        current_skill, current_status = select_patient_skill(pos, guidance)
        skill_history.append(current_skill)

        if "jump" in current_skill:
            counts["jump"] += 1
        elif current_skill == "traverse_rough_terrain":
            counts["rough"] += 1
        elif current_skill == "move":
            counts["move"] += 1
        else:
            counts["other"] += 1

        act = np.full(len(SKILL_NAMES), -1.0, dtype=np.float32)
        act[SKILL_NAMES.index(current_skill)] = 1.0

        obs, r, terminated, truncated, info = env.step(act)
        step += 1

        cur_dist = float(np.linalg.norm(env.env.data.qpos[:2] - goal))
        if cur_dist < 0.45:
            break

    env.on_control_step = None
    final_pos = env.env.data.qpos[:2].copy()
    final_dist = float(np.linalg.norm(final_pos - goal))
    success = bool(info.get("success", False) or final_dist < 0.50)
    env.close()

    if record_frames and frames_recorded:
        badge = f"RESULT: {'SUCCESS (Course Completed - Patient & Smooth)' if success else f'FINAL DIST: {final_dist:.2f}m'}"
        final_hud = draw_hud(frames_recorded[-1], step, (float(final_pos[0]), float(final_pos[1])), final_dist, current_skill, current_status, badge)
        for _ in range(35):
            frames_recorded.append(final_hud)

    return {
        "seed": seed,
        "steps": step,
        "success": success,
        "final_pos": [float(final_pos[0]), float(final_pos[1])],
        "final_dist": final_dist,
        "counts": counts,
        "jump_ratio_pct": float(counts["jump"] / max(1, step) * 100.0),
        "move_ratio_pct": float(counts["move"] / max(1, step) * 100.0),
        "skill_history": skill_history,
        "frames": frames_recorded,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Patient Context-Aware Parkour Skill Arbitration")
    parser.add_argument("--config-name", default="playground_parkour_skills")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=100)
    parser.add_argument("--out-dir", default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/evaluation/patient_arbiter")
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    video_dir = output_dir / "eval_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config_cli(name=args.config_name)
    logger.info(f"Loaded config: {args.config_name}")
    logger.info(f"Evaluating {args.episodes} episodes across seeds {args.seed_offset} to {args.seed_offset + args.episodes - 1}...")

    results = []
    for i in range(args.episodes):
        seed = args.seed_offset + i
        res = evaluate_patient_episode(cfg, seed=seed, record_frames=True, fps=25)
        results.append(res)

        logger.info(
            f"Ep [{i + 1:02d}/{args.episodes:02d}] Seed {seed} | "
            f"Success: {res['success']} | "
            f"Dist: {res['final_dist']:.2f}m | "
            f"Steps: {res['steps']} | "
            f"Jumps: {res['counts']['jump']} ({res['jump_ratio_pct']:.1f}%) | "
            f"Moves: {res['counts']['move']} ({res['move_ratio_pct']:.1f}%)"
        )

        if res["frames"]:
            realtime_path = video_dir / f"patient_parkour_seed_{seed}_{'success' if res['success'] else 'fail'}.mp4"
            imageio.mimsave(str(realtime_path), res["frames"], fps=25)

            slowmo_path = video_dir / f"patient_parkour_seed_{seed}_slowmo.mp4"
            imageio.mimsave(str(slowmo_path), res["frames"], fps=15)

            logger.info(f"Saved real-time video (25 fps, {len(res['frames'])} frames): {realtime_path}")
            logger.info(f"Saved slow-mo video (15 fps): {slowmo_path}")

    # Summary
    success_count = sum(1 for r in results if r["success"])
    mean_dist = float(np.mean([r["final_dist"] for r in results]))
    mean_steps = float(np.mean([r["steps"] for r in results]))
    mean_jumps = float(np.mean([r["counts"]["jump"] for r in results]))
    mean_moves = float(np.mean([r["counts"]["move"] for r in results]))
    mean_jump_ratio = float(np.mean([r["jump_ratio_pct"] for r in results]))

    summary = {
        "total_episodes": args.episodes,
        "success_rate_pct": float(success_count / args.episodes * 100.0),
        "mean_steps": mean_steps,
        "mean_final_dist": mean_dist,
        "mean_jumps": mean_jumps,
        "mean_moves": mean_moves,
        "mean_jump_ratio_pct": mean_jump_ratio,
        "episodes": [
            {
                "seed": r["seed"],
                "success": r["success"],
                "steps": r["steps"],
                "final_pos": r["final_pos"],
                "final_dist": r["final_dist"],
                "counts": r["counts"],
                "jump_ratio_pct": r["jump_ratio_pct"],
                "move_ratio_pct": r["move_ratio_pct"],
            }
            for r in results
        ],
    }

    summary_file = output_dir / "patient_eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("PATIENT ARBITRATION BENCHMARK COMPLETE!")
    logger.info(f"Success Rate:    {summary['success_rate_pct']:.1f}% ({success_count}/{args.episodes})")
    logger.info(f"Mean Final Dist: {mean_dist:.2f}m")
    logger.info(f"Mean Steps:      {mean_steps:.1f}")
    logger.info(f"Mean Jumps:      {mean_jumps:.1f} ({mean_jump_ratio:.1f}% of steps)")
    logger.info(f"Mean Moves:      {mean_moves:.1f} ({100.0 - mean_jump_ratio:.1f}% of steps)")
    logger.info(f"Summary saved:   {summary_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
