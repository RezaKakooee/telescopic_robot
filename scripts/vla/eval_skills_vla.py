"""Closed-Loop Evaluation of Fine-Tuned Skills VLA Policy on 3D Playground Parkour.

Evaluates the multi-task multimodal VLA policy directly on the 3D playground course.
Renders broadcast-quality 640x480 chase footage with Picture-in-Picture (PiP)
onboard VLA perception view and rich real-time HUD telemetry.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import cv2
import imageio
import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from radial_sphere import playground_course as PC
from radial_sphere.config import load_config_cli
from radial_sphere.run_id import build_run_id
from radial_sphere.snapshot import make_run_dir
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from scripts.data.generate_obstacle_vla_dataset import render_vla_frame
from scripts.vla.train_skills_vla import SkillsVLAPolicy
from skills_vla import ENV_SKILL_MAP, SKILL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_skills_vla")


def get_station_context(pos: np.ndarray) -> str:
    """Identify the current parkour station based on ball coordinates."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if y < 1.0 and x < 8.0:
        if PC.HURDLE_X - 1.3 <= x <= PC.HURDLE_X + 0.3:
            return f"Station 2: Hurdle Leap ({PC.HURDLE_H:.2f}m)"
        return "Station 1: Street Cruising"
    if x >= 8.0 and y < 2.0:
        return "Turn 1: Rounding Corner North"
    if x > 7.5 and 2.0 <= y < 8.5:
        if PC.BOX_Y0[0] - 0.4 <= y <= PC.BOX_Y0[0] + 0.3:
            return "Station 3: Platform Box 1 Ascent"
        if PC.BOX_Y1[0] - 0.4 <= y <= PC.BOX_Y0[1]:
            return "Station 3: Valley 1 Chasm Leap"
        if PC.BOX_Y1[1] - 0.4 <= y <= PC.BOX_Y0[2]:
            return "Station 3: Valley 2 Chasm Leap"
        return "Station 3: Elevated Parkour Deck"
    if x > 7.2 and y >= 8.0:
        return "Turn 2: Rounding Corner West"
    if y > 7.5 and 1.5 < x <= 7.2:
        if PC.DECK_X1 <= x <= PC.STAIR_X0 + 0.3:
            return f"Station 4: Stair Ascent ({PC.STAIR_N} Steps, {PC.STAIR_TOP:.2f}m)"
        return "Station 4: Terrace Deck & Ramp"
    if x <= 1.5 and y < 10.5:
        return "Turn 3: Rounding Corner North"
    if 10.5 <= y < 12.4:
        return "Station 5: Rough Cobblestone Bed"
    if 12.4 <= y < PC.WALL_Y - 1.0:
        return "Station 5: Curb Exit"
    if PC.WALL_Y - 1.0 <= y <= PC.WALL_Y + 0.8:
        return f"Station 6: Jump Wall ({PC.WALL_H:.2f}m)"
    return "Station 7: Terminal Goal Pad"


def render_chase_broadcast(env, renderer: mujoco.Renderer) -> np.ndarray:
    """Render smooth 640x480 path-anchored tracking chase camera."""
    core_env = env.env if hasattr(env, "env") else env
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = core_env.core_body_id
    cam.distance = 2.4
    cam.elevation = -24.0

    pos = core_env.data.qpos[:2]
    v = core_env.data.qvel[:2]

    # Align camera yaw with path direction
    if hasattr(env, "scenario") and hasattr(env.scenario, "path_pts") and len(env.scenario.path_pts) > 1:
        dists = np.linalg.norm(env.scenario.path_pts - pos, axis=1)
        idx = int(np.argmin(dists))
        lookahead_idx = min(idx + 10, len(env.scenario.path_pts) - 1)
        fwd = env.scenario.path_pts[lookahead_idx] - pos
        if np.linalg.norm(fwd) < 0.2:
            fwd = env.scenario.path_pts[-1] - env.scenario.path_pts[-2]
        path_yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    else:
        path_yaw = 0.0

    if not hasattr(core_env, "_cam_azimuth_smooth") or core_env._cam_azimuth_smooth is None:
        core_env._cam_azimuth_smooth = path_yaw
    else:
        delta = (path_yaw - core_env._cam_azimuth_smooth + 180.0) % 360.0 - 180.0
        core_env._cam_azimuth_smooth += 0.12 * delta

    cam.azimuth = core_env._cam_azimuth_smooth
    renderer.update_scene(core_env.data, camera=cam)
    return renderer.render()


def compose_telemetry_hud(
    main_frame: np.ndarray,
    vla_frame: np.ndarray,
    step: int,
    sim_time: float,
    pos: np.ndarray,
    vel: np.ndarray,
    goal_dist: float,
    skill_name: str,
    confidence: float,
    pred_params: np.ndarray,
    station: str,
    badge: str | None = None,
) -> np.ndarray:
    """Compose broadcast HUD overlay with PiP VLA camera and telemetry banner."""
    canvas = main_frame.copy()
    h, w, _ = canvas.shape

    # 1. Top HUD Bar
    top_bar = canvas.copy()
    cv2.rectangle(top_bar, (0, 0), (w, 64), (12, 16, 24), -1)
    cv2.addWeighted(top_bar, 0.85, canvas, 0.15, 0, canvas)

    # Telemetry Line 1: Mission Status & Coordinates
    speed = float(np.linalg.norm(vel[:2]))
    t1 = f"Step: {step:03d} | t: {sim_time:5.2f}s | Pos: ({pos[0]:5.2f}, {pos[1]:5.2f}) | Spd: {speed:4.2f}m/s | Goal Dist: {goal_dist:5.2f}m"
    cv2.putText(canvas, t1, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 235, 245), 1, cv2.LINE_AA)

    # Telemetry Line 2: Active Station & VLA Decision
    is_jump = "jump" in skill_name
    skill_col = (50, 220, 255) if is_jump else ((50, 255, 140) if skill_name == "roll" else (255, 180, 50))
    t2 = f"Station: {station}"
    t3 = f"VLA Skill: {skill_name.upper()} ({confidence*100:.1f}%) | Params: [{pred_params[0]:.2f}, {pred_params[1]:.2f}]"
    cv2.putText(canvas, t2, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, t3, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, skill_col, 1, cv2.LINE_AA)

    # 2. Picture-in-Picture (PiP): Onboard VLA Perception (256x256 -> 150x150)
    pip_sz = 140
    pip_img = cv2.resize(vla_frame, (pip_sz, pip_sz), interpolation=cv2.INTER_AREA)
    # Border
    cv2.rectangle(pip_img, (0, 0), (pip_sz - 1, pip_sz - 1), (80, 180, 240), 2)
    cv2.putText(pip_img, "VLA INPUT", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    pip_x = w - pip_sz - 12
    pip_y = 12
    canvas[pip_y : pip_y + pip_sz, pip_x : pip_x + pip_sz] = pip_img

    # 3. Optional Bottom Success/Fail Badge
    if badge:
        bot_bar = canvas.copy()
        cv2.rectangle(bot_bar, (0, h - 42), (w, h), (10, 45, 20) if "SUCCESS" in badge else (45, 10, 10), -1)
        cv2.addWeighted(bot_bar, 0.90, canvas, 0.10, 0, canvas)
        cv2.putText(canvas, badge, (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def evaluate_skills_vla(
    checkpoint_path: Path,
    config_name: str = "playground_parkour_skills",
    output_dir: Path | None = None,
    episodes: int = 5,
    seed_offset: int = 42,
    max_macro_steps: int = 300,
    fps: int = 25,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if output_dir is None:
        output_dir = make_run_dir(build_run_id("eval_skills_vla", tag=config_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    snapshot_dir = output_dir / "snapshots"
    video_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)
    logger.info(f"Evaluating Skills VLA on device: {device}")

    # 1. Load Model
    model = SkillsVLAPolicy().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # RL checkpoints carry an extra value head that the BC policy does not have.
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("value_head.")}
    model.load_state_dict(sd)
    model.eval()
    logger.info(f"Loaded Skills VLA policy from {checkpoint_path} (epoch={ckpt.get('epoch')}, step={ckpt.get('step')}, val_acc={ckpt.get('val_acc')})")

    # 2. Setup Environment
    cfg = load_config_cli(name=config_name)

    results = []

    for ep_idx in range(episodes):
        seed = seed_offset + ep_idx
        scenario = generate_scenario("playground", cfg, seed=seed)
        env = SkillArbitrationEnv(cfg, scenario=scenario, seed=seed, max_steps=1500)
        obs, info = env.reset(seed=seed)

        goal = np.asarray(env.scenario.goal, dtype=np.float32)[:2]

        chase_renderer = mujoco.Renderer(env.env.model, height=480, width=640)
        vla_renderer = mujoco.Renderer(env.env.model, height=256, width=256)

        frames_recorded = []
        snapshots_saved = set()
        step = 0
        terminated = truncated = False
        next_frame_time = 0.0

        current_skill_name = "roll"
        current_confidence = 1.0
        current_params = np.zeros(4, dtype=np.float32)
        current_station = "Station 1: Street Cruising"
        hits_by_station: dict[str, int] = {}
        hit_geoms_seen: set[str] = set()

        def record_control_step(current_env):
            nonlocal next_frame_time
            now = float(current_env.env.data.time)
            while now + 1e-9 >= next_frame_time:
                cpos = current_env.env.data.qpos[:3].copy()
                cvel = current_env.env.data.qvel[:3].copy()
                cdist = float(np.linalg.norm(goal - cpos[:2]))
                station = get_station_context(cpos)

                main_img = render_chase_broadcast(current_env, chase_renderer)
                vla_img = render_vla_frame(current_env.env, width=256, height=256)

                hud = compose_telemetry_hud(
                    main_frame=main_img,
                    vla_frame=vla_img,
                    step=step,
                    sim_time=now,
                    pos=cpos,
                    vel=cvel,
                    goal_dist=cdist,
                    skill_name=current_skill_name,
                    confidence=current_confidence,
                    pred_params=current_params,
                    station=station,
                )
                frames_recorded.append(hud)

                # Capture representative milestone snapshots
                for key_station in ["Station 2", "Station 3", "Station 4", "Station 5", "Station 6", "Station 7"]:
                    if key_station in station and key_station not in snapshots_saved:
                        snap_file = snapshot_dir / f"eval_seed_{seed}_{key_station.lower().replace(' ', '_')}.png"
                        imageio.imwrite(str(snap_file), hud)
                        snapshots_saved.add(key_station)

                next_frame_time += 1.0 / fps

        env.on_control_step = record_control_step

        while not (terminated or truncated) and step < max_macro_steps:
            # 1. Perception frame
            vla_frame = render_vla_frame(env.env, width=256, height=256)

            # 2. Proprioception state
            pos = env.env.data.qpos[:3].copy()
            quat = env.env.data.qpos[3:7].copy()
            vel = env.env.data.qvel[:3].copy()
            ball_xy = pos[:2]
            rel_goal = goal - ball_xy
            dist_goal = float(np.linalg.norm(rel_goal))

            state_vec = np.concatenate([
                ball_xy.astype(np.float32),
                vel[:2].astype(np.float32),
                rel_goal.astype(np.float32),
                np.array([dist_goal], dtype=np.float32),
                quat.astype(np.float32),
            ])

            # 3. Policy Forward Pass
            with torch.no_grad():
                img_t = torch.from_numpy(vla_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                st_t = torch.from_numpy(state_vec).unsqueeze(0).float().to(device)
                logits, pred_params = model(img_t, st_t)
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
                skill_idx = int(logits.argmax(dim=-1).item())
                current_skill_name = SKILL_NAMES[skill_idx]
                current_confidence = float(probs[skill_idx])
                current_params = pred_params[0].cpu().numpy()
                current_station = get_station_context(pos)

            # 4. Environment Step
            env_skill = ENV_SKILL_MAP[current_skill_name]
            act = np.full(len(env.skill_names), -1.0, dtype=np.float32)
            act[env.skill_names.index(env_skill)] = 1.0

            obs, reward, terminated, truncated, info = env.step(act)
            step += 1
            if info.get("obstacle_hit"):
                st = get_station_context(env.env.data.qpos[:3])
                hits_by_station[st] = hits_by_station.get(st, 0) + 1
                hit_geoms_seen.update(info.get("obstacle_hit_geoms", ()))

        final_pos = env.env.data.qpos[:2].copy()
        final_dist = float(np.linalg.norm(final_pos - goal))
        success = bool(info.get("success", False) or final_dist < 0.60)
        n_hits = int(sum(hits_by_station.values()))
        # Clean success: reached the goal and never pushed into an obstacle face.
        clean = success and n_hits == 0

        # Add closing freeze banner
        badge = f"MISSION {'SUCCESS' if success else 'TERMINATED'} | Final Dist: {final_dist:.2f}m | Macro Steps: {step}"
        for _ in range(25):  # 1 second freeze frame
            if frames_recorded:
                end_frame = frames_recorded[-1].copy()
                h, w, _ = end_frame.shape
                cv2.rectangle(end_frame, (0, h - 42), (w, h), (10, 50, 20) if success else (50, 10, 10), -1)
                cv2.putText(end_frame, badge, (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                frames_recorded.append(end_frame)

        video_path = video_dir / f"eval_skills_vla_seed_{seed}_{'success' if success else 'fail'}.mp4"
        if frames_recorded:
            imageio.mimsave(str(video_path), frames_recorded, fps=fps)

        logger.info(
            f"Ep [{ep_idx + 1:02d}/{episodes:02d}] Seed {seed} | "
            f"Success: {success} | Clean: {clean} | Hits: {n_hits} | Dist: {final_dist:.2f}m | Steps: {step} | "
            f"Video: {video_path.name}"
        )
        if hits_by_station:
            logger.info(f"    hits by station: {hits_by_station}")
            logger.info(f"    hit geoms: {sorted(hit_geoms_seen)}")

        results.append({
            "episode": ep_idx + 1,
            "seed": seed,
            "success": success,
            "clean_success": clean,
            "obstacle_hits": n_hits,
            "hits_by_station": hits_by_station,
            "hit_geoms": sorted(hit_geoms_seen),
            "final_dist": final_dist,
            "steps": step,
            "video": str(video_path),
        })

    env.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "total_episodes": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "clean_success_rate": float(np.mean([r["clean_success"] for r in results])),
        "mean_obstacle_hits": float(np.mean([r["obstacle_hits"] for r in results])),
        "mean_final_dist": float(np.mean([r["final_dist"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "results": results,
    }

    summary_file = output_dir / "eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 65)
    logger.info(f"EVALUATION COMPLETE: Success Rate = {summary['success_rate']*100:.1f}% | "
                f"Clean Success = {summary['clean_success_rate']*100:.1f}% | "
                f"Mean Hits = {summary['mean_obstacle_hits']:.1f}")
    logger.info(f"Summary JSON: {summary_file}")
    logger.info("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned Skills VLA on Parkour.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="storage_local/20260914_1317__local__train_skills_vla__v2_clean_oracle/best_policy.pt",
        help="Path to fine-tuned VLA policy checkpoint",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="playground_parkour_skills",
        help="Config preset name",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for videos and summary (default: a new timestamped run dir under storage_local/)",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to evaluate")
    parser.add_argument("--seed-offset", type=int, default=42, help="Seed offset")
    args = parser.parse_args()

    evaluate_skills_vla(
        checkpoint_path=Path(args.checkpoint),
        config_name=args.config_name,
        output_dir=Path(args.out_dir) if args.out_dir else None,
        episodes=args.episodes,
        seed_offset=args.seed_offset,
    )


if __name__ == "__main__":
    main()
