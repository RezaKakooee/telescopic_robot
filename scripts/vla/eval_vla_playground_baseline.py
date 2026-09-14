"""Zero-Shot Baseline Evaluation of Flat-Obstacle VLA Policy with Modular skills_vla.

Evaluates the pre-trained continuous-steering VLA (best_policy.pt) zero-shot on the
3D Playground Parkour task using the new modular `skills_vla` package.
Quantifies the domain gap between flat obstacle navigation and 3D terrain/hurdles.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from scripts.data.generate_obstacle_vla_dataset import render_vla_frame
from scripts.vla.train_vla import RoboballVLAPolicy
from skills_vla import RobotState, dispatch_vla_action

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_vla_skills_zero_shot")


def get_station_description(x: float, y: float) -> str:
    """Classify current course station from sphere coordinates."""
    if y < 1.2 and x < 3.4:
        return "Station 1: Flat Street Cruising"
    elif y < 1.2 and 3.4 <= x < 5.2:
        return "Station 2: 0.18m Hurdle Obstacle"
    elif y < 1.2 and 5.2 <= x < 8.2:
        return "Leg 1: Run-up to Turn 1"
    elif x >= 8.0 and y < 2.5:
        return "Turn 1: Rounding Corner North (+Y)"
    elif x > 7.5 and 2.5 <= y < 5.5:
        return "Station 3: Stepping Boxes"
    elif x > 7.5 and 5.5 <= y < 9.0:
        return "Station 3: Deep Valleys / Chasms"
    elif y >= 8.5 and x > 6.0:
        return "Turn 2: Corner West (-X)"
    elif y > 8.0 and 2.5 <= x < 6.5:
        return "Station 4: 3-Step Stairs to Terrace"
    elif y > 8.0 and x < 2.5:
        return "Turn 3: Corner North (+Y)"
    elif x < 2.5 and y >= 9.0:
        return "Station 5: Cobblestone Rough Terrain"
    return "Course Track"


def draw_hud(
    chase_frame: np.ndarray,
    vla_frame: np.ndarray | None,
    step: int,
    dist_goal: float,
    progress_xy: tuple[float, float],
    pred_action: np.ndarray,
    active_skill: str,
    station_text: str = "",
    status_text: str = "",
) -> np.ndarray:
    """Draw rich telemetry HUD with optional VLA perception PiP inset."""
    img = Image.fromarray(chase_frame)
    draw = ImageDraw.Draw(img)

    # Top banner background (dark slate)
    draw.rectangle([(0, 0), (img.width, 42)], fill=(15, 23, 42))

    # Top telemetry text
    top_text = (
        f"Step {step:03d} | Pos: ({progress_xy[0]:.2f}, {progress_xy[1]:.2f}) | "
        f"Goal: {dist_goal:.2f}m | "
        f"Skill: {active_skill}"
    )
    draw.text((12, 6), top_text, fill=(241, 245, 249))

    # Sub banner text
    act_str = f"[{pred_action[0]:+.2f}, {pred_action[1]:+.2f}, {pred_action[2]:+.2f}]" if len(pred_action) >= 3 else f"[{pred_action[0]:+.2f}, {pred_action[1]:+.2f}]"
    sub_text = f"Terrain: {station_text} | VLA Steering: {act_str}"
    draw.text((12, 24), sub_text, fill=(148, 163, 184))

    # Picture-in-picture (PiP) inset for VLA perception frame
    if vla_frame is not None:
        pip_w, pip_h = 160, 160
        pip_img = Image.fromarray(vla_frame).resize((pip_w, pip_h), Image.Resampling.BILINEAR)
        pip_x = img.width - pip_w - 12
        pip_y = 50

        # Border and background
        draw.rectangle([(pip_x - 2, pip_y - 18), (pip_x + pip_w + 2, pip_y + pip_h + 2)], fill=(15, 23, 42), outline=(56, 189, 248), width=1)
        draw.text((pip_x + 4, pip_y - 15), "VLA Camera View", fill=(56, 189, 248))
        img.paste(pip_img, (pip_x, pip_y))

    # Bottom status banner
    if status_text:
        col = (34, 197, 94) if "SUCCESS" in status_text else (239, 68, 68)
        draw.rectangle([(0, img.height - 34), (img.width, img.height)], fill=(15, 23, 42))
        draw.text((12, img.height - 26), status_text, fill=col)

    return np.asarray(img)


def render_vla_perception(env: MujocoRadialSphereEnv, renderer: mujoco.Renderer) -> np.ndarray:
    """Render 256x256 frame tailored for VLA perception."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.core_body_id
    cam.distance = 2.8
    cam.elevation = -32.0
    cam.azimuth = 135.0
    renderer.update_scene(env.data, camera=cam)
    return renderer.render()


def render_chase_view(env: MujocoRadialSphereEnv, renderer: mujoco.Renderer) -> np.ndarray:
    """Render 640x480 path-anchored behind camera for broadcast evaluation."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.core_body_id
    cam.distance = 2.0
    cam.elevation = -26.0

    pos = env.data.qpos[:2]
    v = env.data.qvel[:2]
    if hasattr(env, "path_pts") and len(env.path_pts) > 1:
        dists = np.linalg.norm(env.path_pts - pos, axis=1)
        idx = int(np.argmin(dists))
        lookahead_idx = min(idx + 12, len(env.path_pts) - 1)
        fwd = env.path_pts[lookahead_idx] - pos
        if np.linalg.norm(fwd) < 0.2:
            fwd = env.path_pts[-1] - env.path_pts[-2]
        path_yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
        path_dir = np.array([np.cos(np.radians(path_yaw)), np.sin(np.radians(path_yaw))])
        forward_speed = float(np.dot(v, path_dir))
        if forward_speed > 0.3:
            v_yaw = float(np.degrees(np.arctan2(v[1], v[0])))
            ang_diff = (v_yaw - path_yaw + 180.0) % 360.0 - 180.0
            target_yaw = path_yaw + 0.25 * ang_diff if abs(ang_diff) < 40.0 else path_yaw
        else:
            target_yaw = path_yaw
    else:
        target_yaw = 0.0

    if not hasattr(env, "_cam_azimuth_smooth") or env._cam_azimuth_smooth is None:
        env._cam_azimuth_smooth = target_yaw
    else:
        delta = (target_yaw - env._cam_azimuth_smooth + 180.0) % 360.0 - 180.0
        env._cam_azimuth_smooth += 0.15 * delta

    cam.azimuth = env._cam_azimuth_smooth
    renderer.update_scene(env.data, camera=cam)
    return renderer.render()


def evaluate_baseline_episode(
    policy: RoboballVLAPolicy,
    cfg,
    device: torch.device,
    max_steps: int = 500,
    seed: int = 42,
    waypoint_guidance: bool = True,
    record_frames: bool = True,
) -> dict:
    """Roll out one zero-shot evaluation episode using modular skills_vla."""
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

    # Dedicated renderers: 256x256 for policy perception, 640x480 for broadcast chase cam
    vla_renderer = mujoco.Renderer(env.model, height=256, width=256)
    chase_renderer = mujoco.Renderer(env.model, height=480, width=640)

    while step < max_steps:
        pos = env.data.qpos[:3].copy()
        quat = env.data.qpos[3:7].copy()
        vel = env.data.qvel[:3].copy()
        ball_xy = pos[:2]

        rel_final_goal = goal - ball_xy
        dist_goal = float(np.linalg.norm(rel_final_goal))

        # Select target: either lookahead path waypoint or global goal
        if waypoint_guidance and hasattr(env, "path_pts") and len(env.path_pts) > 1:
            dists = np.linalg.norm(env.path_pts - ball_xy, axis=1)
            closest_idx = int(np.argmin(dists))
            target_idx = min(closest_idx + 12, len(env.path_pts) - 1)
            target_pt = env.path_pts[target_idx]
            rel_target = target_pt - ball_xy
            dist_target = float(np.linalg.norm(rel_target))
        else:
            rel_target = rel_final_goal
            dist_target = dist_goal

        # Render VLA visual perception frame (256x256)
        vla_frame = render_vla_perception(env, vla_renderer)

        # Build 11-D proprioceptive state vector
        state_vec = np.concatenate([
            ball_xy.astype(np.float32),
            vel[:2].astype(np.float32),
            rel_target.astype(np.float32),
            np.array([dist_target], dtype=np.float32),
            quat.astype(np.float32),
        ])

        img_tensor = torch.from_numpy(vla_frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        st_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred_act = policy(img_tensor, st_tensor).squeeze(0).cpu().numpy()

        v_world = pred_act[:2]
        v_norm = float(np.linalg.norm(v_world))
        drive = float(pred_act[2]) if len(pred_act) > 2 else 1.0

        if step % 20 == 0:
            logger.info(f"Step {step:03d} | Pos: ({ball_xy[0]:.2f}, {ball_xy[1]:.2f}) | Act: [{pred_act[0]:+.3f}, {pred_act[1]:+.3f}, {drive:+.3f}] | v_norm: {v_norm:.3f}")

        # Construct RobotState for skills_vla
        robot_state = RobotState(
            quat=quat,
            dirs_body=env.dirs_body,
            max_extend=env.max_extend,
            lin_vel=vel[:2],
            core_z=float(pos[2]),
            core_vz=float(vel[2]),
        )

        # Dispatch through modular skills_vla package
        if v_norm > 0.15:
            d_hat = v_world / v_norm
            speed = float(np.clip(v_norm, 0.6, 1.4))
            action_res = dispatch_vla_action(
                "roll",
                action_params={"d_world": d_hat, "speed": speed},
                state=robot_state,
            )
            rod_targets = action_res.targets
            active_skill = "skills_vla.roll"
        else:
            # Fallback to goal heading if policy output is low-confidence
            d_hat = rel_goal / max(dist_goal, 1e-4)
            speed = 0.8
            action_res = dispatch_vla_action(
                "roll",
                action_params={"d_world": d_hat, "speed": speed},
                state=robot_state,
            )
            rod_targets = action_res.targets
            active_skill = "skills_vla.roll (fallback)"

        # Step physics environment
        env.step(rod_targets)

        now_pos = env.data.qpos[:2].copy()
        disp = float(np.linalg.norm(now_pos - last_pos))
        if step > 25 and disp < 0.008:
            stuck_counter += 1
        else:
            stuck_counter = 0
        last_pos = now_pos.copy()

        station_desc = get_station_description(float(now_pos[0]), float(now_pos[1]))

        if record_frames:
            chase_frame = render_chase_view(env, chase_renderer)
            hud_frame = draw_hud(
                chase_frame=chase_frame,
                vla_frame=vla_frame,
                step=step,
                dist_goal=dist_goal,
                progress_xy=(float(now_pos[0]), float(now_pos[1])),
                pred_action=pred_act,
                active_skill=active_skill,
                station_text=station_desc,
            )
            frames_recorded.append(hud_frame)

        step += 1

        # Terminate early if stuck on an obstacle/hurdle for > 60 steps
        if stuck_counter > 60 or dist_goal < 0.35:
            break

    final_pos = env.data.qpos[:2].copy()
    final_dist = float(np.linalg.norm(final_pos - goal))
    success = bool(final_dist < 0.35)
    env.close()

    station_final = get_station_description(float(final_pos[0]), float(final_pos[1]))
    failure_reason = ""
    if not success:
        if "Hurdle" in station_final:
            failure_reason = "Zero-Shot Gap: Stalled at Station 2 (0.18m Hurdle) - policy lacks jump skill"
        elif "Boxes" in station_final or "Valleys" in station_final:
            failure_reason = "Zero-Shot Gap: Stalled at Station 3 (Platforms/Gaps) - policy lacks leap skill"
        else:
            failure_reason = f"Stuck at ({final_pos[0]:.2f}, {final_pos[1]:.2f}) along {station_final}"

    if record_frames and frames_recorded:
        badge = "RESULT: SUCCESS" if success else f"RESULT: FAILED - {failure_reason}"
        last_hud = draw_hud(
            chase_frame=frames_recorded[-1],
            vla_frame=None,
            step=step,
            dist_goal=final_dist,
            progress_xy=(float(final_pos[0]), float(final_pos[1])),
            pred_action=pred_act,
            active_skill=active_skill,
            station_text=station_final,
            status_text=badge,
        )
        for _ in range(35):
            frames_recorded.append(last_hud)

    return {
        "steps": step,
        "final_x": float(final_pos[0]),
        "final_y": float(final_pos[1]),
        "final_dist_to_goal": final_dist,
        "station_reached": station_final,
        "failure_reason": failure_reason,
        "success": success,
        "frames": frames_recorded,
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot baseline evaluation of flat VLA with skills_vla.")
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Max evaluation steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Evaluation seed",
    )
    parser.add_argument(
        "--waypoint-guidance",
        action="store_true",
        default=True,
        help="Use corridor lookahead waypoints instead of distant global goal",
    )
    parser.add_argument(
        "--no-waypoint-guidance",
        dest="waypoint_guidance",
        action="store_false",
        help="Disable corridor waypoints and aim directly at global goal",
    )
    parser.add_argument(
        "--artifact-video",
        type=str,
        default="/home/azureuser/.gemini/antigravity-ide/brain/e14032e8-5276-4443-8925-16d7ecbcdbd7/vla_zero_shot_skills_eval.mp4",
        help="Path to copy final video to artifacts dir",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading VLA Checkpoint: {args.checkpoint} on {device}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = RoboballVLAPolicy().to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    cfg = load_config(args.config)
    logger.info(f"Starting Zero-Shot Evaluation on Playground Parkour with skills_vla (seed={args.seed}, waypoint_guidance={args.waypoint_guidance})...")
    res = evaluate_baseline_episode(
        policy,
        cfg,
        device,
        max_steps=args.max_steps,
        seed=args.seed,
        waypoint_guidance=args.waypoint_guidance,
    )

    logger.info("=" * 65)
    logger.info("ZERO-SHOT PLAYGROUND EVALUATION COMPLETE (skills_vla)")
    logger.info(f"Final Position:   ({res['final_x']:.2f}, {res['final_y']:.2f})")
    logger.info(f"Final Dist Goal:  {res['final_dist_to_goal']:.2f}m")
    logger.info(f"Station Reached:  {res['station_reached']}")
    logger.info(f"Steps Executed:   {res['steps']}")
    logger.info(f"Success:          {res['success']}")
    if res["failure_reason"]:
        logger.info(f"Failure Reason:   {res['failure_reason']}")
    logger.info("=" * 65)

    summary_file = out_dir / "zero_shot_summary.json"
    with open(summary_file, "w") as f:
        json.dump({k: v for k, v in res.items() if k != "frames"}, f, indent=2)
    logger.info(f"Saved summary to: {summary_file}")

    if res["frames"]:
        video_path = out_dir / "zero_shot_playground_rollout.mp4"
        imageio.mimsave(str(video_path), res["frames"], fps=25)
        logger.info(f"Saved rollout video to: {video_path}")

        if args.artifact_video:
            artifact_path = Path(args.artifact_video)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(video_path), str(artifact_path))
            logger.info(f"Copied video to artifacts: {artifact_path}")


if __name__ == "__main__":
    main()
