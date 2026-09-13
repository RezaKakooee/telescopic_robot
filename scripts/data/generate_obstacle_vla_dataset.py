"""Demonstration Dataset Generator for Roboball Obstacle Navigation (VLA / LeRobot).

Generates expert demonstration trajectories of the 60-rod spherical robot (RadialSphere)
navigating through randomized obstacle fields to reach target locations without collision.

Records:
- Visual Observations: RGB camera frames (chase camera showing robot, obstacles, and goal)
- Robot States: [pos_x, pos_y, vel_x, vel_y, goal_dx, goal_dy, quat_w, quat_x, quat_y, quat_z]
- Actions: High-level steering [v_fwd, v_lat, drive] + Low-level 60D rod targets
- Language Instruction: "Navigate around the obstacles to reach the target"
- Trajectory Metadata: obstacle coordinates, distances, clearance, timestamps, success

Outputs:
- HDF5 dataset archive containing all episodes with compressed images and state-action pairs
- Individual NPZ episode files for fast per-trajectory inspection
- dataset_summary.json index
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from PIL import Image

import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill
from skills.high_level import go_to_goal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_vla_data")


def render_vla_frame(env: MujocoRadialSphereEnv, width: int = 256, height: int = 256) -> np.ndarray:
    """Render a visual frame tailored for VLA perception.

    Uses a tracking overhead-chase camera showing the sphere, surrounding obstacle pillars,
    and arena floor markers.
    """
    if env.renderer is None:
        env.renderer = mujoco.Renderer(env.model, height=height, width=width)
    elif env.renderer.height != height or env.renderer.width != width:
        env.renderer = mujoco.Renderer(env.model, height=height, width=width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.core_body_id
    cam.distance = 2.8
    cam.elevation = -32.0
    cam.azimuth = 135.0
    env.renderer.update_scene(env.data, camera=cam)
    return env.renderer.render()


def pillar_clearance(obstacles: np.ndarray, ball_xy: np.ndarray) -> float:
    """Distance from ball surface (r=0.15m) to nearest obstacle pillar."""
    if len(obstacles) == 0:
        return float("inf")
    dists = np.linalg.norm(obstacles[:, :2] - ball_xy[:2], axis=1) - obstacles[:, 2]
    return float(np.min(dists) - 0.15)


def collect_single_episode(
    seed: int,
    cfg,
    max_steps: int = 600,
    camera_res: tuple[int, int] = (256, 256),
    task_instruction: str = "Navigate around the obstacles to reach the target",
) -> dict | None:
    """Roll out one expert demonstration episode using the goal-seeking route planner."""
    scenario = generate_scenario("obstacle", cfg, seed=seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=max_steps + 100)
    env.reset(seed=seed)

    # Initial settle steps
    for _ in range(20):
        env.step(np.full(len(env.dirs_body), 0.025, dtype=np.float32))

    goal = np.asarray(scenario.goal, dtype=np.float32)[:2]
    obstacles = np.asarray(scenario.obstacles, dtype=np.float32).reshape(-1, 3)

    frames = []
    states = []
    actions_highlevel = []
    actions_lowlevel = []
    rewards = []
    timestamps = []

    min_clearance = float("inf")
    arrived = False
    step = 0
    hold_steps = 15

    while step < max_steps:
        pos = env.data.qpos[:3].copy()
        quat = env.data.qpos[3:7].copy()
        vel = env.data.qvel[:3].copy()
        ball_xy = pos[:2]
        sim_time = float(env.data.time)

        # Plan expert skill and route
        cmd = go_to_goal(
            ball_xy=ball_xy,
            goal_xy=goal,
            lin_vel=vel[:2],
            obstacles=obstacles,
            steps=scenario.steps,
            speed=1.2,
            clearance=0.35,
            goal_tolerance=0.35,
        )

        # Execute expert skill to generate 60D low-level targets
        rod_targets = execute_skill(cmd.skill, quat, env.dirs_body, env.max_extend, **cmd.kwargs)

        # High-level steering action [v_x, v_y, drive]
        if cmd.skill == "stop":
            act_high = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            # Direction towards current waypoint / goal
            route = cmd.meta.get("route", [ball_xy, goal])
            target_pt = route[1] if len(route) > 1 else goal
            delta = target_pt - ball_xy
            norm_d = np.linalg.norm(delta)
            dir_unit = delta / max(norm_d, 1e-4)
            act_high = np.array([dir_unit[0] * 1.2, dir_unit[1] * 1.2, 1.0], dtype=np.float32)

        # Capture RGB visual observation
        frame = render_vla_frame(env, width=camera_res[0], height=camera_res[1])

        # State vector: [pos_x, pos_y, vel_x, vel_y, rel_goal_x, rel_goal_y, dist_goal, quat(4)]
        rel_goal = goal - ball_xy
        dist_goal = float(np.linalg.norm(rel_goal))
        state_vec = np.concatenate([
            ball_xy.astype(np.float32),
            vel[:2].astype(np.float32),
            rel_goal.astype(np.float32),
            np.array([dist_goal], dtype=np.float32),
            quat.astype(np.float32),
        ])

        # Step simulation
        obs, r, terminated, truncated, info = env.step(rod_targets)
        done = terminated or truncated

        # Track clearance
        now_xy = env.data.qpos[:2].copy()
        curr_clr = pillar_clearance(obstacles, now_xy)
        min_clearance = min(min_clearance, curr_clr)

        frames.append(frame)
        states.append(state_vec)
        actions_highlevel.append(act_high)
        actions_lowlevel.append(rod_targets.astype(np.float32))
        rewards.append(float(r))
        timestamps.append(sim_time)

        step += 1

        if cmd.skill == "stop" and float(np.linalg.norm(env.data.qvel[:2])) < 0.15:
            if not arrived:
                arrived = True
            hold_steps -= 1
            if hold_steps <= 0:
                break

    end_pos = env.data.qpos[:2].copy()
    final_dist = float(np.linalg.norm(end_pos - goal))
    success = bool(arrived and final_dist <= 0.35 and min_clearance > -0.02)
    env.close()

    if not success:
        logger.warning(f"Seed {seed} failed: arrived={arrived}, dist={final_dist:.2f}, min_clr={min_clearance:.2f}")
        return None

    return {
        "seed": seed,
        "steps": len(frames),
        "frames": np.array(frames, dtype=np.uint8),
        "states": np.array(states, dtype=np.float32),
        "actions_highlevel": np.array(actions_highlevel, dtype=np.float32),
        "actions_lowlevel": np.array(actions_lowlevel, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "timestamps": np.array(timestamps, dtype=np.float32),
        "goal": goal,
        "obstacles": obstacles,
        "min_clearance": min_clearance,
        "final_dist": final_dist,
        "success": success,
        "task_instruction": task_instruction,
    }


def generate_vla_demonstrations(
    output_dir: Path,
    config_path: str = "configs/rl/obstacle_realistic.yaml",
    n_episodes: int = 100,
    max_steps: int = 600,
    seed_offset: int = 1000,
    camera_res: tuple[int, int] = (256, 256),
):
    """Generate and save N expert demonstration episodes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = output_dir / "episodes_npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)

    logger.info(f"Starting VLA demonstration collection: {n_episodes} target episodes...")
    start_time = time.time()

    episodes_meta = []
    collected = 0
    attempt = 0
    total_steps = 0

    h5_path = output_dir / "obstacle_vla_demos.h5"
    h5_file = h5py.File(str(h5_path), "w")

    try:
        while collected < n_episodes:
            seed = seed_offset + attempt
            attempt += 1

            ep_data = collect_single_episode(
                seed=seed,
                cfg=cfg,
                max_steps=max_steps,
                camera_res=camera_res,
            )

            if ep_data is None:
                continue

            ep_id = f"episode_{collected:05d}"
            n_steps = ep_data["steps"]
            total_steps += n_steps

            # Save per-episode NPZ
            npz_path = npz_dir / f"{ep_id}.npz"
            np.savez_compressed(
                str(npz_path),
                frames=ep_data["frames"],
                states=ep_data["states"],
                actions_highlevel=ep_data["actions_highlevel"],
                actions_lowlevel=ep_data["actions_lowlevel"],
                rewards=ep_data["rewards"],
                timestamps=ep_data["timestamps"],
                goal=ep_data["goal"],
                obstacles=ep_data["obstacles"],
                min_clearance=ep_data["min_clearance"],
                seed=ep_data["seed"],
                task_instruction=ep_data["task_instruction"],
            )

            # Store in HDF5 archive
            grp = h5_file.create_group(ep_id)
            grp.attrs["seed"] = ep_data["seed"]
            grp.attrs["steps"] = n_steps
            grp.attrs["min_clearance"] = ep_data["min_clearance"]
            grp.attrs["final_dist"] = ep_data["final_dist"]
            grp.attrs["task_instruction"] = ep_data["task_instruction"]

            grp.create_dataset("images", data=ep_data["frames"], compression="gzip")
            grp.create_dataset("states", data=ep_data["states"], compression="gzip")
            grp.create_dataset("actions_highlevel", data=ep_data["actions_highlevel"], compression="gzip")
            grp.create_dataset("actions_lowlevel", data=ep_data["actions_lowlevel"], compression="gzip")
            grp.create_dataset("timestamps", data=ep_data["timestamps"], compression="gzip")

            meta_entry = {
                "episode_id": ep_id,
                "seed": ep_data["seed"],
                "steps": n_steps,
                "min_clearance": float(ep_data["min_clearance"]),
                "final_dist": float(ep_data["final_dist"]),
                "n_obstacles": int(len(ep_data["obstacles"])),
                "task": ep_data["task_instruction"],
            }
            episodes_meta.append(meta_entry)
            collected += 1

            fps = total_steps / max(time.time() - start_time, 1e-3)
            logger.info(
                f"Episode [{collected:04d}/{n_episodes:04d}] collected (seed {seed}) | "
                f"Steps: {n_steps} | Min Clearance: {ep_data['min_clearance']:.2f}m | "
                f"Throughput: {fps:.1f} steps/s"
            )

    finally:
        h5_file.close()

    elapsed = time.time() - start_time
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_episodes": collected,
                "total_timesteps": total_steps,
                "elapsed_seconds": elapsed,
                "camera_resolution": list(camera_res),
                "state_dim": 11,
                "action_highlevel_dim": 3,
                "action_lowlevel_dim": 60,
                "episodes": episodes_meta,
            },
            f,
            indent=2,
        )

    logger.info("=" * 60)
    logger.info("VLA DEMONSTRATION DATASET GENERATION COMPLETE!")
    logger.info(f"Total Episodes:  {collected}")
    logger.info(f"Total Timesteps: {total_steps:,}")
    logger.info(f"HDF5 Archive:    {h5_path}")
    logger.info(f"Summary JSON:    {summary_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate VLA demonstration dataset for obstacle navigation.")
    parser.add_argument("--out-dir", type=str, default="storage_local/obstacle_vla_dataset", help="Output directory")
    parser.add_argument("--config", type=str, default="configs/rl/obstacle_realistic.yaml", help="Scenario config YAML")
    parser.add_argument("--episodes", type=int, default=50, help="Number of successful episodes to collect")
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum steps per episode")
    parser.add_argument("--seed-offset", type=int, default=1000, help="Starting random seed")
    parser.add_argument("--res", type=int, default=256, help="Camera resolution (square)")
    args = parser.parse_args()

    generate_vla_demonstrations(
        output_dir=Path(args.out_dir),
        config_path=args.config,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        seed_offset=args.seed_offset,
        camera_res=(args.res, args.res),
    )


if __name__ == "__main__":
    main()
