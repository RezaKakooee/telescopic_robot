"""Generate Expert Demonstration Dataset for Playground Parkour VLA.

Rolls out the trained PPO expert policy from:
  storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/checkpoints/final.zip
and records:
  - Visual RGB observation frame (256x256x3)
  - Terrain & guidance observation vector (530-D)
  - Robot state vector (pos, vel, quat, waypoint)
  - High-level macro skill action (10-D logits)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import time

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

import mujoco

from radial_sphere.config import load_config
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from scripts.data.generate_obstacle_vla_dataset import render_vla_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_playground_dataset")


def load_obs_stats(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        norm = pickle.load(f)
    norm.training = False
    return norm


from radial_sphere.config import load_config_cli


def collect_playground_demos(
    model_path: Path,
    stats_path: Path,
    config_path: Path,
    output_h5: Path,
    n_episodes: int = 25,
    seed_offset: int = 100,
    max_macro_steps: int = 60,
):
    logger.info(f"Loading expert model: {model_path}")
    model = PPO.load(str(model_path), device="cpu")
    norm = load_obs_stats(stats_path)
    cfg = load_config_cli(path=str(config_path))

    scenario = generate_scenario("playground", cfg, seed=seed_offset)
    env = SkillArbitrationEnv(cfg, scenario=scenario, seed=seed_offset, max_steps=int(getattr(cfg.rl, "max_steps", 1500)))

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(output_h5, "w")

    successful_episodes = 0
    total_transitions = 0
    episodes_meta = []

    logger.info(f"Starting demonstration collection: target {n_episodes} successful episodes...")

    for ep_idx in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep_idx)
        frames = []
        obs_vectors = []
        states = []
        actions = []
        rewards = []

        terminated = truncated = False
        step = 0

        while not (terminated or truncated) and step < max_macro_steps:
            # Capture RGB tracking frame
            frame = render_vla_frame(env.env, width=256, height=256)

            # Capture low-level state
            pos = env.env.data.qpos[:3].copy()
            quat = env.env.data.qpos[3:7].copy()
            vel = env.env.data.qvel[:3].copy()
            goal = np.asarray(env.scenario.goal, dtype=np.float32)[:2]
            rel_goal = goal - pos[:2]
            dist_goal = float(np.linalg.norm(rel_goal))

            st = np.concatenate([
                pos[:2].astype(np.float32),
                vel[:2].astype(np.float32),
                rel_goal.astype(np.float32),
                np.array([dist_goal], dtype=np.float32),
                quat.astype(np.float32),
            ])

            # Expert action
            norm_obs = norm.normalize_obs(obs) if norm is not None else obs
            action, _ = model.predict(norm_obs, deterministic=True)

            frames.append(frame)
            obs_vectors.append(obs.astype(np.float32))
            states.append(st)
            actions.append(action.astype(np.float32))

            # Step environment with macro skill action
            obs, r, terminated, truncated, info = env.step(action)
            rewards.append(float(r))
            step += 1

        success = bool(info.get("success", False))
        final_pos = env.env.data.qpos[:2].copy()
        final_dist = float(np.linalg.norm(final_pos - goal))
        logger.info(f"Ep {ep_idx} finished: steps={step}, term={terminated}, trunc={truncated}, success={success}, pos=({final_pos[0]:.2f}, {final_pos[1]:.2f}), dist={final_dist:.2f}m")

        # Only save successful rollouts (reached goal or progress > 90%)
        if success or final_dist < 1.0:
            ep_grp = h5_file.create_group(f"episode_{successful_episodes:03d}")
            ep_grp.create_dataset("frames", data=np.array(frames, dtype=np.uint8), compression="gzip", compression_opts=4)
            ep_grp.create_dataset("obs_vectors", data=np.array(obs_vectors, dtype=np.float32))
            ep_grp.create_dataset("states", data=np.array(states, dtype=np.float32))
            ep_grp.create_dataset("actions", data=np.array(actions, dtype=np.float32))
            ep_grp.create_dataset("rewards", data=np.array(rewards, dtype=np.float32))

            ep_grp.attrs["steps"] = len(frames)
            ep_grp.attrs["success"] = success
            ep_grp.attrs["final_dist"] = final_dist
            ep_grp.attrs["seed"] = 1000 + ep_idx

            successful_episodes += 1
            total_transitions += len(frames)

            episodes_meta.append({
                "episode": successful_episodes,
                "steps": len(frames),
                "success": success,
                "final_dist": final_dist,
            })

            logger.info(
                f"Saved Demo [{successful_episodes:02d}/{n_episodes:02d}] | "
                f"Steps: {len(frames)} | Success: {success} | Final Dist: {final_dist:.2f}m"
            )

    env.close()
    h5_file.close()

    summary = {
        "dataset_path": str(output_h5),
        "total_episodes": successful_episodes,
        "total_transitions": total_transitions,
        "episodes": episodes_meta,
    }

    summary_file = output_h5.parent / "vla_demos_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"PLAYGROUND VLA DEMONSTRATION DATASET COMPLETE: {output_h5}")
    logger.info(f"Total Episodes:    {successful_episodes}")
    logger.info(f"Total Transitions: {total_transitions}")
    logger.info(f"Summary JSON:      {summary_file}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Collect playground parkour VLA demonstrations.")
    parser.add_argument(
        "--model",
        type=str,
        default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/checkpoints/final.zip",
        help="PPO expert model path",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/vecnormalize.pkl",
        help="VecNormalize stats path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/code/config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/dataset/playground_vla_demos.h5",
        help="Output HDF5 path",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of demonstration episodes")
    args = parser.parse_args()

    collect_playground_demos(
        model_path=Path(args.model),
        stats_path=Path(args.stats),
        config_path=Path(args.config),
        output_h5=Path(args.out),
        n_episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
