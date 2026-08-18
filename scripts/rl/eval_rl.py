"""Evaluate a trained steering policy: stats + episode videos.

Loads a PPO checkpoint (and its VecNormalize obs stats), runs episodes with
the chase camera enabled, and saves videos exactly like the other agents.

Usage (from the repo root):
    python scripts/rl/eval_rl.py --run storage_local/<rl_train run dir>
    python scripts/rl/eval_rl.py --run <run_dir> --episodes 5 --kind obstacle
    python scripts/rl/eval_rl.py --run <run_dir> --scenario <scenario.json>
"""
from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse
import pickle
from pathlib import Path

import numpy as np
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from stable_baselines3 import PPO  # noqa: E402

from radial_sphere import (MultiVideoRecorder, Scenario, SteeringEnv, VideoRecorder, build_run_id,  # noqa: E402
                           generate_scenario, load_config_cli, make_run_dir,
                           save_code, setup_logging)

setup_logging()


def load_obs_stats(path: Path):
    """Load VecNormalize obs statistics without needing a live VecEnv."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        norm = pickle.load(f)
    norm.training = False
    return norm


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained RadialSphere steering policy")
    p.add_argument("--run", required=True,
                   help="training run dir (with checkpoints/final.zip + vecnormalize.pkl)")
    p.add_argument("--kind", default="goal", help="scenario kind to evaluate on")
    p.add_argument("--scenario", default=None, help="or a fixed scenario JSON")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--config", default=None)
    p.add_argument("--no-video", dest="video", action="store_false")
    p.add_argument("--output-dir", default=None,
                   help="explicit output dir (default: creates new timestamped dir)")
    p.add_argument("--config-name", "-cn", dest="config_name", default=None,
                   help="config variant name under configs/rl/")
    p.add_argument("overrides", nargs="*",
                   help="config overrides as key=value (Hydra dotlist)")
    args = p.parse_args()

    train_path = Path(args.run)
    if train_path.is_file():
        model_path = train_path
        stats_path = train_path.parent / train_path.name.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
        if not stats_path.exists():
            stats_path = train_path.parent.parent / "vecnormalize.pkl"
    else:
        model_path = train_path / "checkpoints" / "final.zip"
        stats_path = train_path / "vecnormalize.pkl"
        if not model_path.exists():
            ckpts = sorted((train_path / "checkpoints").glob("ppo_*_steps.zip"),
                           key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0)
            if ckpts:
                model_path = ckpts[-1]
                stats_path = model_path.parent / model_path.name.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")

    log.info(f"Loading checkpoint: {model_path}")
    log.info(f"Loading stats     : {stats_path}")
    model = PPO.load(str(model_path), device="cpu")

    cfg = load_config_cli(path=args.config, name=args.config_name,
                          overrides=args.overrides)
    video_cfg = getattr(cfg, "video", None)
    frame_every = int(getattr(video_cfg, "frame_every", 3))
    fps = int(getattr(video_cfg, "fps", 24))

    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(build_run_id("eval_rl", tag=args.kind))
    setup_logging(run_dir)
    save_code(run_dir, __file__, cfg=cfg)
    log.info(f"Run dir : {run_dir}")

    if args.scenario:
        scenario = Scenario.load(args.scenario)
        randomize = False
    else:
        scenario = generate_scenario(args.kind, cfg, seed=args.seed)
        maze_level = int(getattr(getattr(cfg.scenario, "maze", None), "level", 1))
        randomize = (args.kind in ("goal", "obstacle") or
                     (args.kind == "maze" and maze_level == 3))

    env = SteeringEnv(cfg, scenario=scenario, randomize=randomize,
                      output_dir=run_dir, seed=args.seed)
    # normalise observations with the stats learned during training
    norm = load_obs_stats(stats_path)
    if norm is None:
        log.warning("vecnormalize.pkl not found — evaluating on raw observations")

    returns, steps, successes = [], [], []
    for ep in range(args.episodes):
        recorder = MultiVideoRecorder(run_dir / "renders", ep=ep + 1,
                                      fps=fps) if args.video else None
        obs, info = env.reset(seed=args.seed + ep)
        if recorder is not None:
            recorder.add(env.render_all())
        total_r, hl_step = 0.0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            x = norm.normalize_obs(obs) if norm is not None else obs
            action, _ = model.predict(x, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            hl_step += 1
            if recorder is not None and hl_step % max(frame_every // env.k, 1) == 0:
                recorder.add(env.render_all())
        returns.append(total_r)
        steps.append(info["step"])
        success = bool(info.get("success", False))
        successes.append(success)
        log.info(f"ep {ep + 1:3d}  return={total_r:+.3f}  env_steps={info['step']}  "
                 f"contact={success}  goal={np.round(env.env.scenario.goal, 2).tolist()}")
        if recorder is not None:
            recorder.close()
            for pth in recorder.paths:
                log.info(f"video: {recorder.n_frames} frames → {pth}")

    log.info(f"mean return : {np.mean(returns):+.3f} ± {np.std(returns):.3f}")
    log.info(f"mean steps  : {np.mean(steps):.0f}")
    log.info(f"success rate: {np.mean(successes) * 100:.1f}%")
    env.close()


if __name__ == "__main__":
    main()
