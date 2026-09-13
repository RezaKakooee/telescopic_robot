"""Evaluate a trained steering or skill-arbitration policy with episode videos.

Loads a PPO checkpoint (and its VecNormalize obs stats), runs episodes with
the chase camera enabled, and saves videos exactly like the other agents.

Usage (from the repo root):
    python scripts/rl/eval_rl.py --run storage_local/<rl_train run dir>
    python scripts/rl/eval_rl.py --run <run_dir> --episodes 5 --kind obstacle
    python scripts/rl/eval_rl.py --run <run_dir> --scenario <scenario.json>
"""
from __future__ import annotations

from radial_sphere.config import script_config

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import pickle
from pathlib import Path

import numpy as np
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.utils import check_for_correct_spaces
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv

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
    args = script_config("eval_rl", passthrough=True)

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

    saved_cfg = model_path.parent.parent / "code" / "config.yaml"
    config_path = args.config
    if not config_path and not args.config_name and saved_cfg.exists():
        config_path = saved_cfg
    cfg = load_config_cli(path=config_path, name=args.config_name,
                          overrides=args.scenario_overrides)
    kind = args.kind or str(cfg.scenario.kind)
    video_cfg = getattr(cfg, "video", None)
    frame_every = int(getattr(video_cfg, "frame_every", 3))
    fps = int(getattr(video_cfg, "fps", 24))

    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    elif train_path.is_dir():
        run_dir = train_path / "evaluation"
    else:
        run_dir = make_run_dir(build_run_id("eval_rl", tag=kind))
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)
    save_code(run_dir, __file__, cfg=cfg)
    log.info(f"Run dir : {run_dir}")

    if args.scenario:
        scenario = Scenario.load(args.scenario)
        randomize = False
    else:
        scenario = generate_scenario(kind, cfg, seed=args.seed)
        maze_level = int(getattr(getattr(cfg.scenario, "maze", None), "level", 1))
        randomize = (kind in ("goal", "obstacle") or
                     (kind == "maze" and maze_level == 3))

    if scenario.kind in ("campus", "university_campus", "playground", "robotics_playground", "proving_ground"):
        env = SkillArbitrationEnv(cfg, scenario=scenario, seed=args.seed,
                                  max_steps=int(getattr(cfg.rl, "max_steps", cfg.env.max_steps)))
        log.info(f"skill_backend={env.skill_backend}  action_mode={env.action_mode}")
    else:
        env = SteeringEnv(cfg, scenario=scenario, randomize=randomize,
                          output_dir=run_dir, seed=args.seed)
    check_for_correct_spaces(env, model.observation_space, model.action_space)
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
            if isinstance(env, SkillArbitrationEnv):
                # Sample simulation time inside options so full jumps stay visible
                # and variable-duration decisions do not change playback speed.
                next_frame = float(env.env.data.time) + 1.0 / fps
                def record_control_step(current):
                    nonlocal next_frame
                    now = float(current.env.data.time)
                    while now + 1e-9 >= next_frame:
                        recorder.add(current.render_all())
                        next_frame += 1.0 / fps
                env.on_control_step = record_control_step
        total_r, hl_step = 0.0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            x = norm.normalize_obs(obs) if norm is not None else obs
            action, _ = model.predict(x, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            hl_step += 1
            if (recorder is not None and not isinstance(env, SkillArbitrationEnv)
                    and hl_step % max(frame_every // env.k, 1) == 0):
                recorder.add(env.render_all())
        returns.append(total_r)
        steps.append(info["step"])
        success = bool(info.get("success", False))
        successes.append(success)
        log.info(f"ep {ep + 1:3d}  return={total_r:+.3f}  env_steps={info['step']}  "
                 f"contact={success}  goal={np.round(env.env.scenario.goal, 2).tolist()}")
        if recorder is not None:
            if isinstance(env, SkillArbitrationEnv):
                env.on_control_step = None
            recorder.close()
            for pth in recorder.paths:
                log.info(f"video: {recorder.n_frames} frames → {pth}")

    log.info(f"mean return : {np.mean(returns):+.3f} ± {np.std(returns):.3f}")
    log.info(f"mean steps  : {np.mean(steps):.0f}")
    log.info(f"success rate: {np.mean(successes) * 100:.1f}%")
    env.close()


if __name__ == "__main__":
    main()
