"""High-Performance PPO Training for RadialSphere on Native MuJoCo.

Uses direct native MuJoCo bindings (MujocoSteeringEnv) with vectorized parallel environments.
Zero MetaSim / PyTorch wrapper overhead for ultra-fast training (~1,000+ FPS).

Usage:
    python scripts/rl/train_mujoco_rl.py --kind maze --config-name maze_level3_random_endpoints --steps 300000 --n-envs 4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from radial_sphere import (
    MujocoSteeringEnv,
    build_run_id,
    generate_scenario,
    load_config_cli,
    make_run_dir,
    save_code,
    setup_logging,
)

setup_logging()

TRAIN_KINDS = ("path", "goal", "obstacle", "maze")


def make_mujoco_env(cfg, kind: str, rank: int, seed: int, max_steps: int):
    """Thunk building one native MuJoCo training env."""
    def _thunk():
        scenario = generate_scenario(kind, cfg, seed=seed + rank)
        maze_level = int(getattr(getattr(cfg.scenario, "maze", None), "level", 1))
        randomize = (kind in ("goal", "obstacle") or (kind == "maze" and maze_level == 3))
        return MujocoSteeringEnv(
            cfg,
            scenario=scenario,
            randomize=randomize,
            max_steps=max_steps,
        )
    return _thunk


def main():
    p = argparse.ArgumentParser(description="PPO training on Native MuJoCo")
    p.add_argument("--kind", choices=TRAIN_KINDS, default="maze")
    p.add_argument("--steps", type=int, default=300000,
                   help="total PPO timesteps (default: 300000)")
    p.add_argument("--n-envs", type=int, default=4,
                   help="parallel envs (default: 4)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=None,
                   help="path to a config.yaml")
    p.add_argument("--config-name", "-cn", dest="config_name", default="maze_level3_random_endpoints",
                   help="config variant name under configs/rl/")
    p.add_argument("overrides", nargs="*",
                   help="config overrides as key=value")
    args = p.parse_args()

    cfg = load_config_cli(path=args.config, name=args.config_name, overrides=args.overrides)
    rl = cfg.rl
    max_steps = int(getattr(rl, "max_steps", cfg.env.max_steps))
    total_steps = int(args.steps if args.steps is not None else getattr(rl, "total_steps", 300000))
    n_envs = int(args.n_envs if args.n_envs is not None else 4)

    run_dir = make_run_dir(build_run_id("train_mujoco_rl", tag=f"{args.kind}__{args.config_name}"))
    setup_logging(run_dir)
    save_code(run_dir, __file__, cfg=cfg)
    log.info(f"Native MuJoCo Training Run dir: {run_dir}")
    log.info(f"kind={args.kind}  n_envs={n_envs}  total_steps={total_steps}  seed={args.seed}")

    env_fns = [make_mujoco_env(cfg, args.kind, i, args.seed, max_steps) for i in range(n_envs)]
    vec = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    vec = VecMonitor(vec)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    n_steps = int(getattr(rl, "n_steps", 256))
    batch_size = int(getattr(rl, "batch_size", 512))
    lr = float(getattr(rl, "lr", 3e-4))
    gamma = float(getattr(rl, "gamma", 0.99))
    gae_lambda = float(getattr(rl, "gae_lambda", 0.95))
    clip_range = float(getattr(rl, "clip_range", 0.2))
    ent_coef = float(getattr(rl, "ent_coef", 0.01))
    net_arch = list(getattr(rl, "net", [128, 128]))

    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=dict(pi=net_arch, vf=net_arch)),
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(run_dir / "tb"),
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )

    log.info("Starting PPO training loop...")
    model.learn(total_timesteps=total_steps, callback=ckpt_cb, progress_bar=False)

    model.save(str(run_dir / "checkpoints" / "ppo_final.zip"))
    vec.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    vec.close()

    log.info(f"Native MuJoCo Training complete! Saved to {run_dir}")


if __name__ == "__main__":
    main()
