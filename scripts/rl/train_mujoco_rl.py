"""High-Performance PPO Training for RadialSphere on Native MuJoCo.

Uses direct native MuJoCo bindings (MujocoSteeringEnv) with vectorized parallel environments.
Zero MetaSim / PyTorch wrapper overhead for ultra-fast training (~1,000+ FPS).

Usage:
    python scripts/rl/train_mujoco_rl.py --kind maze --config-name maze_level3_random_endpoints --steps 300000 --n-envs 4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from radial_sphere import (
    MujocoLowLevelEnv,
    MujocoSteeringEnv,
    build_run_id,
    generate_scenario,
    load_config_cli,
    make_run_dir,
    save_code,
    setup_logging,
)

log = logging.getLogger("radial_sphere")
setup_logging()

TRAIN_KINDS = ("path", "goal", "obstacle", "maze")


def make_mujoco_env(cfg, kind: str, rank: int, seed: int, max_steps: int, mode: str = "steering"):
    """Thunk building one native MuJoCo training env."""
    def _thunk():
        scenario = generate_scenario(kind, cfg, seed=seed + rank)
        maze_level = int(getattr(getattr(cfg.scenario, "maze", None), "level", 1))
        randomize = (kind in ("goal", "obstacle") or (kind == "maze" and maze_level in (2, 3)))
        if mode == "lowlevel":
            return MujocoLowLevelEnv(
                cfg,
                scenario=scenario,
                randomize=randomize,
                max_steps=max_steps,
            )
        return MujocoSteeringEnv(
            cfg,
            scenario=scenario,
            randomize=randomize,
            max_steps=max_steps,
        )
    return _thunk


def main():
    p = argparse.ArgumentParser(description="RL training on Native MuJoCo")
    p.add_argument("--kind", choices=TRAIN_KINDS, default="maze")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo",
                   help="RL algorithm: ppo | sac (default: ppo)")
    p.add_argument("--mode", choices=["steering", "lowlevel"], default=None,
                   help="control mode: steering (3D action) | lowlevel (60D slide actions)")
    p.add_argument("--steps", type=int, default=None,
                   help="total RL timesteps (default: from config)")
    p.add_argument("--n-envs", type=int, default=None,
                   help="parallel envs (default: from config)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=None,
                   help="path to a config.yaml")
    p.add_argument("--config-name", "-cn", dest="config_name", default="maze_level3_random_endpoints",
                   help="config variant name under configs/rl/")
    p.add_argument("--device", default=None,
                   help="torch device: cpu | cuda (default: from config)")
    p.add_argument("--resume", default=None,
                   help="path to a run directory or model checkpoint to resume from")
    p.add_argument("overrides", nargs="*",
                   help="config overrides as key=value")
    args = p.parse_args()

    cfg = load_config_cli(path=args.config, name=args.config_name, overrides=args.overrides)
    rl = cfg.rl
    mode = str(args.mode if args.mode is not None else getattr(rl, "mode", "steering"))
    algo = str(args.algo).lower()
    max_steps = int(getattr(rl, "max_steps", cfg.env.max_steps))
    total_steps = int(args.steps if args.steps is not None else getattr(rl, "total_steps", 300000))
    n_envs = int(args.n_envs if args.n_envs is not None else getattr(rl, "n_envs", 4))
    device = str(args.device if args.device is not None else getattr(rl, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    run_dir = make_run_dir(build_run_id("train_mujoco_rl", tag=f"{args.kind}__{args.config_name}__{mode}__{algo}"))
    setup_logging(run_dir)
    save_code(run_dir, __file__, cfg=cfg)
    log.info(f"Native MuJoCo Training Run dir: {run_dir}")
    log.info(f"kind={args.kind}  algo={algo}  mode={mode}  n_envs={n_envs}  total_steps={total_steps}  seed={args.seed}")

    env_fns = [make_mujoco_env(cfg, args.kind, i, args.seed, max_steps, mode=mode) for i in range(n_envs)]
    vec = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    vec = VecMonitor(vec)

    # Check resume paths
    resume_model_path = None
    resume_norm_path = None
    if args.resume:
        r_path = Path(args.resume)
        if r_path.is_dir():
            m_cand = r_path / "checkpoints" / f"{algo}_final.zip"
            if not m_cand.exists():
                m_cand = r_path / "checkpoints" / "ppo_final.zip"
            n_cand = r_path / "checkpoints" / "vecnormalize_final.pkl"
            if m_cand.exists():
                resume_model_path = m_cand
            if n_cand.exists():
                resume_norm_path = n_cand
        elif r_path.is_file():
            resume_model_path = r_path
            n_cand = r_path.parent / "vecnormalize_final.pkl"
            if n_cand.exists():
                resume_norm_path = n_cand

    if resume_norm_path and resume_norm_path.exists():
        log.info(f"Loading VecNormalize statistics from: {resume_norm_path}")
        vec = VecNormalize.load(str(resume_norm_path), vec)
        vec.training = True
        vec.norm_reward = False
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    n_steps = int(getattr(rl, "n_steps", 256))
    batch_size = int(getattr(rl, "batch_size", 512))
    lr = float(getattr(rl, "lr", 3e-4))
    gamma = float(getattr(rl, "gamma", 0.99))
    gae_lambda = float(getattr(rl, "gae_lambda", 0.95))
    clip_range = float(getattr(rl, "clip_range", 0.2))
    ent_coef = float(getattr(rl, "ent_coef", 0.01))
    net_arch = list(getattr(rl, "net", [128, 128]))

    if algo == "sac":
        if resume_model_path and resume_model_path.exists():
            log.info(f"Resuming SAC model weights from: {resume_model_path}")
            model = SAC.load(str(resume_model_path), env=vec, device=device, tensorboard_log=str(run_dir / "tb"))
        else:
            model = SAC(
                "MlpPolicy",
                vec,
                learning_rate=lr,
                buffer_size=100000,
                learning_starts=5000,
                batch_size=batch_size,
                tau=0.005,
                gamma=gamma,
                ent_coef="auto",
                policy_kwargs=dict(net_arch=dict(pi=net_arch, qf=net_arch)),
                verbose=1,
                seed=args.seed,
                device=device,
                tensorboard_log=str(run_dir / "tb"),
            )
    else:
        if resume_model_path and resume_model_path.exists():
            log.info(f"Resuming PPO model weights from: {resume_model_path}")
            model = PPO.load(str(resume_model_path), env=vec, device=device, tensorboard_log=str(run_dir / "tb"))
        else:
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
                device=device,
                tensorboard_log=str(run_dir / "tb"),
            )

    ckpt_cb = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix=algo,
        save_vecnormalize=True,
    )

    log.info(f"Starting {algo.upper()} training loop...")
    model.learn(total_timesteps=total_steps, callback=ckpt_cb, progress_bar=False)

    model.save(str(run_dir / "checkpoints" / f"{algo}_final.zip"))
    vec.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    vec.close()

    log.info(f"Native MuJoCo {algo.upper()} Training complete! Saved to {run_dir}")


if __name__ == "__main__":
    main()
