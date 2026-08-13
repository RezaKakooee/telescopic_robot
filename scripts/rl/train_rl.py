"""Train a high-level steering policy with PPO (stable-baselines3).

The RL agent picks a travel direction (+ drive); the scripted bar controller
turns that into 60 per-bar extensions (see radial_sphere/steering.py).
Training runs without the chase camera (much faster) and, for the goal task,
resamples a random goal every episode.

All outputs go to a timestamped run dir under ``storage_local/``:
checkpoints/, vecnormalize.pkl, tb/ (tensorboard), code/ snapshot.

Usage (from the repo root):
    python scripts/rl/train_rl.py                     # goal finding, defaults
    python scripts/rl/train_rl.py --kind obstacle --steps 300000
    python scripts/rl/train_rl.py rl.n_steps=512 rl.lr=0.0001   # overrides
"""
from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse

import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,  # noqa: E402
                                              VecMonitor, VecNormalize)

from omegaconf import OmegaConf  # noqa: E402

from radial_sphere import (SteeringEnv, build_run_id, generate_scenario,  # noqa: E402
                           load_config_cli, make_run_dir, save_code, setup_logging)

setup_logging()


def load_config_dict_section(section):
    """DictConfig section → plain dict (for the wandb config)."""
    return OmegaConf.to_container(section, resolve=True)

TRAIN_KINDS = ("path", "goal", "obstacle", "maze")


def make_env(cfg, kind: str, rank: int, run_dir, seed: int, max_steps: int):
    """Thunk building one training env in a worker process."""
    def _thunk():
        scenario = generate_scenario(kind, cfg, seed=seed + rank)
        return SteeringEnv(
            cfg,
            scenario=scenario,
            enable_camera=False,             # no rendering during training
            randomize=(kind in ("goal", "obstacle")),   # new task every episode
            max_steps=max_steps,
            output_dir=run_dir / "assets" / f"env_{rank}",
            seed=seed + rank,
        )
    return _thunk


def main():
    p = argparse.ArgumentParser(description="PPO training for RadialSphere steering")
    p.add_argument("--kind", choices=TRAIN_KINDS, default="goal")
    p.add_argument("--steps", type=int, default=None,
                   help="total PPO timesteps (default: config rl.total_steps)")
    p.add_argument("--n-envs", type=int, default=None,
                   help="parallel envs (default: config rl.n_envs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None,
                   help="path to a config.yaml (default: configs/rl/config.yaml)")
    p.add_argument("--config-name", "-cn", dest="config_name", default=None,
                   help="config variant name under configs/rl/")
    p.add_argument("overrides", nargs="*",
                   help="config overrides as key=value (Hydra dotlist)")
    args = p.parse_args()

    cfg = load_config_cli(path=args.config, name=args.config_name,
                          overrides=args.overrides)
    rl = cfg.rl
    # training-time overrides (shorter episodes / settle)
    cfg.env.n_settle_steps = int(getattr(rl, "n_settle_steps", cfg.env.n_settle_steps))
    max_steps = int(getattr(rl, "max_steps", cfg.env.max_steps))
    total_steps = int(args.steps if args.steps is not None else rl.total_steps)
    n_envs = int(args.n_envs if args.n_envs is not None else rl.n_envs)

    run_dir = make_run_dir(build_run_id("train_rl", tag=args.kind))
    setup_logging(run_dir)
    save_code(run_dir, __file__, cfg=cfg)
    log.info(f"Run dir : {run_dir}")
    log.info(f"kind={args.kind}  n_envs={n_envs}  total_steps={total_steps}  "
             f"decision_every={rl.decision_every}")

    # Optional Weights & Biases logging (mirrors the tensorboard metrics).
    use_wandb = bool(getattr(rl, "wandb", True))
    if use_wandb:
        try:
            import wandb
            wandb.init(project="telescopic_robot", name=run_dir.name,
                       config={"kind": args.kind, "n_envs": n_envs,
                               "total_steps": total_steps, "seed": args.seed,
                               **load_config_dict_section(rl)},
                       sync_tensorboard=True, dir=str(run_dir))
        except Exception as e:      # no network on the node, etc.
            log.warning(f"wandb disabled: {e}")
            use_wandb = False

    thunks = [make_env(cfg, args.kind, r, run_dir, args.seed, max_steps)
              for r in range(n_envs)]
    # single env: stay in-process (one ~1.4 GB process instead of two;
    # the login node has a tight per-user memory cap)
    venv = DummyVecEnv(thunks) if n_envs == 1 else SubprocVecEnv(thunks)
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", venv,
        learning_rate=float(rl.lr),
        n_steps=int(rl.n_steps),
        batch_size=int(rl.batch_size),
        gamma=float(rl.gamma),
        ent_coef=float(rl.ent_coef),
        policy_kwargs=dict(net_arch=[int(x) for x in rl.net]),
        seed=args.seed,
        verbose=1,
        device=str(getattr(rl, "device", "cpu")),
        tensorboard_log=str(run_dir / "tb"),
    )

    ckpt = CheckpointCallback(
        save_freq=max(int(rl.checkpoint_every) // n_envs, 1),
        save_path=str(run_dir / "checkpoints"), name_prefix="ppo",
        save_vecnormalize=True,   # lets eval_rl render intermediate checkpoints
    )
    model.learn(total_timesteps=total_steps, callback=ckpt)

    model.save(run_dir / "checkpoints" / "final")
    venv.save(str(run_dir / "vecnormalize.pkl"))   # obs stats, needed at eval
    log.info(f"model  → {run_dir / 'checkpoints' / 'final.zip'}")
    log.info(f"stats  → {run_dir / 'vecnormalize.pkl'}")
    if use_wandb:
        import wandb
        wandb.finish()
    venv.close()


if __name__ == "__main__":
    main()
