"""Fine-Tuning RL starting from Imitation Learning (BC) Pre-trained Weights.

Loads the pre-trained Hierarchical Imitation Policy weights and fine-tunes with PPO/SAC
on native MuJoCo physics across diverse procedural mazes.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import rootutils
import torch
import torch.nn as nn

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


def make_env_thunk(cfg, rank: int, seed: int, max_steps: int, mode: str = "lowlevel"):
    def _thunk():
        sc = generate_scenario("maze", cfg, seed=seed + rank)
        if mode == "lowlevel":
            return MujocoLowLevelEnv(cfg, scenario=sc, randomize=True, max_steps=max_steps)
        return MujocoSteeringEnv(cfg, scenario=sc, randomize=True, max_steps=max_steps)
    return _thunk


def load_bc_weights_into_sb3(sb3_model, bc_checkpoint_path: Path, device: str = "cpu"):
    """Transfers pre-trained BC backbone and actor weights into Stable-Baselines3 actor-critic policy."""
    log.info(f"Loading pre-trained BC weights from: {bc_checkpoint_path}")
    checkpoint = torch.load(str(bc_checkpoint_path), map_location=device)
    bc_state = checkpoint.get("model_state_dict", checkpoint)

    policy_state = sb3_model.policy.state_dict()
    transferred = 0

    # Match weights by tensor shape compatibility
    for k, v in bc_state.items():
        # Match encoder layers into mlp_extractor / features_extractor
        for target_k in policy_state.keys():
            if target_k in k or k in target_k:
                if policy_state[target_k].shape == v.shape:
                    policy_state[target_k].copy_(v)
                    transferred += 1
                    break

    sb3_model.policy.load_state_dict(policy_state)
    log.info(f"Successfully transferred {transferred} weight tensors from BC model into RL policy!")


def main():
    p = argparse.ArgumentParser(description="Fine-tune RL starting from BC Pretrained Checkpoint")
    p.add_argument("--bc-ckpt", default="storage_local/imitation_models/bc_hierarchical_best.pt",
                   help="Path to pre-trained BC checkpoint")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--mode", choices=["steering", "lowlevel"], default="lowlevel")
    p.add_argument("--config-name", default="maze_level3_large_active_braking_multiaxis")
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)  # Lower learning rate for fine-tuning
    p.add_argument("--seed", type=int, default=7001)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config_cli(name=args.config_name)
    max_steps = int(getattr(cfg.rl, "max_steps", 4000))
    run_dir = make_run_dir(build_run_id("finetune_rl_from_bc", tag=f"{args.algo}__{args.mode}"))
    setup_logging(run_dir)
    save_code(run_dir, __file__, cfg=cfg)

    log.info(f"RL Fine-Tuning Run dir: {run_dir}")
    log.info(f"Algorithm: {args.algo.upper()} | Mode: {args.mode} | Steps: {args.steps:,} | Envs: {args.n_envs}")

    env_fns = [make_env_thunk(cfg, i, args.seed, max_steps, mode=args.mode) for i in range(args.n_envs)]
    vec = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
    vec = VecMonitor(vec)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    net_arch = [256, 256, 256]

    if args.algo == "sac":
        model = SAC(
            "MlpPolicy",
            vec,
            learning_rate=args.lr,
            buffer_size=100000,
            learning_starts=2000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=dict(net_arch=dict(pi=net_arch, qf=net_arch)),
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(run_dir / "tb"),
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=args.lr,
            n_steps=512,
            batch_size=1024,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            policy_kwargs=dict(net_arch=dict(pi=net_arch, vf=net_arch)),
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(run_dir / "tb"),
        )

    # Transfer BC weights
    bc_path = Path(args.bc_ckpt)
    if bc_path.exists():
        load_bc_weights_into_sb3(model, bc_path, device=args.device)
    else:
        log.warning(f"BC checkpoint not found at {bc_path}, training from scratch.")

    ckpt_cb = CheckpointCallback(
        save_freq=max(10000 // args.n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix=args.algo,
        save_vecnormalize=True,
    )

    log.info("Starting Fine-Tuning RL training loop...")
    model.learn(total_timesteps=args.steps, callback=ckpt_cb, progress_bar=False)

    model.save(str(run_dir / "checkpoints" / f"{args.algo}_final.zip"))
    vec.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    vec.close()

    log.info(f"RL Fine-Tuning complete! Saved to {run_dir}")


if __name__ == "__main__":
    main()
