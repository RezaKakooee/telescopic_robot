"""PPO maze navigation against a fixed bank of tasks.

`train_mujoco_rl.py` generates one scenario per env at construction and lets
`randomize=True` redraw it at every reset. That gives variety but nothing to
hold back, so a finished policy cannot be told apart from one that memorised
the layout it happened to see. This script draws each episode from
`radial_sphere.task_bank`, which reserves a set of layouts that training never
touches.

    PYTHONPATH=. python scripts/rl/train_maze_bank.py
    PYTHONPATH=. python scripts/rl/train_maze_bank.py steps=250000 n_envs=8
    PYTHONPATH=. python scripts/rl/train_maze_bank.py --help
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecMonitor, VecNormalize)

from radial_sphere.config import load_config, script_config
from radial_sphere.mujoco_steering import MujocoSteeringEnv
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir
from radial_sphere.task_bank import BankedMazeEnv, TaskBank


def make_env(config: str, bank_path: str, split: str, rank: int, seed: int,
             max_steps: int, resample_every: int):
    """Thunk for one banked training env.

    `randomize=True` is what lets `reset` rebuild the scene; the wrapper on top
    decides which maze that will be. Without it the env would hold whichever
    scenario it was constructed with forever.
    """
    def _thunk():
        cfg = load_config(config)
        cfg.camera.enabled = False
        bank = TaskBank(bank_path, split=split)
        first = bank.tasks[rank % len(bank)]
        cfg.scenario.maze.layout_seed = first.layout_seed
        scenario = generate_scenario("maze", cfg, seed=first.endpoint_seed)
        env = MujocoSteeringEnv(cfg, scenario=scenario, randomize=True,
                                max_steps=max_steps)
        return BankedMazeEnv(env, bank, seed=seed + rank,
                             resample_every=resample_every)
    return _thunk


def main():
    args = script_config("train_maze_bank", passthrough=True)
    cfg = load_config(args.config)
    rl = cfg.rl

    n_envs = int(args.n_envs if args.n_envs is not None else getattr(rl, "n_envs", 16))
    total_steps = int(args.steps if args.steps is not None else getattr(rl, "total_steps", 1_000_000))
    max_steps = int(getattr(rl, "max_steps", 8000))
    device = str(args.device if args.device is not None else getattr(rl, "device", "cpu"))
    resample_every = int(args.resample_every)

    bank = TaskBank(args.bank, split="train")
    held = TaskBank(args.bank, split="heldout")
    run_dir = Path(make_run_dir(build_run_id("train_maze_bank", tag=f"{n_envs}env")))
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    print(f"run dir      : {run_dir}")
    print(f"config       : {args.config}")
    print(f"bank         : {bank}  (+ {len(held)} held out, never trained on)")
    print(f"rods         : {cfg.robot.rod_mechanism}   "
          f"actions      : {'steer+drive' if rl.include_drive else 'steer only'}")
    print(f"envs         : {n_envs}   total steps: {total_steps:,}   "
          f"episode cap: {max_steps}")
    print(f"resample     : a new task every {resample_every} episode(s)")

    fns = [make_env(args.config, args.bank, "train", i, args.seed, max_steps,
                    resample_every) for i in range(n_envs)]
    vec = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    vec = VecMonitor(vec, filename=str(run_dir / "monitor.csv"))
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", vec, verbose=1, device=device, seed=args.seed,
        learning_rate=float(getattr(rl, "lr", 3e-4)),
        gamma=float(getattr(rl, "gamma", 0.995)),
        n_steps=int(getattr(rl, "n_steps", 512)),
        batch_size=int(getattr(rl, "batch_size", 1024)),
        gae_lambda=float(getattr(rl, "gae_lambda", 0.95)),
        clip_range=float(getattr(rl, "clip_range", 0.2)),
        ent_coef=float(getattr(rl, "ent_coef", 0.01)),
        policy_kwargs=dict(net_arch=list(getattr(rl, "net", [256, 256]))),
        tensorboard_log=str(run_dir / "tb"),
    )

    ckpt = CheckpointCallback(
        save_freq=max(1, int(getattr(rl, "checkpoint_every", 50000)) // n_envs),
        save_path=str(run_dir / "checkpoints"), name_prefix="ppo",
        save_vecnormalize=True)

    started = time.time()
    try:
        model.learn(total_timesteps=total_steps, callback=ckpt, progress_bar=False)
    finally:
        model.save(str(run_dir / "checkpoints" / "ppo_final"))
        vec.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
        vec.close()
        mins = (time.time() - started) / 60.0
        print(f"\nsaved {run_dir / 'checkpoints' / 'ppo_final.zip'}")
        print(f"wall clock {mins:.1f} min")
        print("evaluate on the held-out layouts with scripts/rl/eval_maze_bank.py")


if __name__ == "__main__":
    main()
