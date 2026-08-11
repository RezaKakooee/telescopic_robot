"""Random agent for the RadialSphere env — samples actions uniformly.

A baseline / sanity-check policy: every step each bar gets a random normalised
extension target from the action space.

All outputs go to a timestamped run dir under ``storage_local/`` (videos in
``renders/``, a code+config snapshot in ``code/``).

Usage:
    python random_agent.py                  # 1 episode, video saved
    python random_agent.py --episodes 3
    python random_agent.py --no-video       # stats only
"""
from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse

import numpy as np
import rootutils
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.utils.obs_utils import ObsSaver  # noqa: E402

from radial_sphere import (  # noqa: E402
    RadialSphereEnv,
    Scenario,
    generate_scenario,
    load_config,
    make_run_dir,
    save_code,
)


def run_episode(env, *, seed=None, obs_saver=None, frame_every=17):
    """Roll out one episode with random actions. Returns (return, steps, reached)."""
    _obs, info = env.reset(seed=seed)
    if obs_saver is not None:
        obs_saver.add(env.state)
    total_r, step = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        _obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        step += 1
        if obs_saver is not None and step % frame_every == 0:
            obs_saver.add(env.state)
    return total_r, info["step"], terminated


def main():
    p = argparse.ArgumentParser(description="Random-action agent for RadialSphere")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=400,
                   help="override env.max_steps from config.yaml")
    p.add_argument("--frame-every", type=int, default=17)
    p.add_argument("--config", default=None,
                   help="path to a config.yaml (default: project config.yaml)")
    p.add_argument("--scenario", default=None,
                   help="path to a scenario JSON (from scenario_generator.py)")
    p.add_argument("--kind", default=None,
                   help="generate a scenario of this kind instead (path|goal)")
    p.add_argument("--no-video", dest="video", action="store_false",
                   help="disable video saving (on by default)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.scenario:
        scenario = Scenario.load(args.scenario)
    elif args.kind:
        scenario = generate_scenario(args.kind, cfg, seed=args.seed)
    else:
        scenario = None   # env default (config scenario.kind)

    run_dir = make_run_dir("rnd")
    save_code(run_dir, __file__)   # snapshot code + config for reproducibility
    log.info(f"Run dir : {run_dir}")

    env = RadialSphereEnv(cfg, scenario=scenario, max_steps=args.max_steps,
                          output_dir=run_dir, seed=args.seed)

    returns, steps, successes = [], [], []
    for ep in range(args.episodes):
        obs_saver = ObsSaver(
            video_path=str(run_dir / "renders" / f"ep_{ep + 1:03d}.mp4")
        ) if args.video else None
        ret, n, ok = run_episode(env, seed=args.seed + ep, obs_saver=obs_saver,
                                 frame_every=args.frame_every)
        returns.append(ret)
        steps.append(n)
        successes.append(ok)
        log.info(f"ep {ep + 1:3d}  return={ret:+.3f}  steps={n}  reached={ok}")
        if obs_saver is not None:
            obs_saver.save()

    log.info(f"mean return : {np.mean(returns):+.3f} ± {np.std(returns):.3f}")
    log.info(f"mean steps  : {np.mean(steps):.0f}")
    log.info(f"success rate: {np.mean(successes) * 100:.1f}%")
    env.close()


if __name__ == "__main__":
    main()
