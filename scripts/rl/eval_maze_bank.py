"""Evaluate a banked-maze policy on layouts it trained on, and on layouts it never saw.

This is the question the task bank was built to answer. A policy that scores
well on its training tasks and badly on held-out layouts memorised corridors;
one that scores the same on both navigates. Training against a fresh random
maze every episode cannot tell you which, because there is nothing held back.

    PYTHONPATH=. python scripts/rl/eval_maze_bank.py run=<run dir>
    PYTHONPATH=. python scripts/rl/eval_maze_bank.py run=<run dir> episodes=30
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from radial_sphere.config import load_config, script_config
from radial_sphere.mujoco_steering import MujocoSteeringEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.task_bank import TaskBank


def run_task(model, norm_path, cfg_path, task, max_steps, goal_radius=0.45,
             video_path=None, fps=20, camera="course_dual"):
    """One episode. With `video_path` set it is also recorded, at 1x.

    Real time needs frames from inside the policy step: one policy step is
    100 ms, so rendering once per step caps the video at 10 fps. The inner
    MuJoCo step advances 10 ms, and a frame every 5th of those is 50 ms, which
    written at 20 fps plays back at exactly 1x.
    """
    cfg = load_config(cfg_path)
    cfg.camera.enabled = False
    cfg.scenario.maze.layout_seed = task.layout_seed
    scenario = generate_scenario("maze", cfg, seed=task.endpoint_seed)
    vec = DummyVecEnv([lambda: MujocoSteeringEnv(cfg, scenario=scenario,
                                                 randomize=False,
                                                 max_steps=max_steps + 10)])
    if norm_path.exists():
        vec = VecNormalize.load(str(norm_path), vec)
        vec.training, vec.norm_reward = False, False
    inner = vec.venv.envs[0] if hasattr(vec, "venv") else vec.envs[0]
    writer, frames, counter = None, [], {"n": 0}
    if video_path is not None:
        import imageio.v2 as imageio
        from radial_sphere.overlay import annotate
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video_path), fps=int(fps), codec="libx264",
                                    pixelformat="yuv420p",
                                    ffmpeg_params=["-movflags", "+faststart", "-crf", "23"])
        original = inner.env.step

        def hooked(targets):
            res = original(targets)
            counter["n"] += 1
            if counter["n"] % 5 == 0:
                d = inner.env._nav_distance(inner.env.data.qpos[:2])
                writer.append_data(annotate(
                    inner.render(camera_name=camera),
                    f"held-out maze  layout {task.layout_seed}",
                    [f"never seen during training   route {task.route_m:.1f} m",
                     f"{d:5.2f} m to go through the maze",
                     f"speed {float(np.linalg.norm(inner.env.data.qvel[:2])):4.2f} m/s",
                     f"t {inner.env.data.time:6.1f} s   (real time)"]))
            return res

        inner.env.step = hooked

    obs = vec.reset()
    best, step, ok = float("inf"), 0, False
    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = vec.step(action)
        step += 1
        d = float(infos[0].get("distance", best))
        best = min(best, d)
        if d < goal_radius:
            ok = True
            break
        if done[0]:
            break
    if writer is not None:
        writer.close()
    vec.close()
    return dict(ok=ok, steps=step, closest=best, route=task.route_m)


def summarise(name, rows):
    ok = [r for r in rows if r["ok"]]
    frac = len(ok) / max(len(rows), 1)
    # Progress is the fraction of the route actually closed, which says
    # something even about episodes that never arrive.
    prog = [1.0 - min(r["closest"], r["route"]) / max(r["route"], 1e-9) for r in rows]
    line = (f"{name:<10} {len(ok):>3}/{len(rows):<3} reached "
            f"({frac * 100:>5.1f}%)   route closed {np.mean(prog) * 100:>5.1f}%")
    if ok:
        line += f"   median {int(np.median([r['steps'] for r in ok]))} steps"
    return line, frac, float(np.mean(prog))


def main():
    args = script_config("eval_maze_bank", passthrough=True)
    run_dir = Path(args.run)
    model = PPO.load(str(run_dir / "checkpoints" / "ppo_final.zip"), device="cpu")
    norm = run_dir / "checkpoints" / "vecnormalize_final.pkl"

    print(f"run   : {run_dir.name}")
    print(f"bank  : {args.bank}")
    print(f"budget: {args.max_steps} policy steps per episode "
          f"({args.max_steps * 0.1:.0f} s of sim)\n")

    out = {}
    for split in ("train", "heldout"):
        bank = TaskBank(args.bank, split=split)
        rng = np.random.default_rng(args.seed)
        picks = [bank.tasks[i] for i in
                 rng.choice(len(bank), size=min(args.episodes, len(bank)), replace=False)]
        rows, t0 = [], time.time()
        for i, task in enumerate(picks):
            # The video goes beside the checkpoints it came from, so a run
            # directory carries its own evidence.
            vid = None
            if args.video and split == "heldout" and i < int(args.n_videos):
                vid = (run_dir / "renders" /
                       f"eval_heldout_L{task.layout_seed:02d}_ep{task.endpoint_seed}.mp4")
            rows.append(run_task(model, norm, args.config, task, int(args.max_steps),
                                 video_path=vid, fps=int(args.fps),
                                 camera=str(args.camera)))
            print(f"  {split:<8} {i + 1:>2}/{len(picks)}  layout {task.layout_seed:>3} "
                  f"ep {task.endpoint_seed}  route {task.route_m:>5.1f} m  "
                  f"{'reached' if rows[-1]['ok'] else 'missed '} "
                  f"in {rows[-1]['steps']:>4} steps, closest {rows[-1]['closest']:.2f} m"
                  + (f"   -> {vid.name}" if vid is not None else ""), flush=True)
        line, frac, prog = summarise(split, rows)
        out[split] = (frac, prog)
        print(f"  -> {line}  ({time.time() - t0:.0f}s)\n", flush=True)

    print("=" * 66)
    for split in ("train", "heldout"):
        frac, prog = out[split]
        print(f"{split:<10} reached {frac * 100:>5.1f}%   route closed {prog * 100:>5.1f}%")
    gap = out["train"][0] - out["heldout"][0]
    print(f"\ngeneralisation gap {gap * 100:+.1f} points on success rate")
    print("A gap near zero means it navigates. A large positive gap means it "
          "learned the training layouts.")


if __name__ == "__main__":
    main()
