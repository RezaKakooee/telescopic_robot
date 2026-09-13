"""Score the hierarchical BC policy on the same task bank the PPO runs use.

`docs/imitation_and_rl_finetuning_report.md` reports `bc_hierarchical_best.pt`
solving 6 of 6 unseen maze topologies with 100 % success and zero wall
contacts. That was 2026-08-22, which is before commit `ead03dd` switched the
default rod to `multi_stage` and before the actuator and latency models were
fixed. The report's config names no `rod_mechanism` at all, so it silently
trained and evaluated on `single_stage`.

So the number may or may not still hold. This runs the same checkpoint on the
same held-out layouts the PPO policies are scored on, on both rod builds, so
the comparison is like for like.

Replaying it faithfully takes care, because the 163D observation it was cloned
against is not any environment's observation. `scripts/data/generate_maze_demonstrations.py`
synthesised it by hand, and two details matter:

* the 66-dim "previous action" block is hard-coded to zeros in the dataset, so
  66 of the model's 163 inputs were dead throughout training. They stay zero
  here, because feeding real values would be a different input distribution
  than the one it learned on;
* the lidar is 24 rays at 3.0 m in the goal frame, not the 3.5 m the steering
  configs use.

What the model was cloned from is also worth stating plainly. The labels are
`bar_targets` -- the hand-written analytic gait -- driven by an expert PPO
steering policy. So "100 % on 6 unseen mazes" is a network reproducing that
pair, not a policy that discovered navigation on its own.

    PYTHONPATH=. python scripts/rl/eval_bc_maze_bank.py
    PYTHONPATH=. python scripts/rl/eval_bc_maze_bank.py rods=single_stage
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

from radial_sphere.config import load_config, script_config
from radial_sphere.controller import desired_direction
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.task_bank import TaskBank

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "imitation"))
from train_bc import HierarchicalImitationPolicy  # noqa: E402


def build_cfg(cfg_path, task, rods, bank_cfg_path):
    """The BC maze config, aligned to the bank so tasks are the same ones.

    The wall grid already matches (level 3, 1.5 m cells, 7x6, 35-55 m routes),
    so a layout seed draws the same labyrinth either way. Only the endpoint
    range differs, and that decides which start/goal pairs are candidates, so
    it is copied across or the same seed would pick a different pair.
    """
    cfg = load_config(cfg_path)
    cfg.camera.enabled = False
    bank_cfg = load_config(bank_cfg_path)
    cfg.scenario.maze.endpoint_min_route = bank_cfg.scenario.maze.endpoint_min_route
    cfg.scenario.maze.endpoint_max_route = bank_cfg.scenario.maze.endpoint_max_route
    cfg.scenario.maze.layout_seed = task.layout_seed
    cfg.scenario.maze.random_endpoints = True
    cfg.robot.rod_mechanism = rods          # never left to the code default
    cfg.robot.appearance_theme = "realistic"
    # The BC head emits 60 rod extensions, which is what the demonstration
    # dataset recorded. `rl.cpg_residual` prepends 6 CPG parameters and makes
    # the action space 66D, so it has to be off or the shapes do not meet.
    cfg.rl.cpg_residual = False
    return cfg


def synth_obs(raw, goal_dir, goal_xy):
    """Rebuild the 163D vector the BC model was cloned against.

    Laid out exactly as `generate_maze_demonstrations.py` wrote it:
    quat 4, velocity in the goal frame 3, angular velocity 3, normalised joint
    positions 60, a 66-wide previous-action block that the dataset always left
    at zero, relative goal 2, normalised distance 1, and 24 lidar rays.
    """
    q = raw.data.qpos[3:7].astype(np.float32)
    v = raw.data.qvel[:3]
    g = np.asarray(goal_dir, dtype=np.float32)
    v_fwd = float(v[0] * g[0] + v[1] * g[1])
    v_lat = float(g[0] * v[1] - g[1] * v[0])
    ball_xy = raw.data.qpos[:2]
    rel_goal = (np.asarray(goal_xy)[:2] - ball_xy).astype(np.float32)
    norm_dist = float(np.linalg.norm(rel_goal) / max(raw.path_length, 1.0))
    norm_joint = (raw.data.qpos[7:7 + raw.n_bars] / raw.max_extend).astype(np.float32)
    lidar = raw.raycast_lidar(n_rays=24, max_range=3.0, g=g).astype(np.float32)
    return np.concatenate([
        q,
        np.array([v_fwd, v_lat, v[2]], dtype=np.float32),
        raw.data.qvel[3:6].astype(np.float32),
        norm_joint,
        np.zeros(66, dtype=np.float32),      # dead in the dataset, dead here
        rel_goal,
        np.array([norm_dist], dtype=np.float32),
        lidar,
    ]).astype(np.float32)


def run_task(model, cfg_path, bank_cfg_path, task, rods, max_steps,
             goal_radius=0.45, video_path=None, fps=20, camera="course_dual"):
    cfg = build_cfg(cfg_path, task, rods, bank_cfg_path)
    scenario = generate_scenario("maze", cfg, seed=task.endpoint_seed)
    # The raw env, driven with rod targets. The BC head emits 60 extensions,
    # which is what `MujocoRadialSphereEnv.step` takes; `MujocoLowLevelEnv`
    # would add its own observation and action conventions on top.
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False,
                                max_steps=max_steps + 10)
    env.reset(seed=task.endpoint_seed)
    goal_xy = np.asarray(scenario.goal, dtype=float)[:2]
    path_pts = np.asarray(scenario.path_pts, dtype=np.float64)

    writer, n_inner = None, {"i": 0}
    if video_path is not None:
        import imageio.v2 as imageio
        from radial_sphere.overlay import annotate
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video_path), fps=int(fps), codec="libx264",
                                    pixelformat="yuv420p",
                                    ffmpeg_params=["-movflags", "+faststart", "-crf", "23"])

    best, step, ok, hits = float("inf"), 0, False, 0
    while step < max_steps:
        g, _ = desired_direction(env.data.qpos[:2], path_pts,
                                 float(getattr(cfg.controller, "lookahead", 0.9)))
        obs = synth_obs(env, g, goal_xy)
        with torch.no_grad():
            _high, low = model(torch.as_tensor(obs).unsqueeze(0))
        _o, _r, term, trunc, info = env.step(low.squeeze(0).numpy())
        step += 1
        d = float(info.get("distance", best))
        best = min(best, d)
        hits += int(bool(info.get("wall_contact", False)))
        if writer is not None and step % 5 == 0:
            writer.append_data(annotate(
                env.render(camera_name=camera),
                f"BC policy, held-out maze  layout {task.layout_seed}",
                [f"bc_hierarchical_best.pt   rods: {rods}",
                 f"route {task.route_m:.1f} m   {d:5.2f} m to go   wall hits {hits}",
                 f"speed {float(np.linalg.norm(env.data.qvel[:2])):4.2f} m/s",
                 f"t {env.data.time:6.1f} s"]))
        if d < goal_radius:
            ok = True
            break
        if term or trunc:
            break
    if writer is not None:
        writer.close()
    env.close()
    return dict(ok=ok, steps=step, closest=best, route=task.route_m, hits=hits)


def main():
    args = script_config("eval_bc_maze_bank", passthrough=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = HierarchicalImitationPolicy(obs_dim=int(ckpt["obs_dim"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"checkpoint : {args.checkpoint}")
    print(f"             obs {ckpt['obs_dim']}D, epoch {ckpt['epoch']}, "
          f"val loss {ckpt['val_loss']:.5f}")
    print(f"config     : {args.config}")
    print(f"budget     : {args.max_steps} steps\n")

    for rods in [r.strip() for r in str(args.rods).split(",")]:
        bank = TaskBank(args.bank, split=args.split)
        rng = np.random.default_rng(args.seed)
        picks = [bank.tasks[i] for i in
                 rng.choice(len(bank), size=min(args.episodes, len(bank)), replace=False)]
        rows, t0 = [], time.time()
        for i, task in enumerate(picks):
            vid = None
            if args.video and i < int(args.n_videos):
                vid = (Path(args.out_dir) /
                       f"bc_{rods}_L{task.layout_seed:02d}_ep{task.endpoint_seed}.mp4")
            r = run_task(model, args.config, args.bank_config, task, rods,
                         int(args.max_steps), video_path=vid, fps=int(args.fps),
                         camera=str(args.camera))
            rows.append(r)
            print(f"  {rods:<13} {i + 1:>2}/{len(picks)}  layout {task.layout_seed:>3} "
                  f"ep {task.endpoint_seed}  route {task.route_m:>5.1f} m  "
                  f"{'reached' if r['ok'] else 'missed '} in {r['steps']:>4} steps, "
                  f"closest {r['closest']:5.2f} m, wall hits {r['hits']:>4}"
                  + (f"   -> {vid.name}" if vid else ""), flush=True)
        n = sum(x["ok"] for x in rows)
        prog = np.mean([1.0 - min(x["closest"], x["route"]) / max(x["route"], 1e-9)
                        for x in rows])
        print(f"  => {rods}: {n}/{len(rows)} reached ({n / len(rows) * 100:.1f}%)   "
              f"route closed {prog * 100:.1f}%   "
              f"total wall hits {sum(x['hits'] for x in rows)}   "
              f"({time.time() - t0:.0f}s)\n", flush=True)


if __name__ == "__main__":
    main()
