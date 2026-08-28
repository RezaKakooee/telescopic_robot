"""Show the parametric gait: ask for a speed in m/s, ask for an angle in radians.

Three parts.

1. **Speed ladder.** Command 0.4 -> 2.4 m/s in steps, straight ahead. The
   overlay carries both the asked and the achieved speed, so the calibration
   is visible rather than claimed.

2. **Turn sweep.** Hold one speed and sweep `turn` smoothly from -90 to +90
   degrees against a fixed world reference. The path bends continuously,
   which is the thing a set of discrete left/right skills cannot do.

3. **Adaptive forward.** Use the ball's OWN velocity as the reference and
   hold a constant `turn` against it. Because the robot has no front, its
   heading is whatever it is currently doing, and a fixed offset from that
   traces a circle. This is the isotropy argument made visible.

    python scripts/skills/run_parametric_demo.py --video
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.render import VideoRecorder
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir
from skills import execute_skill
from skills.locomotion import MAX_SPEED, MIN_SPEED

X = np.array([1.0, 0.0])


def main():
    p = argparse.ArgumentParser(description="Parametric gait demo")
    p.add_argument("--config", default="configs/rl/skill_course.yaml")
    p.add_argument("--video", action="store_true")
    p.add_argument("--camera", default="fixed_close_dual")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--frame-every", type=int, default=4)
    p.add_argument("--seconds-per-speed", type=float, default=4.0)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.scenario.goal.x_range = [0.0, 0.0]
    cfg.scenario.goal.y_range = [-400.0, -400.0]
    cfg.floor.square_m = 0.5
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False,
                                max_steps=1_000_000)
    env.reset(seed=42)

    recorder = None
    if args.video:
        run_dir = make_run_dir(build_run_id("run_parametric", "gait"))
        out = Path(run_dir) / "renders" / "parametric_gait.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        recorder = VideoRecorder(out, fps=args.fps)

    from skills.overlay import annotate
    tick = {"n": 0}
    # The chase camera cannot show the shape of a path, so log it and draw it.
    track = {"speed": [], "sweep": [], "own": []}
    leg = {"name": "speed"}

    def frame(title, lines):
        tick["n"] += 1
        track[leg["name"]].append(env.data.qpos[0:2].copy())
        if recorder is None or tick["n"] % args.frame_every:
            return
        recorder.add(annotate(env.render(camera_name=args.camera), title, lines))

    def speed_now():
        return float(np.linalg.norm(env.data.qvel[0:2]))

    def q():
        return env.data.qpos[3:7].copy()

    steps_per = int(args.seconds_per_speed * 100)   # 1 env step = 0.01 s

    # ---- 1. speed ladder ---------------------------------------------
    print("part 1 - speed ladder")
    print(f"{'asked m/s':>10}{'achieved':>10}{'error':>9}")
    for asked in [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]:
        samples = []
        for i in range(steps_per):
            env.step(execute_skill("move", q(), env.dirs_body, env.max_extend,
                                   d_hat=X, speed=asked, turn=0.0))
            if i > steps_per * 0.5:
                samples.append(speed_now())
            frame(f"move(speed={asked:.1f} m/s)",
                  ["one gait, speed asked in m/s",
                   f"asked     {asked:5.2f} m/s",
                   f"achieved  {speed_now():5.2f} m/s",
                   f"turn       0.0 deg"])
        got = float(np.mean(samples))
        print(f"{asked:>10.2f}{got:>10.2f}{got - asked:>+9.2f}")

    # ---- 2. turn sweep ------------------------------------------------
    leg["name"] = "sweep"
    print("\npart 2 - continuous turn sweep, -90 to +90 deg at 1.2 m/s")
    sweep_steps = 1400
    for i in range(sweep_steps):
        frac = i / (sweep_steps - 1)
        deg = -90.0 + 180.0 * frac
        env.step(execute_skill("move", q(), env.dirs_body, env.max_extend,
                               d_hat=X, speed=1.2, turn=np.radians(deg)))
        frame("move(turn = sweeping)",
              ["turn is continuous, not four buttons",
               f"turn      {deg:+6.1f} deg",
               f"speed      {speed_now():5.2f} m/s",
               "reference: fixed world +x"])
    print(f"  swept -90 to +90 deg, ended at {float(env.data.qpos[0]):.1f}, "
          f"{float(env.data.qpos[1]):.1f}")

    # ---- 3. adaptive forward: reference = own velocity ----------------
    leg["name"] = "own"
    print("\npart 3 - reference is the ball's OWN velocity, constant turn")
    for _ in range(220):                       # get rolling first
        env.step(execute_skill("move", q(), env.dirs_body, env.max_extend,
                               d_hat=X, speed=1.4, turn=0.0))
        frame("building speed", ["about to hand the reference to the ball itself",
                                 f"speed      {speed_now():5.2f} m/s", "", ""])
    p0 = env.data.qpos[0:2].copy()
    turn_deg = 22.0
    for _ in range(2600):
        v = env.data.qvel[0:2].copy()
        n = float(np.linalg.norm(v))
        own = v / n if n > 1e-3 else X          # the ball's own heading
        env.step(execute_skill("move", q(), env.dirs_body, env.max_extend,
                               d_hat=own, speed=1.4, turn=np.radians(turn_deg)))
        frame("move(reference = own velocity)",
              ["the robot has no front, so forward is",
               "whatever it is already doing",
               f"turn      {turn_deg:+5.1f} deg off its own heading",
               f"speed      {speed_now():5.2f} m/s"])
    d = env.data.qpos[0:2].copy() - p0
    print(f"  held {turn_deg:+.0f} deg off its own heading for 26 s -> "
          f"net displacement {float(np.linalg.norm(d)):.2f} m (a closed curve)")

    # ---- trajectory figure -------------------------------------------
    if args.video:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, key, ttl in zip(axes, ["speed", "sweep", "own"], [
                "1. speed ladder\n0.4 -> 2.4 m/s, straight",
                "2. turn swept -90 to +90 deg\nagainst a fixed world reference",
                f"3. constant {turn_deg:+.0f} deg off its OWN heading\nno fixed reference at all"]):
            a = np.asarray(track[key])
            if len(a) == 0:
                continue
            ax.plot(a[:, 0], a[:, 1], lw=1.6)
            ax.plot(a[0, 0], a[0, 1], "o", ms=6, label="start")
            ax.plot(a[-1, 0], a[-1, 1], "s", ms=6, label="end")
            ax.set_title(ttl, fontsize=10)
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path = Path(recorder.path).parent / "parametric_paths.png"
        fig.savefig(fig_path, dpi=110)
        print(f"paths: {fig_path}")

    if recorder is not None:
        recorder.close()
        print(f"\nvideo: {recorder.path}  ({recorder.n_frames / args.fps:.1f}s)")
    env.close()


if __name__ == "__main__":
    main()
