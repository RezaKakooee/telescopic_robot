"""Zig-zag wall-jump up a chimney and out onto the top, under free physics.

Two facing walls 0.40 m apart. The ball wall-jumps up -- push off one wall,
fly to the other, push again -- until it is level with the lower wall's lip,
bursts out over it and lands on the wall top. With `--target` it instead
clamps both walls at that height, hangs, and slides back down on a friction
servo. Nothing is pinned or teleported; the ball rotates and drifts as
physics says.

The phase machine is the `zigzag_climb` skill and the loop that drives it
is `skills.runner.run_shaft_climb`. This script only builds the arena and
records what happened.

    python demos/chimney/runner.py --video
    python demos/chimney/runner.py --seed 3 --target 3.0
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from radial_sphere.config import load_config, demo_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.demo import Recorder
from skills.mid_level.shaft_climbing import Shaft, ZigzagTuning, on_wall_top, shaft_from_boxes
from skills.runner import run_shaft_climb
from radial_sphere.overlay import annotate

AXIS = np.array([0.0, 1.0])


def main():
    args = demo_config("chimney").knobs

    cfg = load_config(args.config)
    scenario = generate_scenario("chimney", cfg, seed=1)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=1_000_000)
    env.reset(seed=1)
    env.data.qpos[0:3] = [0.0, 0.0, 0.20]
    env.data.qvel[:] = 0
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        qq = rng.normal(size=4)
        env.data.qpos[3:7] = qq / np.linalg.norm(qq)
    mujoco.mj_forward(env.model, env.data)
    for _ in range(40):
        env.step(np.zeros(60, dtype=np.float32))

    # Shared plumbing; the climb phase machine is the skill's own.
    recorder = Recorder(env, "run_chimney", tag=f"chimney_seed{args.seed}",
                        enabled=args.video, fps=args.fps, every=1,
                        out_name="chimney_climb")

    def render_pair():
        if env.renderer is None:
            env.render(camera_name="fixed_angle_close_3d")
        frames = []
        for dist, elev in ((1.35, -5.0), (2.6, -12.0)):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.core_body_id
            cam.distance, cam.elevation, cam.azimuth = dist, elev, 180.0   # look along the shaft
            env.renderer.update_scene(env.data, camera=cam)
            frames.append(env.renderer.render())
        return np.concatenate(frames, axis=1)

    def frame(state, side, step, z, vz, y):
        if recorder is None or step % args.frame_every:
            return
        label = {"launch": "jump up off the floor", "recentre": "roll back to mid-shaft",
                 "settle": "wait at mid-shaft",
                 "push": f"push off the {'+y' if side > 0 else '-y'} wall",
                 "fly": "fly to the other wall", "exit": "burst out over the lip",
                 "fly_out": "over the low wall top", "land": "land on the top",
                 "brake": "brake on the top",
                 "hold": "clamp both walls, hang", "descend": "friction slide down",
                 "stand": "stopped on top"}.get(state, state)
        if recorder.enabled:
            recorder.add(annotate(render_pair(), f"zigzag_climb [{state}]",
                              [label, f"height {z:5.2f} m   vz {vz:+4.1f} m/s",
                               f"y {y:+.2f} m",
                               f"t {step * 0.01:5.1f}s"]))

    tuning = ZigzagTuning(push_frac=args.push_frac, exit_band=tuple(args.exit_band),
                          exit_from=args.exit_from)
    boxes = np.asarray(scenario.steps, dtype=float)
    if args.target is None:
        shaft = shaft_from_boxes(boxes, axis=AXIS)
        top, low_sign, box_y = shaft.top, shaft.low_sign, shaft.box_lat
        r = run_shaft_climb(env, shaft, skill="zigzag_climb", frame=frame, tuning=tuning, trace=args.trace)
        print(f"chimney (climb out onto the low wall top): walls {boxes[:, 4].max():.1f} / "
              f"{top:.1f} m, seed {args.seed}, push_frac {args.push_frac}")
        print(f"  ascent : peak {r['peak']:.2f} m, cleared the lip: {r['reached']}"
              + (f" at {r['t_up']:.1f}s" if r["t_up"] else "")
              + f", relaunches {r['relaunches']}, recentres {r['recentres']}")
        print(f"  exit   : (vy, vz, z) at the last push {r.get('exit_v')}")
        print(f"  landing: on the wall top: {r['on_top']}"
              + (f" at y {r['top_y']:+.2f} (top spans {box_y[0]:.2f}..{box_y[1]:.2f}), "
                 f"t {r['t_down']:.1f}s" if r["t_down"] else ""))
        ok = r["on_top"] and on_wall_top(env.data.qpos[0:3], shaft)
    else:
        shaft = Shaft(axis=tuple(AXIS), target_z=float(args.target))
        r = run_shaft_climb(env, shaft, skill="zigzag_climb", frame=frame, tuning=tuning, trace=args.trace)
        print(f"chimney (hold and descend): target {args.target:.1f} m, seed {args.seed}")
        print(f"  ascent : peak {r['peak']:.2f} m, target reached: {r['reached']}"
              + (f" at {r['t_up']:.1f}s" if r["t_up"] else ""))
        print(f"  hold   : creep {r['hold_creep']:.3f} m over 1.5 s" if r["hold_creep"] is not None
              else "  hold   : not reached")
        print(f"  descent: landed {r['landed']}"
              + (f" at {r['t_down']:.1f}s, touchdown vz {r['land_vz']:+.2f} m/s, "
                 f"fastest {r['max_down_vz']:.2f} m/s" if r["landed"] else ""))
        ok = r["reached"] and r["landed"] and abs(float(env.data.qpos[1])) < 0.15
    print(f"  final  : x {float(env.data.qpos[0]):+.2f} y {float(env.data.qpos[1]):+.2f} "
          f"z {float(env.data.qpos[2]):.2f} -> {'SUCCESS' if ok else 'FAILED'}")
    recorder.close()
    if recorder.enabled:
        print(f"video: {recorder.path}  ({recorder.n_frames / args.fps:.1f}s)")
    env.close()


if __name__ == "__main__":
    main()
