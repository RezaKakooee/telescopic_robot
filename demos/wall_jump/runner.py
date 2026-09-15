"""Wall-jump zig-zag across a wide gap and out onto a wall top, free physics.

Two facing walls 1.0 m apart, both 2.5 m tall. The ball jumps at the
nearer wall from the floor, pushes off it up and across the gap, pushes off
the other wall, and so on, gaining height with every push. Once it is above
the lip it flies over and lands on the wall top, brakes, and stands.
Nothing is pinned or teleported.

The phase machine is the `wall_jump_climb` skill and the loop that drives
it is `skills.runner.run_shaft_climb`. This script only builds the arena
and records what happened. The course needs the stiffer rod of
`configs/rl/wall_jump.yaml`; see the comment there.

    python demos/wall_jump/runner.py --video
    python demos/wall_jump/runner.py --seed 3
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
from radial_sphere.overlay import annotate
from skills.mid_level.shaft_climbing import on_wall_top, shaft_from_boxes
from skills.runner import run_shaft_climb

AXIS = np.array([0.0, 1.0])


def main():
    args = demo_config("wall_jump").knobs

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

    recorder = Recorder(env, "run_wall_jump", tag=f"wall_jump_seed{args.seed}",
                        enabled=args.video, fps=args.fps, every=1,
                        out_name="wall_jump_climb")

    def render_pair():
        if env.renderer is None:
            env.render(camera_name="fixed_angle_close_3d")
        frames = []
        for dist, elev in ((2.2, -5.0), (4.5, -12.0)):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = env.core_body_id
            cam.distance, cam.elevation, cam.azimuth = dist, elev, 180.0   # look along the gap
            env.renderer.update_scene(env.data, camera=cam)
            frames.append(env.renderer.render())
        return np.concatenate(frames, axis=1)

    def frame(state, side, step, z, vz, y):
        if recorder is None or step % args.frame_every:
            return
        label = {"crouch": "crouch on the floor", "takeoff": "jump at the nearer wall",
                 "recentre": "roll back to the middle", "settle": "wait at the middle",
                 "push": f"push off the {'+y' if side > 0 else '-y'} wall",
                 "fly": "fly across the gap", "exit": "eased push, just over the lip",
                 "fly_out": "over the lip", "land": "land on the top",
                 "brake": "brake on the top", "stand": "stopped on top"}.get(state, state)
        if recorder.enabled:
            recorder.add(annotate(render_pair(), f"wall_jump_climb [{state}]",
                              [label, f"height {z:5.2f} m   vz {vz:+4.1f} m/s",
                               f"y {y:+.2f} m",
                               f"t {step * 0.01:5.1f}s"]))

    boxes = np.asarray(scenario.steps, dtype=float)
    shaft = shaft_from_boxes(boxes, axis=AXIS)
    r = run_shaft_climb(env, shaft, skill="wall_jump_climb", frame=frame, trace=args.trace)
    print(f"wall_jump (climb out onto a wall top): gap {2 * shaft.half_width:.1f} m, "
          f"walls {shaft.top:.1f} m, seed {args.seed}")
    print(f"  ascent : peak {r['peak']:.2f} m, pushes {r['pushes']}, cleared the lip: {r['reached']}"
          + (f" at {r['t_up']:.1f}s" if r["t_up"] else "")
          + f", relaunches {r['relaunches']}, recentres {r['recentres']}")
    print(f"  landing: on a wall top: {r['on_top']}"
          + (f" at y {r['top_y']:+.2f} (tops span |y| {shaft.box_lat[0]:.2f}..{shaft.box_lat[1]:.2f}), "
             f"t {r['t_down']:.1f}s" if r["t_down"] else ""))
    ok = r["on_top"] and on_wall_top(env.data.qpos[0:3], shaft)
    print(f"  final  : x {float(env.data.qpos[0]):+.2f} y {float(env.data.qpos[1]):+.2f} "
          f"z {float(env.data.qpos[2]):.2f} -> {'SUCCESS' if ok else 'FAILED'}")
    recorder.close()
    if recorder.enabled:
        print(f"video: {recorder.path}  ({recorder.n_frames / args.fps:.1f}s)")
    env.close()


if __name__ == "__main__":
    main()
