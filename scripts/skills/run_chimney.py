"""Climb a chimney and come back down, under free physics.

Two facing walls 0.40 m apart. The ball wall-jumps up -- push off one wall,
fly to the other, push again -- until it reaches the target height, clamps
both walls and hangs there, then slides down on a friction servo and lands.
Nothing is pinned or teleported; the ball rotates and drifts as physics says.

    python scripts/skills/run_chimney.py --video
    python scripts/skills/run_chimney.py --seed 3 --target 3.0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.render import VideoRecorder
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir
from skills import execute_skill
from skills.overlay import annotate

AXIS = np.array([0.0, 1.0])


def climb_chimney(env, *, target_z=None, top=None, box_y=(0.20, 0.70), low_sign=+1,
                  descent_vz=-0.4, hold_steps=150, frame=None,
                  max_steps=3500, push_frac=1.0, trace=False,
                  exit_band=(0.45, 0.20, 0.85), exit_from=0.02):
    """Zig-zag up the chimney, then either

    * `top` given  -- burst out over the lip and land on a box top, brake,
      stop there (the default course);
    * `target_z`   -- clamp both walls at that height, hang, and slide back
      down on the friction servo (the up-and-down demo).

    Returns a dict of measurements. Nothing is pinned; the state machine
    reads position and velocity and reacts.
    """
    q = lambda: env.data.qpos[3:7].copy()
    E = env.max_extend
    state, side, timer = "launch", +1, 0
    peak, hold_z0, ext = 0.0, None, E
    out = {"peak": 0.0, "reached": False, "hold_creep": None, "landed": False,
           "land_vz": None, "t_up": None, "t_down": None, "max_down_vz": 0.0,
           "on_top": False, "top_y": None, "relaunches": 0, "recentres": 0}
    for step in range(max_steps):
        x, y, z = (float(v) for v in env.data.qpos[0:3])
        vx, vy, vz = (float(v) for v in env.data.qvel[0:3])
        peak = max(peak, z)
        kw = {}
        skill = "chimney_climb"

        if target_z is not None and state in ("push", "fly") and z >= target_z:
            state, timer = "hold", 0             # arrived: clamp on
            out["reached"], out["t_up"] = True, step * 0.01
            hold_z0 = z
        # Exit: pushing off the TALL wall, toward the low box, from above its lip.
        if (top is not None and state == "push" and timer == 0 and side == -low_sign
                and z >= top + exit_from):
            state = "exit"
        if (top is not None and state in ("exit", "fly_out", "fly", "push")
                and low_sign * y > box_y[0] + 0.02 and z < top + 0.35 and vz < 0):
            state, timer = "land", 0             # coming down over a box top
            out["reached"], out["t_up"] = True, step * 0.01

        if state == "recentre":
            # Drifted along the shaft toward an open end: roll back to the
            # middle before launching again. `move` at a crawl, then stop.
            timer += 1
            skill = "move" if abs(x) > 0.05 else "stop"
            kw = (dict(d_hat=np.array([-np.sign(x), 0.0]), speed=0.45) if skill == "move"
                  else dict(lin_vel=env.data.qvel[0:2].copy()))
            if abs(x) < 0.05 and abs(vx) < 0.1 and timer > 10:
                state, timer = "launch", 0
        elif state == "launch":
            # From the floor the wall-push rods point at the FLOOR, not the
            # wall: a wall push down there just rattles the ball sideways.
            # Jump straight up first, then start the zig-zag in the air.
            timer += 1
            kw = dict(phase="launch")
            if timer > 14 or (timer > 4 and vz < 0.2 and z > 0.35):
                state, timer = "fly", 0
                side = +1 if y >= 0 else -1
        elif state == "push":
            timer += 1
            if (timer > 4 and side * vy < -0.4) or timer > 25:
                state, timer = "fly", 0
            kw = dict(phase="push", side=side, push_frac=push_frac,
                      x_off=float(env.data.qpos[0]))
        elif state == "fly":
            timer += 1
            if side > 0 and y < -0.02 and vy < 0:
                side, state, timer = -1, "push", 0
            elif side < 0 and y > 0.02 and vy > 0:
                side, state, timer = +1, "push", 0
            elif z < 0.30 and abs(vy) < 0.15 and timer > 20:
                out["relaunches"] += 1
                if abs(x) > 0.12:
                    out["recentres"] += 1
                    state, timer = "recentre", 0
                else:
                    state, timer = "launch", 0      # back on the floor: relaunch
            kw = dict(phase="fly")
        elif state == "exit":
            # The final wall push, made as high as the wall allows. The normal
            # band measured the most lift (steeper bands gave less); the
            # sideways carry it leaves is absorbed by a wide box top.
            timer += 1
            kw = dict(phase="push", side=side, push_lat=exit_band[0],
                      push_z_lo=exit_band[1], push_z_hi=exit_band[2],
                      x_off=float(env.data.qpos[0]))
            if timer > 14 or (timer > 4 and side * vy < -0.25):
                state, timer = "fly_out", 0
                out["exit_v"] = (round(vy, 2), round(vz, 2), round(z, 2))
        elif state == "fly_out":
            timer += 1
            kw = dict(phase="fly")
            if abs(y) < box_y[0] and vz < 0 and z < top:
                state, timer = "fly", 0             # fell back in: resume the zig-zag
        elif state == "land":
            # On the box top: gear underneath, and brake the sideways carry
            # before it runs off the far edge. `stop` is the kickstand.
            timer += 1
            if z > top + 0.10 and abs(vz) < 0.6:
                skill, kw = "stop", dict(lin_vel=env.data.qvel[0:2].copy(), stop_distance=0.12)
            else:
                kw = dict(phase="hold", clamp_ext=0.0, near_floor=True)
            if timer > 120:
                settled = abs(vy) < 0.15 and abs(vz) < 0.15
                on = settled and top + 0.10 < z < top + 0.55 and box_y[0] < low_sign * y < box_y[1]
                out["on_top"], out["top_y"], out["landed"] = on, y, on
                out["t_down"] = step * 0.01
                if on:
                    state, timer = "stand", 0
                elif abs(y) < box_y[0] and z < top:
                    state, timer = "fly", 0
                else:
                    break
        elif state == "hold":
            timer += 1
            kw = dict(phase="hold")
            if timer >= hold_steps:
                out["hold_creep"] = hold_z0 - z
                state, timer, ext = "descend", 0, E
        elif state == "descend":
            # Friction servo: loosen while falling slower than wanted,
            # tighten while faster. Extension is the friction knob.
            ext = float(np.clip(ext - 0.04 * (vz - descent_vz), 0.02, E))
            out["max_down_vz"] = max(out["max_down_vz"], -vz)
            kw = dict(phase="descend", clamp_ext=ext, near_floor=z < 0.45)
            if z < 0.26 and abs(vz) < 0.3:
                state, timer = "stand", 0
                out["landed"], out["land_vz"], out["t_down"] = True, vz, step * 0.01
        else:
            timer += 1
            kw = dict(phase="stand")
            if timer > 60:
                break

        if skill == "chimney_climb":
            env.step(execute_skill(skill, q(), env.dirs_body, E, AXIS, **kw))
        else:
            env.step(execute_skill(skill, q(), env.dirs_body, E, **kw))
        if trace and step % 25 == 0:
            print(f"    t={step*0.01:4.1f} {state:7s} x {float(env.data.qpos[0]):+.2f} y {y:+.2f} z {z:.2f} vz {vz:+.1f}")
        if frame is not None:
            frame(state, side, step, z, vz, y)
    out["peak"] = peak
    return out


def main():
    p = argparse.ArgumentParser(description="Chimney climb, up and down")
    p.add_argument("--config", default="configs/rl/chimney.yaml")
    p.add_argument("--target", type=float, default=None,
                   help="hold at this height and slide back down instead of "
                        "climbing out onto a box top")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--push-frac", type=float, default=1.0,
                   help="push stroke fraction; smaller = gentler, smaller hops")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--exit-band", type=float, nargs=3, default=[0.45, 0.20, 0.85],
                   metavar=("LAT", "ZLO", "ZHI"),
                   help="rod band for the last push; the normal band gives the "
                        "most lift, steeper bands measured LESS")
    p.add_argument("--exit-from", type=float, default=0.02,
                   help="arm the exit push this far ABOVE the low lip")
    p.add_argument("--video", action="store_true")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--frame-every", type=int, default=4)
    args = p.parse_args()

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

    recorder = None
    if args.video:
        run_dir = make_run_dir(build_run_id("run_chimney", f"chimney_seed{args.seed}"))
        out = Path(run_dir) / "renders" / "chimney_climb.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        recorder = VideoRecorder(out, fps=args.fps)

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
                 "push": f"push off the {'+y' if side > 0 else '-y'} wall",
                 "fly": "fly to the other wall", "exit": "burst out over the lip",
                 "fly_out": "over the low box", "land": "land on the box, brake",
                 "hold": "clamp both walls, hang", "descend": "friction slide down",
                 "stand": "stopped on top"}.get(state, state)
        recorder.add(annotate(render_pair(), f"chimney_climb [{state}]",
                              [label, f"height {z:5.2f} m   vz {vz:+4.1f} m/s",
                               f"y {y:+.2f} m",
                               f"t {step * 0.01:5.1f}s"]))

    boxes = np.asarray(scenario.steps, dtype=float)      # [x, y, hx, hy, h]
    low = boxes[int(np.argmin(boxes[:, 4]))]
    top = float(low[4]); low_sign = int(np.sign(low[1]))
    box_y = (float(abs(low[1]) - low[3]), float(abs(low[1]) + low[3]))
    if args.target is None:
        r = climb_chimney(env, top=top, box_y=box_y, low_sign=low_sign, frame=frame,
                          push_frac=args.push_frac, trace=args.trace,
                          exit_band=tuple(args.exit_band), exit_from=args.exit_from)
        print(f"chimney (climb out onto the low box): walls {boxes[:, 4].max():.1f} / "
              f"{top:.1f} m, seed {args.seed}, push_frac {args.push_frac}")
        print(f"  ascent : peak {r['peak']:.2f} m, cleared the lip: {r['reached']}"
              + (f" at {r['t_up']:.1f}s" if r["t_up"] else "")
              + f", relaunches {r['relaunches']}, recentres {r['recentres']}")
        print(f"  exit   : (vy, vz, z) at the last push {r.get('exit_v')}")
        print(f"  landing: on a box top: {r['on_top']}"
              + (f" at y {r['top_y']:+.2f} (box spans {box_y[0]:.2f}..{box_y[1]:.2f}), "
                 f"t {r['t_down']:.1f}s" if r["t_down"] else ""))
        ok = r["on_top"]
    else:
        r = climb_chimney(env, target_z=args.target, frame=frame,
                          push_frac=args.push_frac, trace=args.trace)
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
    if recorder is not None:
        recorder.close()
        print(f"video: {recorder.path}  ({recorder.n_frames / args.fps:.1f}s)")
    env.close()


if __name__ == "__main__":
    main()
