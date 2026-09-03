"""Wall of Death: spiral up the inside of the drome, and back down again.

Run::

    python scripts/skills/run_motordrome_wall_of_death.py --seconds 75 --video
    python scripts/skills/run_motordrome_wall_of_death.py --seconds 75 \
        --descend-after 45 --video

What it does, in one line: drive round, and let the speed decide the height.

With ``--descend-after`` the run has three parts. It climbs, holds the high
orbit, then winds the spiral back in and parks itself on the flat floor.
Coming down is not the climb run backwards: on the way up the bank has to be
earned, on the way down the speed has to be given away, and a bank only lets
go of speed as fast as the robot can shed it. So the descent asks for less
speed than the circle can hold and uses the leading-sector kickstand from
:func:`~skills.locomotion.stop` to get rid of the rest.

The physics is in :mod:`skills.wall_of_death`; this file only drives it and
draws the picture. The two calls that matter are per step:

    r_cmd = advance_radius(r_cmd, r, speed, bowl)     # how wide to circle
    targets, info = wall_of_death(..., r_cmd=r_cmd)   # one step of the gait

Nothing is timed. The radius opens when the measured speed can hold the next
circle out, and closes again when it cannot, so the same script works whatever
the robot's speed turns out to be.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir
from skills.overlay import annotate
from skills.locomotion import stop as stop_skill
from skills.wall_of_death import (
    Bowl, advance_radius, descend_radius, surface_frame, wall_of_death,
)


def run(
    seconds: float = 60.0,
    seed: int = 42,
    record_video: bool = True,
    config: str = "configs/rl/motordrome.yaml",
    open_rate: float = 0.10,
    steer_gain: float = 0.30,
    max_steer: float = 0.35,
    camera: str = "rim",
    fps: int = 25,
    frame_every: int = 4,
    descend_after: float | None = None,
    close_rate: float = 0.20,
    brake_gain: float = 1.5,
    descend_speed: float = 0.80,
):
    cfg = load_config(config)
    OmegaConf.set_struct(cfg, False)
    scenario = generate_scenario("motordrome", cfg, seed=seed)
    md = scenario.motordromes[0]
    bowl = Bowl.from_motordrome(md)
    rim_r, wall_top, mu = float(md[3]), float(md[5]), float(md[6])

    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10 ** 7)
    env.reset(seed=seed)
    env.data.qpos[0] = 0.55
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.16 + 0.20 * env.max_extend
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print(f"=== Wall of Death ===")
    print(f"  bowl      rim r {rim_r:.2f} m, depth {bowl.rim_z:.2f} m, "
          f"steepest bank {np.degrees(np.arctan(bowl.tan_bank.max())):.0f} deg")
    print(f"  boards    friction {mu:.2f}, wall top {wall_top:.2f} m")
    print(f"  robot     stroke {env.max_extend:.2f} m")
    print(f"  the bank carries {bowl.hold_speed(rim_r):.2f} m/s at the rim")

    writer = out_video = None
    if record_video:
        run_dir = make_run_dir(build_run_id("motordrome", "wall_of_death"))
        out_video = Path(run_dir) / "renders" / "motordrome_wall_of_death.mp4"
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out_video), fps=fps, codec="libx264")

    steps = int(seconds * 100)
    descend_step = int(descend_after * 100) if descend_after else None
    if descend_step:
        print(f"  coming back down from {descend_after:.0f} s")
    phase = "climb"
    r_cmd = 0.55
    laps = 0.0
    prev_th = 0.0
    peak_z = 0.0
    peak_v = 0.0
    hist = []

    for i in range(steps):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        n_hat, _dist = surface_frame(env.model, env.data, pos)

        if descend_step and i >= descend_step and phase == "climb":
            phase = "descend"

        if phase == "park":
            # On the flat floor and slow: brake to a halt with the stance skill.
            targets = stop_skill(env.data.qpos[3:7].copy(), env.dirs_body,
                                 env.max_extend, lin_vel=vel)
            info = dict(r=float(np.hypot(pos[0], pos[1])), z=float(pos[2]),
                        r_cmd=r_cmd, speed=float(np.linalg.norm(vel)),
                        hold_speed=0.0, bank_deg=0.0, v_t=0.0)
        else:
            # Coming down, ask for a little less than the commanded circle can
            # hold on the bank alone. The climb's throttle asks for a little
            # more, and a little more is exactly what keeps the robot up there.
            want = None
            if phase == "descend":
                want = descend_speed * bowl.hold_speed(r_cmd)
            targets, info = wall_of_death(
                env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, pos, vel,
                r_cmd=r_cmd, bowl=bowl, normal=n_hat, wall_radius=rim_r, ccw=True,
                steer_gain=steer_gain, max_steer=max_steer, speed=want,
                brake_gain=(brake_gain if phase == "descend" else 0.0),
            )
            if phase == "climb":
                r_cmd = advance_radius(r_cmd, info["r"], info["speed"], bowl,
                                       open_rate=open_rate)
            else:
                r_cmd = descend_radius(r_cmd, info["r"], info["speed"], bowl,
                                       close_rate=close_rate)
                # Hand over once it is back on the flat and not going anywhere.
                if info["r"] <= bowl.floor_r + 0.10 and info["z"] < 0.35:
                    phase = "park"
        env.step(targets)

        th = float(np.arctan2(pos[1], pos[0]))
        if i:
            laps += ((th - prev_th + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi)
        prev_th = th
        peak_z = max(peak_z, info["z"])
        peak_v = max(peak_v, info["speed"])
        hist.append((info["z"], info["speed"], info["r"], float(vel[2]), phase))

        if writer is not None and i % frame_every == 0:
            writer.append_data(_frame(env, info, th, abs(laps), bowl,
                                      camera, rim_r, wall_top, pos))
        if i % 500 == 0:
            print(f"  t={i/100:5.1f}s [{phase:7s}] r={info['r']:4.2f}  cmd={r_cmd:4.2f}  "
                  f"z={info['z']:4.2f}  v={info['speed']:4.2f}  "
                  f"bank={info['bank_deg']:4.1f}deg  holds={info['hold_speed']:4.2f}")

    if writer is not None:
        writer.close()
    env.close()

    tail = hist[-800:]
    z_tail = [h[0] for h in tail]
    v_tail = [h[1] for h in tail]
    print("\n--- summary ---")
    print(f"  laps            {abs(laps):.1f}")
    print(f"  peak height     {peak_z:.2f} m   (bowl is {bowl.rim_z:.2f} m deep)")
    print(f"  peak speed      {peak_v:.2f} m/s")
    print(f"  last 8 s: height {np.mean(z_tail):.2f} m (min {np.min(z_tail):.2f}), "
          f"speed {np.mean(v_tail):.2f} m/s")

    out = {"laps": abs(laps), "peak_z": peak_z, "peak_v": peak_v,
           "z_mean": float(np.mean(z_tail)),
           "video": str(out_video) if out_video else None}

    if descend_step:
        down = hist[descend_step:]
        drop_v = [-h[3] for h in down]           # positive when falling
        parked = [h for h in down if h[4] == "park"]
        z_end = down[-1][0]
        v_end = down[-1][1]
        # A smooth way down is one the robot is in charge of the whole time:
        # no stretch of free fall, and a stop at the bottom rather than a stop
        # against something.
        print(f"  --- coming down ---")
        print(f"  fastest drop    {max(drop_v):.2f} m/s   (free fall from "
              f"{peak_z:.1f} m would reach {np.sqrt(2*9.81*peak_z):.1f})")
        print(f"  time down       {(len(down) - len(parked)) / 100:.1f} s")
        print(f"  ends at         z {z_end:.2f} m, {v_end:.2f} m/s"
              f"{' , parked' if parked else ''}")
        out.update(max_drop=float(max(drop_v)), z_end=float(z_end),
                   v_end=float(v_end), parked=bool(parked),
                   down_time=(len(down) - len(parked)) / 100.0)

    if out_video is not None:
        print(f"  video           {out_video}")
    return out


def _aim(cam, eye, target):
    """Point a MuJoCo free camera from `eye` at `target`.

    MuJoCo places a free camera by ``lookat``, ``distance``, ``azimuth`` and
    ``elevation`` rather than by a world position, so a camera mounted at a
    real place has to be solved for. The camera sits at

        lookat - distance * (cos(el)cos(az), cos(el)sin(az), sin(el))

    which inverts directly.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    v = eye - target
    dist = float(np.linalg.norm(v))
    if dist < 1e-6:
        dist = 1e-6
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = target
    cam.distance = dist
    cam.elevation = float(np.degrees(np.arcsin(np.clip(-v[2] / dist, -1.0, 1.0))))
    cam.azimuth = float(np.degrees(np.arctan2(-v[1], -v[0])))


def _frame(env, info, theta, laps, bowl, mode, rim_r, wall_top, ball_pos):
    if env.renderer is None:
        env.render(camera_name="fixed_angle_close_3d")
    cam = mujoco.MjvCamera()

    if mode == "rim":
        # Mounted on the top edge of the drome wall, on the rail, opposite the
        # robot and looking across at it.
        #
        # Opposite is not a stylistic choice. The robot rides at r = 2.96 m and
        # the boards are at 3.18 m, so a camera bolted to one point of the rail
        # loses it completely once a lap, when it passes directly underneath
        # and the near wall covers it. Sliding round the rail to stay across
        # the bowl keeps it in frame the whole way and keeps the framing
        # identical every lap.
        phi = float(np.arctan2(ball_pos[1], ball_pos[0])) + np.pi
        eye = np.array([(rim_r + 0.60) * np.cos(phi),
                        (rim_r + 0.60) * np.sin(phi),
                        wall_top + 0.55])
        # Aim mostly at the robot, partly at a point over the middle, so the
        # bowl stays in shot instead of the frame chasing every wobble. The
        # middle point is lifted so the robot sits below the text panel rather
        # than behind it.
        target = 0.55 * np.asarray(ball_pos, dtype=np.float64) \
            + 0.45 * np.array([0.0, 0.0, bowl.rim_z * 0.55 + 0.90])
        _aim(cam, eye, target)
    elif mode == "rim_fixed":
        # Bolted to one point of the rail and never moved. Truer to a real
        # spectator, but the robot disappears behind the near boards for about
        # a fifth of every lap.
        phi = np.radians(150.0)
        eye = np.array([(rim_r + 1.60) * np.cos(phi),
                        (rim_r + 1.60) * np.sin(phi),
                        wall_top + 1.60])
        _aim(cam, eye, np.array([0.0, 0.0, 1.10]))
    else:
        # Free camera aimed at the middle of the drome, swinging round with
        # the robot. Tracking the robot itself puts the camera down inside the
        # bowl during the early laps, where the boards block everything.
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.85]
        cam.distance = 8.6
        cam.elevation = -30.0
        cam.azimuth = float(np.degrees(theta)) + 48.0

    env.renderer.update_scene(env.data, camera=cam)
    return annotate(
        env.renderer.render(),
        "Wall of Death",
        [
            f"height   {info['z']:.2f} m  of a {bowl.rim_z:.2f} m bowl",
            f"radius   {info['r']:.2f} m  asked {info['r_cmd']:.2f} m",
            f"speed    {info['speed']:.2f} m/s  (bank holds {info['hold_speed']:.2f})",
            f"bank     {info['bank_deg']:.0f} deg",
            f"laps     {laps:.1f}",
        ],
        margin=14,
    )


def main():
    p = argparse.ArgumentParser(description="Wall of Death benchmark")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str, default="configs/rl/motordrome.yaml")
    p.add_argument("--open-rate", type=float, default=0.10)
    p.add_argument("--steer-gain", type=float, default=0.30)
    p.add_argument("--max-steer", type=float, default=0.35)
    p.add_argument("--camera", type=str, default="rim",
                   choices=("rim", "rim_fixed", "orbit"),
                   help="rim: mounted on the top edge, aim follows the robot. "
                        "rim_fixed: same mount, aim never moves. "
                        "orbit: outside the drome, swinging round.")
    p.add_argument("--descend-after", type=float, default=None,
                   help="seconds of riding before winding the spiral back down")
    p.add_argument("--close-rate", type=float, default=0.20,
                   help="metres of radius per second on the way down")
    p.add_argument("--brake-gain", type=float, default=1.5)
    p.add_argument("--descend-speed", type=float, default=0.80,
                   help="fraction of the bank's hold speed to aim for on the way down")
    p.add_argument("--video", action="store_true", default=False)
    args = p.parse_args()
    run(seconds=args.seconds, seed=args.seed, config=args.config,
        open_rate=args.open_rate, steer_gain=args.steer_gain,
        max_steer=args.max_steer, record_video=args.video, camera=args.camera,
        descend_after=args.descend_after, close_rate=args.close_rate,
        brake_gain=args.brake_gain, descend_speed=args.descend_speed)


if __name__ == "__main__":
    main()
