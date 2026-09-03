"""Horizontal wall run: sprint beside a wall, leap at it, ride it, push off.

Supports three modular modes:
1. 'curved': A wall that bends in plan, testing a changing surface frame.
2. 'banked': An inclined wall, testing the full 3-D wall plane and normal.
3. 'flat_multistep' / 'flat': A vertical wall where successive rotating rods
   make the airborne contact sequence before push-off.

Run::

    python scripts/skills/run_wall_run.py --mode curved --video
    python scripts/skills/run_wall_run.py --mode banked --video
    python scripts/skills/run_wall_run.py --mode flat_multistep --video
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
from skills.wall_run import FOOT_BASE, get_wall_frame, next_phase, wall_run


def run(
    seconds: float = 12.0,
    seed: int = 3,
    record_video: bool = True,
    config: str = "configs/rl/wall_run.yaml",
    mode: str | None = None,
    speed: float = 7.5,
    recover_speed: float = 1.5,
    approach_angle: float = 28.0,
    launch_gap: float | None = None,
    launch_in: float = 3.6,
    launch_up: float = 5.2,
    give: float = 0.13,
    squash_span: float = 0.26,
    cushion_max: float = 0.025,
    push_frac: float = 1.00,
    along_drive: float = 0.55,
    upward_bias: float = 0.35,
    min_turns: float = 0.25,
    target_distance: float = 1.0,
    fps: int = 50,
    frame_every: int = 2,
    repeats: int = 1,
    slowmo: int = 1,
    video_name: str | None = None,
):
    cfg = load_config(config)
    OmegaConf.set_struct(cfg, False)
    if mode is not None:
        cfg.scenario.mode = mode
    scenario = generate_scenario("wall_run", cfg, seed=seed)
    wall_mode = getattr(scenario, "wall_mode", "curved")
    if launch_gap is None:
        # The curved wall bends toward the trajectory, so it needs more lead
        # than a straight plane to meet the cage near the jump apex.
        launch_gap = 1.60 if wall_mode == "curved" else 1.35

    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10 ** 7)
    env.reset(seed=seed)
    mujoco.mj_forward(env.model, env.data)

    init_pos = env.data.qpos[:3].copy()
    init_gap, _, _ = get_wall_frame(init_pos, scenario)
    lane_gap = init_gap
    full_reach = FOOT_BASE + env.max_extend
    step_dt = float(env.model.opt.timestep * env.action_repeat)

    print(f"=== horizontal wall run [{wall_mode.upper()}] ===")
    print(f"  wall mode: {wall_mode}, spawn gap {init_gap:.2f} m out")
    print(f"  rods reach {full_reach:.2f} m; asking for {speed:.1f} m/s before the leap")

    if slowmo > 1:
        frame_every = 1
        fps = max(int(100 / slowmo), 1)

    writer = out_video = None
    if record_video:
        vid_title = video_name or f"wall_run_{wall_mode}"
        run_dir = make_run_dir(build_run_id("wall_run", wall_mode))
        out_video = Path(run_dir) / "renders" / f"{vid_title}.mp4"
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out_video), fps=fps, codec="libx264")

    phase = "sprint"
    phase_started = 0.0
    runs = 1
    per_run = [_new_run_metrics()]
    hist = []
    gap_at = []

    for i in range(int(seconds * 100)):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        wall_dist, n_hat, travel = get_wall_frame(pos, scenario)
        on_ground = _touching_floor(env)
        current = per_run[-1]
        turns_now = current["roll_rad"] / (2.0 * np.pi)
        phase_elapsed = i * step_dt - phase_started

        nxt = next_phase(
            phase, wall_dist=wall_dist, lin_vel=vel, height=float(pos[2]),
            on_ground=on_ground, max_extend=env.max_extend, speed_ready=speed,
            launch_gap=launch_gap, lane_gap=lane_gap, wall_normal=n_hat,
            turns_completed=turns_now, min_turns=min_turns, wall_mode=wall_mode,
            ride_distance=current["along"],
            ride_time=current["contact_steps"] * step_dt,
            target_distance=target_distance,
            phase_elapsed=phase_elapsed,
        )
        if phase == "settle" and runs < repeats and float(np.linalg.norm(vel)) < 0.25:
            runs += 1
            nxt = "recover"
            per_run.append(_new_run_metrics())
        if nxt != phase:
            print(f"  t={i/100:5.2f}s  {phase} -> {nxt}   x={pos[0]:.2f} "
                  f"z={pos[2]:.2f} gap={wall_dist:.2f} v={np.linalg.norm(vel):.2f} turns={turns_now:.2f}")
            phase = nxt
            phase_started = i * step_dt

        targets = wall_run(
            quat, env.dirs_body, env.max_extend,
            phase=phase, wall_normal=n_hat, wall_dist=wall_dist,
            travel=travel, lin_vel=vel, speed=speed,
            approach_angle=approach_angle, recover_speed=recover_speed,
            lane_gap=lane_gap,
            launch_in=launch_in, launch_up=launch_up,
            give=give, squash_span=squash_span,
            cushion_max=cushion_max, push_frac=push_frac,
            along_drive=along_drive, upward_bias=upward_bias,
        )
        env.step(targets)

        # Measure the state that resulted from this command.  The old runner
        # paired post-step contacts with pre-step positions and velocities.
        pos_now = env.data.qpos[:3].copy()
        vel_now = env.data.qvel[:3].copy()
        angvel_now = env.data.qvel[3:6].copy()
        gap_now, n_now, travel_now = get_wall_frame(pos_now, scenario)
        contact_rods, core_contact = _wall_contact_rods(env)
        touching_wall = bool(contact_rods) or core_contact
        touching_floor = _touching_floor(env)
        compress_mm = None

        if touching_wall:
            current = per_run[-1]
            current["contact_steps"] += 1
            current["airborne_steps"] += int(not touching_floor)
            current["ground_steps"] += int(touching_floor)
            current["core_contact"] = current["core_contact"] or core_contact
            current["unique_rods"].update(contact_rods)
            current["contact"].append((float(pos_now[0]), float(pos_now[2])))
            current["z_min"] = min(current["z_min"], float(pos_now[2]))
            current["z_max"] = max(current["z_max"], float(pos_now[2]))
            if current["entry_speed"] is None:
                current["entry_speed"] = float(np.linalg.norm(vel_now))
                current["entry_z"] = float(pos_now[2])

            if current["prev_contact_pos"] is not None:
                delta = pos_now - current["prev_contact_pos"]
                current["along"] += abs(float(np.dot(delta, travel_now)))
            current["prev_contact_pos"] = pos_now.copy()

            roll_axis = np.cross(n_now, travel_now)
            roll_axis /= max(float(np.linalg.norm(roll_axis)), 1e-9)
            current["roll_rad"] += abs(float(np.dot(angvel_now, roll_axis))) * step_dt

            if contact_rods:
                rod_extensions = env.data.qpos[7:7+len(env.dirs_body)]
                compression = max(float(env.max_extend - rod_extensions[k]) for k in contact_rods)
                current["compression"] = max(current["compression"], compression)
                compress_mm = 1000.0 * compression
            gap_at.append(gap_now)
        else:
            per_run[-1]["prev_contact_pos"] = None

        hist.append((float(pos_now[0]), float(pos_now[1]), float(pos_now[2]), phase,
                     float(np.dot(vel_now, n_now))))

        if writer is not None and i % frame_every == 0:
            writer.append_data(_frame(env, phase, pos_now, vel_now, gap_now,
                                      slowmo, wall_mode=wall_mode,
                                      turns=per_run[-1]["roll_rad"] / (2.0 * np.pi), n_hat=n_now,
                                      compress_mm=compress_mm))

    if writer is not None:
        writer.close()
    env.close()

    peak_z = max(h[2] for h in hist)
    after = [h for h in hist if h[3] in ("push", "land")]
    exit_v = max((-h[4] for h in after), default=0.0)
    end_z = float(np.mean([h[2] for h in hist[-150:]]))

    print("\n--- summary ---")
    good = []
    for k, r in enumerate(per_run, 1):
        c = r["contact"]
        if not c:
            print(f"  run {k}: never reached the wall")
            continue
        t_sec = r["contact_steps"] * step_dt
        airborne_sec = r["airborne_steps"] * step_dt
        rot_turns = r["roll_rad"] / (2.0 * np.pi)
        airborne_fraction = airborne_sec / max(t_sec, 1e-9)
        print(f"  run {k}: wall {t_sec:.2f} s ({airborne_fraction:.0%} airborne), "
              f"{r['along']:.2f} m along, height {r['z_min']:.2f}-{r['z_max']:.2f} m, "
              f"{rot_turns:.2f} true roll turns, {len(r['unique_rods'])} contact rods")
        good.append(dict(
            secs=t_sec, airborne_secs=airborne_sec,
            airborne_fraction=airborne_fraction, along=r["along"],
            z=(r["z_min"], r["z_max"]), entry_z=r["entry_z"],
            entry_speed=r["entry_speed"], turns=rot_turns,
            contact_rods=len(r["unique_rods"]), compression=r["compression"],
            core_contact=r["core_contact"],
        ))

    max_compression = max((r["compression"] for r in per_run), default=0.0)
    min_gap = min(gap_at, default=float("inf"))
    print(f"  wall rods folded {max_compression*1000:.0f} mm of a "
          f"{env.max_extend*1000:.0f} mm stroke, at the deepest contact")
    print(f"  peak height     {peak_z:.2f} m")
    print(f"  push off        {exit_v:.2f} m/s away from the wall")
    print(f"  settles at      z {end_z:.2f} m")

    out = {"runs": good, "peak_z": peak_z, "exit_v": float(exit_v),
           "end_z": end_z, "min_gap": float(min_gap),
           "squash": float(max_compression),
           "mode": wall_mode,
           "hist": hist,
           "video": str(out_video) if out_video else None}
    if out_video is not None:
        print(f"  video           {out_video}")
    return out


def _touching_floor(env):
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        for g in (c.geom1, c.geom2):
            if mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g) == "floor":
                return True
    return False


def _touching_wall(env):
    rods, core = _wall_contact_rods(env)
    return bool(rods) or core


def _wall_contact_rods(env):
    """Return rod indices touching a wall and whether the core hit it."""
    rods = set()
    core_contact = False
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        n1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        n2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        if n1.startswith("wall_"):
            robot_geom = g2
        elif n2.startswith("wall_"):
            robot_geom = g1
        else:
            continue
        if robot_geom == env.core_geom_id:
            core_contact = True
        rod = env.rod_geom_map.get(robot_geom)
        if rod is not None:
            rods.add(int(rod))
    return rods, core_contact


def _new_run_metrics():
    return {
        "contact": [], "contact_steps": 0, "airborne_steps": 0,
        "ground_steps": 0, "along": 0.0, "roll_rad": 0.0,
        "unique_rods": set(), "compression": 0.0, "core_contact": False,
        "prev_contact_pos": None, "entry_speed": None, "entry_z": None,
        "z_min": float("inf"), "z_max": float("-inf"),
    }


def _aim(cam, eye, target):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    v = eye - target
    dist = max(float(np.linalg.norm(v)), 1e-6)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = target
    cam.distance = dist
    cam.elevation = float(np.degrees(np.arcsin(np.clip(-v[2] / dist, -1.0, 1.0))))
    cam.azimuth = float(np.degrees(np.arctan2(-v[1], -v[0])))


def _frame(env, phase, pos, vel, wall_dist, slowmo=1, wall_mode="curved", turns=0.0, n_hat=None, compress_mm=None):
    if env.renderer is None:
        env.render(camera_name="fixed_angle_close_3d")
    cam = mujoco.MjvCamera()
    eye = np.array([pos[0] - 2.4, pos[1] - 5.5, pos[2] + 3.2])
    look = np.array([pos[0] + 0.8, pos[1] + 0.6, pos[2] + 0.30])
    _aim(cam, eye, look)
    env.renderer.update_scene(env.data, camera=cam)

    into_wall = float(np.dot(vel, n_hat)) if n_hat is not None else float(vel[1])
    phase_label = phase.upper()
    if phase == "fly":
        phase_label = "FLY (ALL 60 RODS OPEN)"
    elif phase == "ride":
        phase_label = "RIDE (WALL CONTACT)"

    return annotate(
        env.renderer.render(),
        f"Horizontal Wall Run [{wall_mode.upper()}]" + (f"  ({slowmo}x slow)" if slowmo > 1 else ""),
        [
            f"mode        {wall_mode}",
            f"phase       {phase_label}",
            f"gap         {wall_dist:.2f} m to wall",
            f"height      {pos[2]:.2f} m",
            f"speed       {np.linalg.norm(vel):.2f} m/s  (into wall: {into_wall:+.2f} m/s)",
            (f"wall stroke {compress_mm:5.1f} mm / 300 mm" if compress_mm is not None
             else "wall stroke -- (no wall contact)"),
            f"wall roll   {turns:.2f} rotations",
        ],
        margin=14,
    )


def main():
    p = argparse.ArgumentParser(description="Horizontal wall run (Modular Modes)")
    p.add_argument("--mode", type=str, default="curved",
                   choices=["curved", "banked", "flat_multistep", "flat", "all"],
                   help="wall run mode to execute")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--speed", type=float, default=7.5)
    p.add_argument("--approach-angle", type=float, default=28.0)
    p.add_argument("--launch-gap", type=float, default=None,
                   help="takeoff distance to the wall (mode-tuned by default)")
    p.add_argument("--launch-in", type=float, default=3.6)
    p.add_argument("--launch-up", type=float, default=5.2)
    p.add_argument("--give", type=float, default=0.13)
    p.add_argument("--squash-span", type=float, default=0.26)
    p.add_argument("--cushion-max", type=float, default=0.025)
    p.add_argument("--push-frac", type=float, default=1.00)
    p.add_argument("--along-drive", type=float, default=0.55)
    p.add_argument("--upward-bias", type=float, default=0.35)
    p.add_argument("--min-turns", type=float, default=0.25,
                   help="minimum true wall-roll turns before push-off")
    p.add_argument("--target-distance", type=float, default=1.0)
    p.add_argument("--slowmo", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--video", action="store_true", default=False)
    a = p.parse_args()

    if a.mode == "all":
        for m in ["curved", "banked", "flat_multistep"]:
            print(f"\n{'='*60}\n  RUNNING WALL RUN MODE: {m.upper()}\n{'='*60}")
            run(mode=m, seconds=a.seconds, seed=a.seed, speed=a.speed,
                approach_angle=a.approach_angle, launch_gap=a.launch_gap,
                launch_in=a.launch_in, launch_up=a.launch_up,
                give=a.give, squash_span=a.squash_span, cushion_max=a.cushion_max,
                push_frac=a.push_frac, along_drive=a.along_drive,
                upward_bias=a.upward_bias, min_turns=a.min_turns,
                target_distance=a.target_distance,
                record_video=a.video, repeats=a.repeats, slowmo=a.slowmo)
    else:
        run(mode=a.mode, seconds=a.seconds, seed=a.seed, speed=a.speed,
            approach_angle=a.approach_angle, launch_gap=a.launch_gap,
            launch_in=a.launch_in, launch_up=a.launch_up,
            give=a.give, squash_span=a.squash_span, cushion_max=a.cushion_max,
            push_frac=a.push_frac, along_drive=a.along_drive,
            upward_bias=a.upward_bias, min_turns=a.min_turns,
            target_distance=a.target_distance,
            record_video=a.video, repeats=a.repeats, slowmo=a.slowmo)


if __name__ == "__main__":
    main()
