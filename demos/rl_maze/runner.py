"""Cross a maze using nothing but the `skills_rl` primitives.

The point is the basis, not the controller. `choose` below is about thirty
lines of hand-written policy: follow the route, conform on rough ground, hop
when jammed, brake at the goal. It is deliberately unclever. If seven
primitives can carry a robot through a labyrinth under a chooser this simple,
they are enough for a learned one to work with, and an RL policy would replace
`choose` and nothing else.

    python scripts/run_demo.py demo=rl_maze
    python scripts/run_demo.py demo=rl_maze video=true
    python demos/rl_maze/runner.py knobs.layout_seed=7 knobs.video=true
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

import skills_rl as S
from radial_sphere.config import demo_config, load_config
from radial_sphere.controller import desired_direction
from radial_sphere.demo import Recorder
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.overlay import annotate
from radial_sphere.scenario import generate_scenario
from skills_rl.base import RobotState
from skills_rl.calibration import full_stroke_steps, profile_for_config

BANK_CONFIG = "configs/rl/maze_complex_blockers_random_endpoints.yaml"


class Chooser:
    """Pick one primitive per control step, from live state only.

    Four rules, in priority order. Nothing here is tuned beyond the thresholds
    in `demo.yaml`, and nothing remembers more than how long the robot has been
    stuck and how much of a hop is left to run.
    """

    def __init__(self, args, profile: str, top_speed: float):
        self.a = args
        self.burn = full_stroke_steps(profile)
        self.top = top_speed
        self.stuck_for = 0
        self.hop_left = 0
        self.fly_left = 0
        self.counts: dict[str, int] = {}

    def __call__(self, *, heading, dist_to_goal, speed, rough) -> tuple[str, dict, str]:
        az = float(np.arctan2(heading[1], heading[0]))

        # A hop, once started, runs to the end. A rod capped at 0.28 m/s needs
        # 0.48 s to cross its stroke, so an interrupted burst delivers almost
        # none of its power: 10 steps rises 0.030 m against 0.128 m for 50.
        if self.hop_left > 0:
            self.hop_left -= 1
            return "thrust", dict(azimuth=az, elevation=-np.pi / 2, power=1.0,
                                  spread=0.55), "jammed: hop over it"
        if self.fly_left > 0:
            self.fly_left -= 1
            return "tuck", dict(extension=0.01), "airborne, rods in"

        if dist_to_goal < 0.5:
            return ("brake", dict(strength=3.2), "at the goal, braking") if speed > 0.15 \
                else ("stance", dict(height=0.045), "stopped on the goal")

        if speed < float(self.a.stuck_speed):
            self.stuck_for += 1
            if self.stuck_for >= int(self.a.stuck_steps):
                self.stuck_for = 0
                self.hop_left, self.fly_left = self.burn, 40
                return "thrust", dict(azimuth=az, elevation=-np.pi / 2, power=1.0,
                                      spread=0.55), "jammed: hop over it"
        else:
            self.stuck_for = 0

        if rough:
            # `conform` composes: it corrects a base rather than replacing it,
            # which is what `traverse_rough_terrain` is in the hand-written
            # library. Rough ground is where that matters.
            base = ("drive", dict(azimuth=az, speed=0.85 * self.top,
                                  vault=float(self.a.vault), flank=0.0))
            return ("conform",
                    dict(ride_height=0.21, terrain_gain=0.85, stiffness=0.75,
                         base=base),
                    "rough: drive + vault + conform")

        return ("drive", dict(azimuth=az, speed=self.top, vault=1.0, flank=0.0),
                "follow the route")


def main():
    args = demo_config("rl_maze").knobs
    profile = str(args.profile)

    cfg = load_config(BANK_CONFIG)
    cfg.camera.enabled = False
    cfg.sim2real.enabled = True
    cfg.sim2real.actuator_in_env = (profile == "hardware")
    cfg.robot.appearance_theme = "realistic"
    cfg.scenario.maze.layout_seed = int(args.layout_seed)
    cfg.scenario.maze.random_endpoints = True
    assert profile_for_config(cfg) == profile, "profile and env disagree"
    S.use_profile(profile)

    scenario = generate_scenario("maze", cfg, seed=int(args.endpoint_seed))
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False,
                                max_steps=200000)
    env.reset(seed=int(args.endpoint_seed))

    recorder = Recorder(env, "run_rl_maze", tag=f"{profile}_L{args.layout_seed}",
                        enabled=args.video, fps=args.fps, every=1,
                        out_name="rl_maze")

    def dual():
        return env.render(camera_name="dual")

    dt = float(env.model.opt.timestep) * env.action_repeat
    steps = int(round(float(args.seconds) / dt))
    top = S.PRIMITIVES["drive"].SPEC.params[1].high
    choose = Chooser(args, profile, top)
    path_pts = np.asarray(scenario.path_pts, dtype=np.float64)
    route = float(scenario.path_length)

    print(f"rl maze   layout {args.layout_seed} ep {args.endpoint_seed}   "
          f"route {route:.1f} m   profile {profile}   drive tops at {top:.2f} m/s")
    print(f"recording {args.seconds:.0f} s at 1x = {steps} control steps")

    start = env.data.qpos[:2].copy()
    travelled, prev = 0.0, start.copy()
    best = env._nav_distance(start)

    for step in range(steps):
        xy = env.data.qpos[:2].copy()
        heading, _ = desired_direction(xy, path_pts, float(args.lookahead))
        speed = float(np.linalg.norm(env.data.qvel[:2]))
        dist = float(env._nav_distance(xy))
        best = min(best, dist)

        # NaN means a rod's ray hit nothing, which is most of them at any
        # moment. Comparing NaN to a threshold is always False, so without this
        # the rough-ground test never fires and `conform` is never chosen:
        # measured, that turned 57 % of steps into 0 %.
        clear = np.nan_to_num(env.get_terrain_clearances(), nan=0.0)
        rough = bool(np.max(np.abs(clear)) > float(args.rough_clearance))

        name, params, note = choose(heading=heading, dist_to_goal=dist,
                                    speed=speed, rough=rough)
        choose.counts[name] = choose.counts.get(name, 0) + 1

        st = RobotState(quat=env.data.qpos[3:7].copy(), dirs_body=env.dirs_body,
                        max_extend=env.max_extend, lin_vel=env.data.qvel[:3].copy(),
                        core_z=float(env.data.qpos[2]), core_vz=float(env.data.qvel[2]),
                        contact_forces=env.get_rod_contact_forces(),
                        terrain_clearances=clear, profile=profile)

        # `conform` takes its base from another primitive, so it is composed
        # here rather than dispatched flat.
        if name == "conform":
            base_name, base_params = params.pop("base")
            params["base"] = S.act(base_name, st, **base_params)
        env.step(S.act(name, st, **params))

        travelled += float(np.linalg.norm(env.data.qpos[:2] - prev))
        prev = env.data.qpos[:2].copy()

        if recorder.enabled and step % args.frame_every == 0:
            recorder.add(annotate(
                dual(), f"skills_rl in a maze  [{name}]",
                [note,
                 f"route {route:.1f} m   {dist:5.2f} m to go   closest {best:5.2f} m",
                 f"speed {speed:4.2f} m/s   travelled {travelled:5.1f} m",
                 f"t {step * dt:6.1f} s   (real time, profile {profile})"]))

        if dist < 0.45 and speed < 0.15:
            print(f"  goal reached at {step * dt:.1f} s")
            break

    print(f"  closest approach {best:.2f} m of a {route:.1f} m route, "
          f"travelled {travelled:.1f} m")
    print(f"  primitives used: {dict(sorted(choose.counts.items(), key=lambda kv: -kv[1]))}")
    recorder.close()
    if recorder.enabled:
        print(f"video: {recorder.path}  ({recorder.n_frames / args.fps:.1f} s, real time)")
    env.close()


if __name__ == "__main__":
    main()
