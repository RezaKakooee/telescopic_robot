"""Tour the seven RL primitives on one arena, and record what each one does.

`skills_rl/` re-cuts the skill library into a basis a policy can choose from.
This demo drives that basis by hand, one primitive at a time, so the mechanics
can be watched rather than inferred from a reward curve.

The arena carries what the tour needs: open floor to roll and stop on, a patch
of stones for `conform`, and a wall to `brace` against.

    python scripts/run_demo.py demo=rl_primitives
    python scripts/run_demo.py demo=rl_primitives video=true
    python demos/rl_primitives/runner.py knobs.profile=ideal knobs.video=true
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
from radial_sphere.demo import Recorder
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.overlay import annotate
from radial_sphere.scenario import Scenario
from skills_rl.base import RobotState
from skills_rl.calibration import full_stroke_steps, profile_for_config

# The wall sits where the ball can actually reach it. With a 0.15 m shell,
# contact happens 0.15 m short of the face, and the first layout put the wall
# far enough out that `brace` pressed thin air.
WALL_Y = -0.55
#: Centred where the tour's drive segment ends, not on the nominal path. The
#: first layout put the stones at x=4.2 and the ball finished at x=2.8.
STONES_XY = (2.6, 0.35)

#: One segment per line: primitive, how many control steps, its parameters,
#: and what the viewer should be looking for.
#
# `thrust` is held for `full_stroke_steps`, not an arbitrary number. A rod
# capped at 0.28 m/s needs 0.48 s to cross its stroke, so a shorter burst
# delivers a fraction of the commanded power: measured, 10 steps rises 0.030 m
# and 50 steps rises 0.128 m.
def tour(profile: str):
    burn = full_stroke_steps(profile)
    top = S.PRIMITIVES["drive"].SPEC.params[1].high
    cruise = 0.95 * top
    return [
        ("stance", 60, dict(height=0.045), "settle on a wide base"),
        ("drive", 240, dict(azimuth=0.0, speed=cruise, flank=0.0),
         "roll out along +x on clear floor"),
        ("brake", 90, dict(strength=3.2), "plant a leading kickstand"),
        ("thrust", burn, dict(azimuth=0.0, elevation=-np.pi / 2, power=1.0,
                              spread=0.55), f"fire downward, held {burn} steps"),
        ("tuck", 70, dict(extension=0.01), "pull every rod in for the flight"),
        # The wall comes before the stones. Driving off a rock field and into a
        # wall in one go does neither well: the first run of this tour spent
        # 260 steps on the stones and never reached the wall at all.
        ("drive", 320, dict(azimuth=-np.pi / 2, speed=cruise, flank=0.0),
         "turn and roll toward the wall"),
        ("brace", 110, dict(azimuth=-np.pi / 2, elevation=0.0, extension=1.0,
                            spread=0.45, both_sides=0.0),
         "press the wall and be shoved clear"),
        ("drive", 380, dict(azimuth=0.35, speed=cruise, flank=0.0),
         "head out to the stone field"),
        ("conform", 200, dict(ride_height=0.21, terrain_gain=0.85,
                              stiffness=0.75), "follow the ground, no drive"),
        ("stance", 50, dict(height=0.045), "settle again"),
    ]


def arena() -> Scenario:
    """Open floor, a stone patch to conform over, and a wall to brace on."""
    return Scenario(
        kind="goal", name="rl_primitive_tour",
        spawn_xy=np.array([0.0, 0.0], dtype=np.float32),
        goal=np.array([6.0, 0.0], dtype=np.float32),
        path_pts=np.array([[0.0, 0.0], [6.0, 0.0]], dtype=np.float32),
        markers=np.empty((0, 2), dtype=np.float32), path_length=6.0,
        walls=np.array([[-1.0, WALL_Y, 8.0, WALL_Y]], dtype=np.float32),
        # (x, y, hx, hy, n, max_size)
        stones=np.array([[STONES_XY[0], STONES_XY[1], 0.9, 0.9, 60, 0.045]],
                        dtype=np.float32),
    )


def main():
    args = demo_config("rl_primitives").knobs
    profile = str(args.profile)

    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.sim2real.enabled = True
    # The calibration and the env have to agree about which robot this is.
    cfg.sim2real.actuator_in_env = (profile == "hardware")
    cfg.robot.appearance_theme = "realistic"
    assert profile_for_config(cfg) == profile, "profile and env disagree"
    S.use_profile(profile)

    scenario = arena()
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False,
                                max_steps=4000)
    env.reset(seed=args.seed)

    recorder = Recorder(env, "run_rl_primitives", tag=profile,
                        enabled=args.video, fps=args.fps, every=1,
                        out_name="rl_primitives")

    def chase():
        if env.renderer is None:
            env.render(camera_name="fixed_angle_close_3d")
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = env.core_body_id
        cam.distance, cam.elevation, cam.azimuth = 2.1, -24.0, 130.0
        env.renderer.update_scene(env.data, camera=cam)
        return env.renderer.render()

    def state():
        return RobotState(
            quat=env.data.qpos[3:7].copy(), dirs_body=env.dirs_body,
            max_extend=env.max_extend, lin_vel=env.data.qvel[:3].copy(),
            core_z=float(env.data.qpos[2]), core_vz=float(env.data.qvel[2]),
            contact_forces=env.get_rod_contact_forces(),
            terrain_clearances=env.get_terrain_clearances(),
            profile=profile)

    dt = float(env.model.opt.timestep) * env.action_repeat
    print(f"rl primitives tour   profile {profile}   "
          f"drive tops out at {S.PRIMITIVES['drive'].SPEC.params[1].high:.2f} m/s")
    print(f"{'primitive':<10} {'steps':>6} {'travelled':>10} {'peak rise':>10}  note")

    step = 0
    for name, steps, params, note in tour(profile):
        p0 = env.data.qpos[:3].copy()
        peak = float(p0[2])
        for _ in range(steps):
            env.step(S.act(name, state(), **params))
            peak = max(peak, float(env.data.qpos[2]))
            step += 1
            if recorder.enabled and step % args.frame_every == 0:
                q = env.data.qpos
                recorder.add(annotate(
                    chase(), f"skills_rl.{name}",
                    [note,
                     f"profile {profile}   "
                     + "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in params.items()),
                     f"speed {float(np.linalg.norm(env.data.qvel[:2])):4.2f} m/s   "
                     f"core z {float(q[2]):4.2f} m",
                     f"t {step * dt:5.2f} s   (real time)"]))
        moved = float(np.linalg.norm(env.data.qpos[:2] - p0[:2]))
        q = env.data.qpos
        print(f"{name:<10} {steps:>6} {moved:>9.2f} m {peak - float(p0[2]):>9.3f} m  "
              f"at ({float(q[0]):+.2f},{float(q[1]):+.2f})  {note}")

    recorder.close()
    if recorder.enabled:
        print(f"video: {recorder.path}  ({recorder.n_frames / args.fps:.1f} s, real time)")
    env.close()


if __name__ == "__main__":
    main()
