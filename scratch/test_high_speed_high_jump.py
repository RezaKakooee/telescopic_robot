import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import numpy as np
import mujoco
from omegaconf import OmegaConf

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat
from skills.wall_run import FOOT_BASE, _frame, get_wall_frame, next_phase, wall_run
from skills.locomotion import move, stop

def test_jump(mode="curved", speed=6.5, launch_up=4.2, launch_in=2.6, launch_gap=1.6, approach_angle=20.0):
    cfg = load_config("configs/rl/wall_run.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.scenario.mode = mode
    if mode == "curved":
        cfg.scenario.curved.radius = 7.5
        cfg.scenario.curved.arc_deg = 50.0

    scenario = generate_scenario("wall_run", cfg, seed=3)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10 ** 7)
    env.reset(seed=3)
    mujoco.mj_forward(env.model, env.data)

    lane_gap = float(scenario.walls[0, 1] - scenario.spawn_xy[1])
    full_reach = FOOT_BASE + env.max_extend
    n_bars = len(env.dirs_body)

    phase = "approach"
    contact_steps = 0
    cum_rot_rad = 0.0
    wall_x_start = None
    wall_x_end = None
    z_min_wall = 999.0
    z_max_wall = -999.0
    v_hit_wall = 0.0

    print(f"\n=======================================================")
    print(f"  Testing Mode: {mode.upper()} (speed={speed} m/s, launch_up={launch_up} m/s, launch_gap={launch_gap} m)")
    print(f"=======================================================")

    for step in range(int(5.0 * 100)):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        angvel = env.data.qvel[3:6].copy()
        quat = env.data.qpos[3:7].copy()

        wall_dist, n_hat, travel = get_wall_frame(pos, scenario)
        on_ground = any(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom1) == "floor" or
                        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom2) == "floor"
                        for c_i in range(env.data.ncon))

        touching_wall = any((mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom1) or "").startswith("wall_") or
                            (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom2) or "").startswith("wall_")
                            for c_i in range(env.data.ncon))

        if touching_wall:
            if contact_steps == 0:
                v_hit_wall = float(np.linalg.norm(vel))
                print(f"  --> HIT WALL at t={step/100:.2f}s: height z={pos[2]:.2f}m, speed v={v_hit_wall:.2f}m/s (into wall: {np.dot(vel, n_hat):.2f})")
            contact_steps += 1
            spin = float(np.linalg.norm(angvel))
            cum_rot_rad += spin * 0.01
            z_min_wall = min(z_min_wall, pos[2])
            z_max_wall = max(z_max_wall, pos[2])
            if wall_x_start is None:
                wall_x_start = pos[0]
            wall_x_end = pos[0]

        turns_now = cum_rot_rad / (2.0 * np.pi)

        nxt = next_phase(
            phase, wall_dist=wall_dist, lin_vel=vel, height=float(pos[2]),
            on_ground=on_ground, max_extend=env.max_extend, speed_ready=speed * 0.85,
            launch_gap=launch_gap, lane_gap=lane_gap, wall_normal=n_hat,
            turns_completed=turns_now, min_turns=1.0, wall_mode=mode,
        )
        if nxt != phase:
            print(f"  t={step/100:5.2f}s  {phase} -> {nxt}   x={pos[0]:.2f} "
                  f"z={pos[2]:.2f} gap={wall_dist:.2f} v={np.linalg.norm(vel):.2f}")
            phase = nxt

        targets = wall_run(
            quat, env.dirs_body, env.max_extend,
            phase=phase, wall_normal=n_hat, wall_dist=wall_dist,
            travel=travel, lin_vel=vel, speed=speed,
            approach_angle=approach_angle, recover_speed=1.5,
            launch_in=launch_in, launch_up=launch_up,
            give=0.18, squash_span=0.26, cushion_max=0.05, push_frac=0.70,
            along_drive=0.80, upward_bias=0.45,
        )
        env.step(targets)

    env.close()
    wall_dist_along = (wall_x_end - wall_x_start) if wall_x_start and wall_x_end else 0.0
    print(f"Summary:")
    print(f"  Hit Wall Speed   : {v_hit_wall:.2f} m/s")
    print(f"  Wall Height Range: {z_min_wall:.2f} m to {z_max_wall:.2f} m")
    print(f"  Wall Contact Time: {contact_steps/100:.2f} s")
    print(f"  Along Wall Dist  : {wall_dist_along:.2f} m")
    print(f"  Turns Completed  : {turns_now:.2f}")

if __name__ == "__main__":
    print("\n--- TEST: Apex High Jump (launch_up=5.5 m/s, speed=7.5 m/s, launch_gap=1.40 m, angle=22 deg) ---")
    test_jump("curved", speed=7.5, launch_up=5.5, launch_in=3.2, launch_gap=1.40, approach_angle=22.0)
    test_jump("banked", speed=7.5, launch_up=5.5, launch_in=3.2, launch_gap=1.40, approach_angle=22.0)
    test_jump("flat_multistep", speed=7.5, launch_up=5.5, launch_in=3.2, launch_gap=1.40, approach_angle=22.0)
