import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from omegaconf import OmegaConf

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat
from skills.wall_run import FOOT_BASE, _frame, get_wall_frame, wall_reach
from skills.locomotion import move, stop

def test_extended_runway(mode="curved", app_len=10.0, back_gain=6.0, launch_up=5.5, launch_in=3.8, approach_angle=20.0):
    cfg = load_config("configs/rl/wall_run.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.scenario.mode = mode
    cfg.scenario.lane_offset = 2.80
    if mode == "curved":
        cfg.scenario.curved.radius = 8.0
        cfg.scenario.curved.arc_deg = 50.0

    # Build scenario with extended runway
    scenario = generate_scenario("wall_run", cfg, seed=3)
    # If curved, update app_len in scenario walls
    if mode == "curved":
        wall_y = 1.30
        radius = 8.0
        arc_deg = 50.0
        arc_rad = np.radians(arc_deg)
        wall_segs = [[0.0, wall_y, app_len, wall_y]]
        n_facets = 32
        alphas = np.linspace(0.0, arc_rad, n_facets + 1)
        arc_xs = app_len + radius * np.sin(alphas)
        arc_ys = (wall_y - radius) + radius * np.cos(alphas)
        for i in range(n_facets):
            wall_segs.append([arc_xs[i], arc_ys[i], arc_xs[i+1], arc_ys[i+1]])
        exit_dir = np.array([np.cos(arc_rad), -np.sin(arc_rad)])
        last_pt = np.array([arc_xs[-1], arc_ys[-1]])
        exit_end = last_pt + 30.0 * exit_dir
        wall_segs.append([last_pt[0], last_pt[1], exit_end[0], exit_end[1]])
        scenario.walls = np.array(wall_segs, dtype=np.float32)

    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10 ** 7)
    env.reset(seed=3)
    mujoco.mj_forward(env.model, env.data)

    n_bars = len(env.dirs_body)
    full_reach = FOOT_BASE + env.max_extend

    phase = "approach"
    contact_steps = 0
    cum_rot_rad = 0.0
    wall_x_start = None
    wall_x_end = None
    z_min_wall = 999.0
    z_max_wall = -999.0
    v_hit_wall = 0.0
    max_compression_mm = 0.0
    peak_speed = 0.0

    print(f"\n==========================================================================")
    print(f"  EXTENDED RUNWAY TEST: {mode.upper()} (runway={app_len}m, back_gain={back_gain})")
    print(f"==========================================================================")

    for step in range(int(6.0 * 100)):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        angvel = env.data.qvel[3:6].copy()
        quat = env.data.qpos[3:7].copy()

        speed_now = float(np.linalg.norm(vel))
        peak_speed = max(peak_speed, speed_now)

        wall_dist, n_hat, travel = get_wall_frame(pos, scenario)
        dirs_world, n, t = _frame(quat, env.dirs_body, n_hat, travel)
        u_n = dirs_world @ n
        u_t = dirs_world @ t
        u_z = dirs_world[:, 2]

        on_ground = any(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom1) == "floor" or
                        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom2) == "floor"
                        for c_i in range(env.data.ncon))

        touching_wall = any((mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom1) or "").startswith("wall_") or
                            (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom2) or "").startswith("wall_")
                            for c_i in range(env.data.ncon))

        rod_extensions = env.data.qpos[7:7+n_bars].copy()
        compression = (env.max_extend - rod_extensions) * 1000.0

        if touching_wall:
            if contact_steps == 0:
                v_hit_wall = speed_now
                print(f"  --> 🔥 SLAM INTO WALL at t={step/100:.2f}s: x={pos[0]:.2f}m, height z={pos[2]:.2f}m, speed v={v_hit_wall:.2f}m/s (along: {np.dot(vel, t):.2f}, into wall: {np.dot(vel, n):.2f})")
            contact_steps += 1
            spin = float(np.linalg.norm(angvel))
            cum_rot_rad += spin * 0.01
            z_min_wall = min(z_min_wall, pos[2])
            z_max_wall = max(z_max_wall, pos[2])
            max_compression_mm = max(max_compression_mm, float(np.max(compression)))
            if wall_x_start is None:
                wall_x_start = pos[0]
            wall_x_end = pos[0]

        turns_now = cum_rot_rad / (2.0 * np.pi)

        # 1. Extended Ground Acceleration Approach
        if phase == "approach":
            # If still early on the runway, sprint straight along +X to reach maximum acceleration
            if mode == "curved" and pos[0] < (app_len - 3.5):
                head = np.array([1.0, 0.0, 0.0])
            else:
                head = t + np.tan(np.radians(approach_angle)) * n
                head = head / max(float(np.linalg.norm(head)), 1e-9)

            targets = move(quat, env.dirs_body, env.max_extend, np.array([head[0], head[1]]), back_gain=back_gain)
            along = float(np.linalg.norm(vel[:2]))

            # Launch triggered when reaching launch gap with high speed
            launch_gap = 1.65
            if pos[0] >= (app_len - 2.5) if mode == "curved" else True:
                if along >= 6.5 and wall_dist <= launch_gap:
                    phase = "launch"
                    print(f"  t={step/100:.2f}s LAUNCH: x={pos[0]:.2f}m, speed={along:.2f}m/s, gap={wall_dist:.2f}m, vx={vel[0]:.2f}, vy={vel[1]:.2f}")

        # 2. Explosive Ground Launch
        elif phase == "launch":
            want = launch_in * n + launch_up * np.array([0.0, 0.0, 1.0])
            want = want / max(float(np.linalg.norm(want)), 1e-9)
            aim = -(dirs_world @ want)
            gain = float(np.clip(max((launch_in - float(np.dot(vel, n))) / 0.50,
                                     (launch_up - float(vel[2])) / 0.50), 0.0, 1.0))
            wave = np.clip((aim - 0.05) / 0.95, 0.0, 1.0) * gain
            wave[u_z > 0.08] = 0.0
            targets = (0.025 + (env.max_extend - 0.025) * wave).astype(np.float32)

            if not on_ground and pos[2] > 0.35:
                phase = "fly"
                print(f"  t={step/100:.2f}s FLY: ALL RODS OPEN 100%! Apex climbing, z={pos[2]:.2f}m, v={np.linalg.norm(vel):.2f}m/s, vz={vel[2]:.2f}m/s")

        # 3. Mid-Air: ALL 60 RODS FULLY OPEN (0.30m)
        elif phase == "fly":
            targets = np.full(n_bars, env.max_extend, dtype=np.float32)
            if wall_dist <= full_reach:
                phase = "ride"
                print(f"  t={step/100:.2f}s RIDE/COMPRESSION: Impacting wall with all rods open! z={pos[2]:.2f}m, speed={np.linalg.norm(vel):.2f}m/s")

        # 4. Ride & Deep Compression: All rods touching/facing wall get compressed
        elif phase == "ride":
            reach, facing = wall_reach(dirs_world, n, wall_dist, env.max_extend)
            depth = max(full_reach - wall_dist, 0.0)
            soft = float(np.clip(depth / 0.28, 0.0, 1.0))
            cushion = soft * 0.04 - (1.0 - soft) * 0.22

            targets = np.full(n_bars, env.max_extend, dtype=np.float64)
            # Wall-contact rods deeply compress
            targets[facing] = np.clip(reach[facing] + cushion, 0.0, env.max_extend)

            # Trailing rods push along -t to maintain forward rolling torque
            trailing = facing & (u_t < -0.02)
            targets[trailing] = np.clip(targets[trailing] + 1.20 * env.max_extend * (-u_t[trailing]), 0.0, env.max_extend)
            # Lower rods push downward into wall
            lower = facing & (u_z < -0.02)
            targets[lower] = np.clip(targets[lower] + 0.60 * env.max_extend * (-u_z[lower]), 0.0, env.max_extend)

            if mode in ("curved", "banked"):
                if (turns_now >= 1.0 and contact_steps >= 40) or pos[2] < 0.40:
                    phase = "push"
            else:
                if np.dot(vel, n) <= 0.05 or wall_dist <= FOOT_BASE + 0.45 * env.max_extend:
                    phase = "push"

        elif phase == "push":
            reach, facing = wall_reach(dirs_world, n, wall_dist, env.max_extend)
            targets = np.full(n_bars, env.max_extend, dtype=np.float64)
            targets[facing] = 0.70 * env.max_extend
            clear = wall_dist > FOOT_BASE + 0.75 * env.max_extend
            if (clear and np.dot(vel, n) < -0.20) or wall_dist > FOOT_BASE + 0.95 * env.max_extend:
                phase = "land"

        elif phase == "land":
            targets = np.full(n_bars, env.max_extend, dtype=np.float32)
            if on_ground and abs(float(vel[2])) < 0.35:
                phase = "settle"

        elif phase == "settle":
            targets = stop(quat, env.dirs_body, env.max_extend, lin_vel=vel)

        targets = np.asarray(targets, dtype=np.float32)
        env.step(targets)

    env.close()
    wall_dist_along = (wall_x_end - wall_x_start) if wall_x_start and wall_x_end else 0.0
    print(f"\nSummary for {mode.upper()}:")
    print(f"  Peak Ground Speed: {peak_speed:.2f} m/s")
    print(f"  Hit Wall Speed   : {v_hit_wall:.2f} m/s")
    print(f"  Max Rod Compress : {max_compression_mm:.1f} mm of 300 mm stroke")
    print(f"  Wall Height Range: {z_min_wall:.2f} m to {z_max_wall:.2f} m")
    print(f"  Wall Contact Time: {contact_steps/100:.2f} s")
    print(f"  Along Wall Dist  : {wall_dist_along:.2f} m")
    print(f"  Turns Completed  : {turns_now:.2f}")

if __name__ == "__main__":
    test_extended_runway("curved", app_len=10.0, back_gain=6.0, launch_up=5.8, launch_in=3.8)
    test_extended_runway("banked", app_len=10.0, back_gain=6.0, launch_up=5.8, launch_in=3.8)
    test_extended_runway("flat_multistep", app_len=10.0, back_gain=6.0, launch_up=5.8, launch_in=3.8)
