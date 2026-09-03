import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from omegaconf import OmegaConf

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat
from skills.wall_run import FOOT_BASE, _frame, get_wall_frame
from skills.locomotion import move, stop

def test_crouch_takeoff(mode="curved", speed=7.5, launch_gap=1.55):
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
    crouch_timer = 0
    takeoff_timer = 0

    for step in range(int(5.0 * 100)):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        angvel = env.data.qvel[3:6].copy()
        quat = env.data.qpos[3:7].copy()

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

        # Crouch-takeoff state machine
        if phase == "approach":
            head = t + np.tan(np.radians(22.0)) * n
            head = head / max(float(np.linalg.norm(head)), 1e-9)
            targets = move(quat, env.dirs_body, env.max_extend, np.array([head[0], head[1]]), speed=speed)
            along = float(np.linalg.norm(vel[:2]))
            if along >= 5.2 and wall_dist <= launch_gap:
                phase = "crouch"
                crouch_timer = 0
                print(f"  t={step/100:.2f}s CROUCH: speed={along:.2f}m/s, gap={wall_dist:.2f}m")

        elif phase == "crouch":
            crouch_timer += 1
            targets = np.zeros(n_bars, dtype=np.float32)
            if crouch_timer >= 6:  # 0.06 s rapid crouch
                phase = "takeoff"
                takeoff_timer = 0
                print(f"  t={step/100:.2f}s TAKEOFF: explosive pop!")

        elif phase == "takeoff":
            takeoff_timer += 1
            # Ground and away-facing rods fire 100% stroke
            ground_mask = (u_z < 0.15) & (u_n < 0.20)
            targets = np.zeros(n_bars, dtype=np.float32)
            targets[ground_mask] = env.max_extend
            if takeoff_timer >= 10 or (not on_ground and pos[2] > 0.40):
                phase = "fly"
                print(f"  t={step/100:.2f}s FLY: apex climbing, pos[2]={pos[2]:.2f}m, vz={vel[2]:.2f}m/s, vy={np.dot(vel, n):.2f}m/s")

        elif phase == "fly":
            targets = np.full(n_bars, env.max_extend, dtype=np.float32)
            if wall_dist <= full_reach:
                phase = "ride"
                print(f"  t={step/100:.2f}s RIDE START: height z={pos[2]:.2f}m, speed={np.linalg.norm(vel):.2f}m/s")

        elif phase == "ride":
            facing = u_n > 0.05
            dot = np.maximum(u_n, 1e-6)
            reach = np.where(facing, wall_dist / dot - FOOT_BASE, env.max_extend)
            depth = max(full_reach - wall_dist, 0.0)
            soft = float(np.clip(depth / max(0.26, 1e-6), 0.0, 1.0))
            cushion = soft * 0.05 - (1.0 - soft) * 0.18

            targets = np.full(n_bars, env.max_extend, dtype=np.float64)
            targets[facing] = np.clip(reach[facing] + cushion, 0.0, env.max_extend)

            # Active along-wall traction & anti-gravity lift
            trailing = facing & (u_t < -0.02)
            targets[trailing] = np.clip(targets[trailing] + 0.90 * env.max_extend * (-u_t[trailing]), 0.0, env.max_extend)
            lower = facing & (u_z < -0.02)
            targets[lower] = np.clip(targets[lower] + 0.60 * env.max_extend * (-u_z[lower]), 0.0, env.max_extend)

            if mode in ("curved", "banked"):
                if (turns_now >= 1.0 and contact_steps >= 40) or pos[2] < 0.40:
                    phase = "push"
            else:
                if np.dot(vel, n) <= 0.05 or wall_dist <= FOOT_BASE + 0.45 * env.max_extend:
                    phase = "push"

        elif phase == "push":
            facing = u_n > 0.05
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
    print(f"  Hit Wall Speed   : {v_hit_wall:.2f} m/s")
    print(f"  Wall Height Range: {z_min_wall:.2f} m to {z_max_wall:.2f} m")
    print(f"  Wall Contact Time: {contact_steps/100:.2f} s")
    print(f"  Along Wall Dist  : {wall_dist_along:.2f} m")
    print(f"  Turns Completed  : {turns_now:.2f}")

if __name__ == "__main__":
    test_crouch_takeoff("curved", speed=7.5, launch_gap=1.60)
    test_crouch_takeoff("banked", speed=7.5, launch_gap=1.60)
    test_crouch_takeoff("flat_multistep", speed=7.5, launch_gap=1.60)
