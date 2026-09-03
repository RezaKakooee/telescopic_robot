import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from omegaconf import OmegaConf
import imageio

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat
from skills.wall_run import FOOT_BASE, _frame
from skills.locomotion import move, stop

def get_wall_frame(pos, scenario):
    """Compute local wall_dist, wall_normal, and travel_tangent for any scenario mode."""
    mode = getattr(scenario, "wall_mode", "curved")
    wall_y = float(scenario.walls[0, 1]) if len(scenario.walls) > 0 else 1.30

    if mode == "curved":
        app_len = 4.0
        # Find closest wall segment in scenario.walls
        walls = np.asarray(scenario.walls, dtype=float)
        p = pos[:2]
        best_d = 999.0
        best_n = np.array([0.0, 1.0, 0.0])
        best_t = np.array([1.0, 0.0, 0.0])

        for (x1, y1, x2, y2) in walls:
            seg = np.array([x2 - x1, y2 - y1])
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-6:
                continue
            seg_u = seg / seg_len
            # Normal perpendicular to segment (pointing towards wall)
            seg_n = np.array([-seg_u[1], seg_u[0]])
            # Vector from segment start to robot
            v = p - np.array([x1, y1])
            proj = np.clip(np.dot(v, seg_u), 0.0, seg_len)
            closest = np.array([x1, y1]) + proj * seg_u
            d = np.linalg.norm(p - closest)
            if d < best_d:
                best_d = d
                # Check sign: robot is on lane side
                best_n = np.array([seg_n[0], seg_n[1], 0.0])
                best_t = np.array([seg_u[0], seg_u[1], 0.0])

        return best_d, best_n, best_t

    elif mode == "banked":
        bank_deg = float(getattr(scenario, "wall_bank_deg", 12.0))
        rad = np.radians(bank_deg)
        # Wall is tilted inward by bank_deg around X axis
        n = np.array([0.0, np.cos(rad), -np.sin(rad)])
        t = np.array([1.0, 0.0, 0.0])
        # Perpendicular distance from (pos[1], pos[2]) to plane y*cos(rad) - z*sin(rad) = wall_y*cos(rad)
        wall_dist = (wall_y - pos[1]) * np.cos(rad) - pos[2] * np.sin(rad)
        return max(wall_dist, 0.0), n, t

    else: # "flat" or "flat_multistep"
        wall_dist = wall_y - pos[1]
        n = np.array([0.0, 1.0, 0.0])
        t = np.array([1.0, 0.0, 0.0])
        return wall_dist, n, t

def run_mode_simulation(mode="curved", seconds=8.0, min_turns=1.00):
    cfg = load_config("configs/rl/wall_run.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.scenario.mode = mode
    if mode == "curved":
        cfg.scenario.curved.radius = 6.0
        cfg.scenario.curved.arc_deg = 70.0
    elif mode == "banked":
        cfg.scenario.banked.bank_deg = 75.0
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
    hist = []

    print(f"\n=======================================================")
    print(f"  Testing Mode: {mode.upper()} (Target: >= {min_turns:.1f} complete rotations)")
    print(f"=======================================================")

    for step in range(int(seconds * 100)):
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
            contact_steps += 1
            spin = float(np.linalg.norm(angvel))
            cum_rot_rad += spin * 0.01
            z_min_wall = min(z_min_wall, pos[2])
            z_max_wall = max(z_max_wall, pos[2])
            if wall_x_start is None:
                wall_x_start = pos[0]
            wall_x_end = pos[0]

        turns_completed = cum_rot_rad / (2.0 * np.pi)
        closing = float(np.dot(vel, n))

        # --- State Machine ---
        if phase == "recover":
            head = t - np.tan(np.radians(18.0)) * n
            head = head / max(float(np.linalg.norm(head)), 1e-9)
            targets = move(quat, env.dirs_body, env.max_extend, np.array([head[0], head[1]]), speed=1.5)
            if wall_dist >= lane_gap:
                phase = "approach"

        elif phase == "approach":
            head = t + np.tan(np.radians(18.0)) * n
            head = head / max(float(np.linalg.norm(head)), 1e-9)
            targets = move(quat, env.dirs_body, env.max_extend, np.array([head[0], head[1]]), speed=4.5)
            along = float(np.linalg.norm(vel[:2]))
            if along >= 3.8 and wall_dist <= 1.05:
                phase = "launch"

        elif phase == "launch":
            want = 2.0 * n + 2.8 * np.array([0.0, 0.0, 1.0])
            want = want / max(float(np.linalg.norm(want)), 1e-9)
            aim = -(dirs_world @ want)
            short = max((2.0 - float(np.dot(vel, n))) / 0.60,
                        (2.8 - float(vel[2])) / 0.60)
            gain = float(np.clip(short, 0.0, 1.0))
            wave = np.clip((aim - 0.15) / 0.85, 0.0, 1.0) * gain
            wave[u_z > 0.15] = 0.0
            targets = (0.025 + (env.max_extend - 0.025) * wave).astype(np.float32)
            if not on_ground and pos[2] > 0.30:
                phase = "fly"

        elif phase == "fly":
            targets = np.full(n_bars, env.max_extend, dtype=np.float32)
            if wall_dist <= full_reach:
                phase = "ride"

        elif phase == "ride":
            facing = u_n > 0.05
            dot = np.maximum(u_n, 1e-6)
            reach = np.where(facing, wall_dist / dot - FOOT_BASE, env.max_extend)

            depth = max(full_reach - wall_dist, 0.0)
            soft = float(np.clip(depth / max(0.26, 1e-6), 0.0, 1.0))
            cushion = soft * 0.05 - (1.0 - soft) * 0.18

            targets = np.full(n_bars, env.max_extend, dtype=np.float64)
            targets[facing] = np.clip(reach[facing] + cushion, 0.0, env.max_extend)

            # Active along-wall rolling wave: trailing rods push along -t with high torque to drive true rolling
            trailing = facing & (u_t < -0.02)
            targets[trailing] = np.clip(
                targets[trailing] + 1.20 * env.max_extend * (-u_t[trailing]),
                0.0, env.max_extend
            )
            # Upper-trailing rods also extend to roll the core forward
            lower = facing & (u_z < -0.02)
            targets[lower] = np.clip(
                targets[lower] + 0.60 * env.max_extend * (-u_z[lower]),
                0.0, env.max_extend
            )

            # Transition out of ride:
            if mode in ("curved", "banked"):
                # Ride continuously until target rotations completed or height drops too low
                if (turns_completed >= min_turns and contact_steps >= 30) or pos[2] < 0.35:
                    phase = "push"
            else:
                if closing <= 0.05 or wall_dist <= FOOT_BASE + 0.45 * env.max_extend:
                    phase = "push"

        elif phase == "push":
            facing = u_n > 0.05
            targets = np.full(n_bars, env.max_extend, dtype=np.float64)
            targets[facing] = 0.70 * env.max_extend
            clear = wall_dist > FOOT_BASE + 0.75 * env.max_extend
            if (clear and closing < -0.20) or wall_dist > FOOT_BASE + 0.95 * env.max_extend:
                phase = "land"

        elif phase == "land":
            targets = np.full(n_bars, env.max_extend, dtype=np.float32)
            if on_ground and abs(float(vel[2])) < 0.35:
                phase = "settle"

        elif phase == "settle":
            targets = stop(quat, env.dirs_body, env.max_extend, lin_vel=vel)

        targets = np.asarray(targets, dtype=np.float32)
        env.step(targets)
        hist.append((pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], phase, turns_completed, touching_wall))

    env.close()

    wall_dist_along = (wall_x_end - wall_x_start) if wall_x_start and wall_x_end else 0.0
    print(f"Summary for {mode.upper()}:")
    print(f"  Wall Contact Time : {contact_steps / 100.0:.2f} s")
    print(f"  Turns Completed   : {cum_rot_rad / (2.0 * np.pi):.2f} full rotations ({cum_rot_rad:.2f} rad)")
    print(f"  Along Wall Dist   : {wall_dist_along:.2f} m")
    print(f"  Height on Wall    : {z_min_wall:.2f} m to {z_max_wall:.2f} m")
    print(f"  Settles At        : z = {pos[2]:.2f} m")
    return cum_rot_rad / (2.0 * np.pi), contact_steps / 100.0

if __name__ == "__main__":
    run_mode_simulation("curved", seconds=6.0, min_turns=1.00)
    run_mode_simulation("banked", seconds=6.0, min_turns=1.00)
    run_mode_simulation("flat_multistep", seconds=6.0, min_turns=0.50)
