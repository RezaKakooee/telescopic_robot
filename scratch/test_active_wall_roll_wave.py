import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from omegaconf import OmegaConf

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat
from skills.wall_run import FOOT_BASE, _frame
from skills.locomotion import move, stop
from scratch.test_modular_wall_run_modes import get_wall_frame

def test_active_wall_roll_wave(mode="curved", seconds=6.0):
    cfg = load_config("configs/rl/wall_run.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.scenario.mode = mode
    if mode == "curved":
        cfg.scenario.curved.radius = 7.0
        cfg.scenario.curved.arc_deg = 60.0
    elif mode == "banked":
        cfg.scenario.banked.bank_deg = 76.0

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
    ride_step = 0

    print(f"\n=======================================================")
    print(f"  Testing Mode: {mode.upper()} with Active Rolling Wave")
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

        if phase == "approach":
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
                ride_step = 0

        elif phase == "ride":
            ride_step += 1
            facing = u_n > 0.05
            dot = np.maximum(u_n, 1e-6)
            reach = np.where(facing, wall_dist / dot - FOOT_BASE, env.max_extend)

            depth = max(full_reach - wall_dist, 0.0)
            soft = float(np.clip(depth / max(0.26, 1e-6), 0.0, 1.0))
            cushion = soft * 0.05 - (1.0 - soft) * 0.18

            targets = np.full(n_bars, env.max_extend, dtype=np.float64)
            targets[facing] = np.clip(reach[facing] + cushion, 0.0, env.max_extend)

            # Active wall-rolling CPG: rotate the contact wave at omega = 12 rad/s
            cpg_phase = (ride_step * 0.12) % (2.0 * np.pi)
            angle_in_tangent_plane = np.arctan2(u_z, u_t)
            roll_wave = np.sin(angle_in_tangent_plane - cpg_phase)
            targets[facing] = np.clip(
                targets[facing] + 0.30 * env.max_extend * roll_wave[facing],
                0.0, env.max_extend
            )

            # Trailing drive & lower lift
            trailing = facing & (u_t < -0.02)
            targets[trailing] = np.clip(
                targets[trailing] + 0.80 * env.max_extend * (-u_t[trailing]),
                0.0, env.max_extend
            )
            lower = facing & (u_z < -0.02)
            targets[lower] = np.clip(
                targets[lower] + 0.50 * env.max_extend * (-u_z[lower]),
                0.0, env.max_extend
            )

            if mode in ("curved", "banked"):
                if (turns_completed >= 1.00 and contact_steps >= 40) or pos[2] < 0.35:
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
    test_active_wall_roll_wave("curved", seconds=6.0)
    test_active_wall_roll_wave("banked", seconds=6.0)
    test_active_wall_roll_wave("flat_multistep", seconds=6.0)
