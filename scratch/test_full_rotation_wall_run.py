import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from omegaconf import OmegaConf
import imageio

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_run import FOOT_BASE, _frame
from skills.locomotion import move, stop

def test_full_turn_wall_run(
    approach_angle: float = 16.0,
    launch_in: float = 0.90,     # Gentle inward float (0.9 m/s) so it drifts through the 0.27m stroke over 0.35s
    launch_up: float = 3.20,     # High launch to give 0.4-0.5s airtime
    along_drive: float = 0.60,   # Active torque driving rolling rotation along wall
    lift_drive: float = 0.40,    # Active lift against gravity
    give: float = 0.22,
    squash_span: float = 0.28,
    cushion_max: float = 0.02,
    seconds: float = 7.0,
):
    cfg = load_config("configs/rl/wall_run.yaml")
    OmegaConf.set_struct(cfg, False)
    scenario = generate_scenario("wall_run", cfg, seed=3)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10 ** 7)
    env.reset(seed=3)
    mujoco.mj_forward(env.model, env.data)

    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "wall_0")
    pos_w = env.model.geom_pos[gid]
    size_w = env.model.geom_size[gid]
    n_hat = np.array([0.0, 1.0, 0.0])
    face_y = float(pos_w[1] - size_w[1])
    lane_gap = float(face_y - scenario.spawn_xy[1])
    travel = np.array([1.0, 0.0, 0.0])
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

    for step in range(int(seconds * 100)):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        angvel = env.data.qvel[3:6].copy()
        quat = env.data.qpos[3:7].copy()
        wall_dist = face_y - float(pos[1])
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
            # Yaw/pitch spin on the wall
            spin = float(np.linalg.norm(angvel))
            cum_rot_rad += spin * 0.01
            z_min_wall = min(z_min_wall, pos[2])
            z_max_wall = max(z_max_wall, pos[2])
            if wall_x_start is None:
                wall_x_start = pos[0]
            wall_x_end = pos[0]

        turns_completed = cum_rot_rad / (2.0 * np.pi)
        closing = float(np.dot(vel, n))

        if phase == "recover":
            head = t - np.tan(np.radians(approach_angle)) * n
            head = head / max(float(np.linalg.norm(head)), 1e-9)
            targets = move(quat, env.dirs_body, env.max_extend, np.array([head[0], head[1]]), speed=1.5)
            if wall_dist >= lane_gap:
                phase = "approach"

        elif phase == "approach":
            head = t + np.tan(np.radians(approach_angle)) * n
            head = head / max(float(np.linalg.norm(head)), 1e-9)
            targets = move(quat, env.dirs_body, env.max_extend, np.array([head[0], head[1]]), speed=4.5)
            along = float(np.linalg.norm(vel[:2]))
            if along >= 4.0 and wall_dist <= 1.05:
                phase = "launch"

        elif phase == "launch":
            want = launch_in * n + launch_up * np.array([0.0, 0.0, 1.0])
            want = want / max(float(np.linalg.norm(want)), 1e-9)
            aim = -(dirs_world @ want)
            short = max((launch_in - float(np.dot(vel, n))) / 0.60,
                        (launch_up - float(vel[2])) / 0.60)
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

            # Progressive soft absorption to eliminate violent rebound
            depth = max(full_reach - wall_dist, 0.0)
            soft = float(np.clip(depth / max(squash_span, 1e-6), 0.0, 1.0))
            cushion = soft * cushion_max - (1.0 - soft) * give

            targets = np.full(n_bars, env.max_extend, dtype=np.float64)
            targets[facing] = np.clip(reach[facing] + cushion, 0.0, env.max_extend)

            # Active along-wall drive: trailing rods push backward along -t to drive rotation
            if along_drive > 0.0:
                trailing = facing & (u_t < -0.05)
                targets[trailing] = np.clip(
                    targets[trailing] + along_drive * env.max_extend * (-u_t[trailing]),
                    0.0, env.max_extend
                )

            # Active anti-drop lift: lower rods push downward to float against gravity
            if lift_drive > 0.0:
                lower = facing & (u_z < -0.05)
                targets[lower] = np.clip(
                    targets[lower] + lift_drive * env.max_extend * (-u_z[lower]),
                    0.0, env.max_extend
                )

            # Push off only after completing >= 0.80-1.00 rotations OR core getting too close (<0.18m) OR falling below 0.35m
            if (turns_completed >= 1.00) or (wall_dist <= FOOT_BASE + 0.10 * env.max_extend) or (pos[2] < 0.38 and turns_completed >= 0.3):
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
    print(f"Result (launch_in={launch_in:.2f}, launch_up={launch_up:.2f}, drive={along_drive:.2f}):")
    print(f"  Time on wall : {contact_steps / 100.0:.2f} s")
    print(f"  Rotations    : {cum_rot_rad / (2.0 * np.pi):.2f} full turns ({cum_rot_rad:.2f} rad)")
    print(f"  Along wall   : {wall_dist_along:.2f} m")
    print(f"  Height range : {z_min_wall:.2f} m to {z_max_wall:.2f} m")
    print(f"  Max Height   : {max(h[2] for h in hist):.2f} m")
    print(f"  Final pos    : x={pos[0]:.2f}, z={pos[2]:.2f}, phase={phase}\n")
    return cum_rot_rad / (2.0 * np.pi), contact_steps / 100.0, wall_dist_along

if __name__ == "__main__":
    for lin in [0.40, 0.70, 1.00, 1.30]:
        for lup in [2.8, 3.2, 3.6]:
            test_full_turn_wall_run(launch_in=lin, launch_up=lup, along_drive=0.70, lift_drive=0.50)
