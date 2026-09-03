import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from pathlib import Path
from omegaconf import OmegaConf
import imageio

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat
from skills.wall_run import FOOT_BASE, next_phase, wall_reach, _frame
from skills.locomotion import move, stop

def wall_run_advanced(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    *,
    phase: str,
    wall_normal: np.ndarray,
    wall_dist: float,
    travel: np.ndarray,
    lin_vel: np.ndarray,
    speed: float = 4.0,
    recover_speed: float = 1.5,
    approach_angle: float = 24.0,
    launch_in: float = 2.1,
    launch_up: float = 2.8,
    servo_band: float = 0.60,
    give: float = 0.18,
    squash_span: float = 0.26,
    cushion_max: float = 0.05,
    push_frac: float = 0.70,
    along_drive: float = 0.0,
    min_offset: float = 0.025,
) -> np.ndarray:
    vel = np.asarray(lin_vel, dtype=np.float64)
    dirs_world, n, t = _frame(quat, dirs_body, wall_normal, travel)
    u_n = dirs_world @ n
    u_t = dirs_world @ t
    u_z = dirs_world[:, 2]
    n_bars = len(dirs_body)
    full_reach = FOOT_BASE + max_extend

    if phase == "recover":
        head = t - np.tan(np.radians(approach_angle)) * n
        head = head / max(float(np.linalg.norm(head)), 1e-9)
        return move(quat, dirs_body, max_extend, np.array([head[0], head[1]]), speed=recover_speed)

    if phase == "approach":
        head = t + np.tan(np.radians(approach_angle)) * n
        head = head / max(float(np.linalg.norm(head)), 1e-9)
        return move(quat, dirs_body, max_extend, np.array([head[0], head[1]]), speed=speed)

    if phase == "launch":
        want = launch_in * n + launch_up * np.array([0.0, 0.0, 1.0])
        want = want / max(float(np.linalg.norm(want)), 1e-9)
        aim = -(dirs_world @ want)
        short = max((launch_in - float(np.dot(vel, n))) / servo_band,
                    (launch_up - float(vel[2])) / servo_band)
        gain = float(np.clip(short, 0.0, 1.0))
        wave = np.clip((aim - 0.15) / 0.85, 0.0, 1.0) * gain
        wave[u_z > 0.15] = 0.0
        return (min_offset + (max_extend - min_offset) * wave).astype(np.float32)

    if phase == "settle":
        return stop(quat, dirs_body, max_extend, lin_vel=vel)

    if phase in ("fly", "land"):
        return np.full(n_bars, max_extend, dtype=np.float32)

    reach, facing = wall_reach(dirs_world, n, wall_dist, max_extend)

    if phase == "ride":
        depth = max(full_reach - wall_dist, 0.0)
        soft = float(np.clip(depth / max(squash_span, 1e-6), 0.0, 1.0))
        cushion = soft * cushion_max - (1.0 - soft) * give
        targets = np.full(n_bars, max_extend, dtype=np.float64)
        targets[facing] = np.clip(reach[facing] + cushion, 0.0, max_extend)

        # Active along-wall drive / spin:
        # Rods pointing backward relative to travel direction push along wall to spin and maintain forward speed
        if along_drive > 0.0:
            # Trailing rods (pointing along -t) push backward against wall to add forward rolling torque
            trailing_wall = facing & (u_t < -0.10)
            targets[trailing_wall] = np.clip(
                targets[trailing_wall] + along_drive * max_extend * (-u_t[trailing_wall]),
                0.0, max_extend
            )
        return targets.astype(np.float32)

    if phase == "push":
        targets = np.full(n_bars, max_extend, dtype=np.float64)
        targets[facing] = float(push_frac) * max_extend
        return targets.astype(np.float32)

    raise ValueError(f"unknown phase {phase!r}")

def evaluate_params(approach_angle=24.0, launch_up=2.8, launch_in=2.1, along_drive=0.0, give=0.18, squash_span=0.26, cushion_max=0.05, push_frac=0.70, seconds=14.0):
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

    phase = "approach"
    runs = 1
    per_run = [dict(contact=[], turns=0.0)]
    gap_at = [999.0]
    hist = []

    for i in range(int(seconds * 100)):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        wall_dist = face_y - float(pos[1])
        on_ground = any(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom1) == "floor" or
                        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom2) == "floor"
                        for c_i in range(env.data.ncon))

        nxt = next_phase(
            phase, wall_dist=wall_dist, lin_vel=vel, height=float(pos[2]),
            on_ground=on_ground, max_extend=env.max_extend, speed_ready=4.0,
            launch_gap=1.05, lane_gap=lane_gap, wall_normal=n_hat,
        )
        if phase == "settle" and runs < 2 and float(np.linalg.norm(vel)) < 0.25:
            runs += 1
            nxt = "recover"
            per_run.append(dict(contact=[], turns=0.0))
        phase = nxt

        targets = wall_run_advanced(
            env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
            phase=phase, wall_normal=n_hat, wall_dist=wall_dist,
            travel=travel, lin_vel=vel, speed=4.0,
            approach_angle=approach_angle, recover_speed=1.5,
            launch_in=launch_in, launch_up=launch_up,
            give=give, squash_span=squash_span,
            cushion_max=cushion_max, push_frac=push_frac,
            along_drive=along_drive,
        )
        env.step(targets)

        touching_wall = any((mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom1) or "").startswith("wall_") or
                            (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom2) or "").startswith("wall_")
                            for c_i in range(env.data.ncon))

        if touching_wall:
            gap_at.append(wall_dist)
            per_run[-1]["contact"].append((float(pos[0]), float(pos[2])))
            per_run[-1]["turns"] += float(np.linalg.norm(env.data.qvel[3:6])) * 0.01

        hist.append((float(pos[0]), float(pos[1]), float(pos[2]), phase, float(np.dot(vel, n_hat))))

    env.close()

    r1 = per_run[0]
    c1 = r1["contact"]
    if c1:
        xs = [p[0] for p in c1]
        zs = [p[1] for p in c1]
        t_wall = len(c1) / 100.0
        d_wall = max(xs) - min(xs)
        z_min, z_max = min(zs), max(zs)
        turns = r1["turns"] / (2 * np.pi)
    else:
        t_wall, d_wall, z_min, z_max, turns = 0.0, 0.0, 0.0, 0.0, 0.0

    peak_z = max(h[2] for h in hist)
    squash = full_reach - min(gap_at)
    return {
        "angle": approach_angle,
        "launch_up": launch_up,
        "launch_in": launch_in,
        "along_drive": along_drive,
        "t_wall": t_wall,
        "d_wall": d_wall,
        "z_range": (z_min, z_max),
        "turns": turns,
        "squash_mm": squash * 1000,
        "peak_z": peak_z,
    }

def main():
    print("=== Sweep 1: Approach Angle ===")
    for ang in [14.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]:
        res = evaluate_params(approach_angle=ang)
        print(f"Angle {ang:4.1f}°: t_wall={res['t_wall']:.2f}s | along={res['d_wall']:.2f}m | z={res['z_range'][0]:.2f}-{res['z_range'][1]:.2f}m | turns={res['turns']:.2f} | squash={res['squash_mm']:.0f}mm")

    print("\n=== Sweep 2: Along-Wall Drive during Ride ===")
    for ad in [0.0, 0.15, 0.30, 0.50, 0.70]:
        res = evaluate_params(along_drive=ad)
        print(f"Along-Drive {ad:4.2f}: t_wall={res['t_wall']:.2f}s | along={res['d_wall']:.2f}m | z={res['z_range'][0]:.2f}-{res['z_range'][1]:.2f}m | turns={res['turns']:.2f} | squash={res['squash_mm']:.0f}mm")

    print("\n=== Sweep 3: Shallower Angle + Along-Wall Drive ===")
    for ang in [16.0, 18.0, 20.0]:
        for ad in [0.0, 0.30, 0.50]:
            res = evaluate_params(approach_angle=ang, along_drive=ad)
            print(f"Angle {ang:4.1f}° + Drive {ad:4.2f}: t_wall={res['t_wall']:.2f}s | along={res['d_wall']:.2f}m | z={res['z_range'][0]:.2f}-{res['z_range'][1]:.2f}m | turns={res['turns']:.2f} | squash={res['squash_mm']:.0f}mm")

if __name__ == "__main__":
    main()
