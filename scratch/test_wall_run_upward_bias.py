import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from omegaconf import OmegaConf

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat
from skills.wall_run import FOOT_BASE, next_phase, wall_reach, _frame
from skills.locomotion import move, stop

def wall_run_enhanced(
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
    approach_angle: float = 18.0,
    launch_in: float = 2.1,
    launch_up: float = 2.8,
    servo_band: float = 0.60,
    give: float = 0.18,
    squash_span: float = 0.26,
    cushion_max: float = 0.05,
    push_frac: float = 0.70,
    along_drive: float = 0.40,
    upward_bias: float = 0.0,
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

        # 1. Along-wall active drive (trailing rods push backward against wall)
        if along_drive > 0.0:
            trailing = facing & (u_t < -0.10)
            targets[trailing] = np.clip(
                targets[trailing] + along_drive * max_extend * (-u_t[trailing]),
                0.0, max_extend
            )

        # 2. Upward anti-drop bias: rods below the center push downward-inward into wall to lift the core
        if upward_bias > 0.0:
            lower_wall = facing & (u_z < -0.10)
            targets[lower_wall] = np.clip(
                targets[lower_wall] + upward_bias * max_extend * (-u_z[lower_wall]),
                0.0, max_extend
            )

        return targets.astype(np.float32)

    if phase == "push":
        # Targeted push off: push off wall AND upward/along travel
        targets = np.full(n_bars, max_extend, dtype=np.float64)
        targets[facing] = float(push_frac) * max_extend
        return targets.astype(np.float32)

    raise ValueError(f"unknown phase {phase!r}")

def test_config(approach_angle=18.0, launch_up=2.8, along_drive=0.40, upward_bias=0.0, repeats=3):
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

    for i in range(int(24.0 * 100)):
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
        if phase == "settle" and runs < repeats and float(np.linalg.norm(vel)) < 0.25:
            runs += 1
            nxt = "recover"
            per_run.append(dict(contact=[], turns=0.0))
        phase = nxt

        targets = wall_run_enhanced(
            env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
            phase=phase, wall_normal=n_hat, wall_dist=wall_dist,
            travel=travel, lin_vel=vel, speed=4.0,
            approach_angle=approach_angle, recover_speed=1.5,
            launch_in=2.1, launch_up=launch_up,
            along_drive=along_drive, upward_bias=upward_bias,
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

    after = [h for h in hist if h[3] in ("push", "land")]
    exit_v = max((-h[4] for h in after), default=0.0)
    end_z = float(np.mean([h[2] for h in hist[-150:]]))
    squash = full_reach - min(gap_at)

    print(f"Angle {approach_angle:.1f}°, Upward {upward_bias:.2f}, Drive {along_drive:.2f}:")
    for k, r in enumerate(per_run, 1):
        c = r["contact"]
        if c:
            xs = [p[0] for p in c]
            zs = [p[1] for p in c]
            print(f"  Run {k}: {len(c)/100:.2f}s | {max(xs)-min(xs):.2f}m along | z={min(zs):.2f}-{max(zs):.2f}m | {r['turns']/(2*np.pi):.2f} turns")
        else:
            print(f"  Run {k}: missed wall")
    print(f"  Squash: {squash*1000:.0f}mm | Exit_v: {exit_v:.2f}m/s | End_z: {end_z:.2f}m\n")

if __name__ == "__main__":
    for ub in [0.0, 0.20, 0.40, 0.60]:
        test_config(approach_angle=18.0, along_drive=0.40, upward_bias=ub)
