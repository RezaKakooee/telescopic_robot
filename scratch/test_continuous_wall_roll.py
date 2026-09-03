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

def test_continuous_wall_roll(
    ride_min_time: float = 0.50,
    ride_min_turns: float = 1.00,
    along_torque_gain: float = 1.20,
    lift_gain: float = 0.80,
    adhesion_gain: float = 0.15,
    seconds: float = 6.0,
):
    """Test sustained active rolling along the vertical wall until at least 1 full rotation (2pi)."""
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
    cum_turn_rad = 0.0
    wall_rot_x_start = None
    wall_rot_x_end = None
    hist = []
    contact_pts = []

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
            # Rotation around normal (yaw spin relative to wall) + pitch spin
            spin_mag = float(np.linalg.norm(angvel))
            cum_turn_rad += spin_mag * 0.01
            contact_pts.append((pos[0], pos[1], pos[2]))
            if wall_rot_x_start is None:
                wall_rot_x_start = pos[0]
            wall_rot_x_end = pos[0]

        turns_completed = cum_turn_rad / (2.0 * np.pi)
        closing = float(np.dot(vel, n))

        # --- Enhanced State Machine for Sustained Wall Rolling ---
        if phase == "recover":
            head = t - np.tan(np.radians(18.0)) * n
            head = head / max(float(np.linalg.norm(head)), 1e-9)
            targets = move(quat, env.dirs_body, env.max_extend, np.array([head[0], head[1]]), speed=1.5)
            if wall_dist >= lane_gap:
                phase = "approach"

        elif phase == "approach":
            head = t + np.tan(np.radians(18.0)) * n
            head = head / max(float(np.linalg.norm(head)), 1e-9)
            targets = move(quat, env.dirs_body, env.max_extend, np.array([head[0], head[1]]), speed=4.0)
            along = float(np.linalg.norm(vel[:2]))
            if along >= 4.0 and wall_dist <= 1.05:
                phase = "launch"

        elif phase == "launch":
            want = 2.1 * n + 2.8 * np.array([0.0, 0.0, 1.0])
            want = want / max(float(np.linalg.norm(want)), 1e-9)
            aim = -(dirs_world @ want)
            short = max((2.1 - float(np.dot(vel, n))) / 0.60,
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
            # --- Continuous Active Wall Rolling Controller ---
            # Wall contact geometry
            facing = u_n > 0.05
            dot = np.maximum(u_n, 1e-6)
            reach = np.where(facing, wall_dist / dot - FOOT_BASE, env.max_extend)

            # Base conformal surface tracking
            targets = np.full(n_bars, env.max_extend, dtype=np.float64)
            targets[facing] = np.clip(reach[facing], 0.0, env.max_extend)

            # Active wall rolling wave:
            # 1. Trailing rods (u_t < -0.05) push backward along wall with high torque to drive continuous rolling rotation
            trailing = facing & (u_t < -0.05)
            targets[trailing] = np.clip(
                targets[trailing] + along_torque_gain * env.max_extend * (-u_t[trailing]),
                0.0, env.max_extend
            )

            # 2. Lower rods (u_z < -0.05) push downward into the wall to provide continuous vertical lift
            lower = facing & (u_z < -0.05)
            targets[lower] = np.clip(
                targets[lower] + lift_gain * env.max_extend * (-u_z[lower]),
                0.0, env.max_extend
            )

            # 3. Wall adhesion/inward bias: keep gentle normal pressure into the wall to maintain contact
            targets[facing] = np.clip(
                targets[facing] + adhesion_gain * env.max_extend,
                0.0, env.max_extend
            )

            # Check if rotation goal is reached or robot has descended
            # Only transition to push if we have completed >= ride_min_turns OR dropped too low
            if (turns_completed >= ride_min_turns and contact_steps >= int(ride_min_time * 100)) or (pos[2] < 0.35 and turns_completed >= 0.5):
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
        hist.append((pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], phase, turns_completed))

    env.close()

    wall_dist_along = (wall_rot_x_end - wall_rot_x_start) if wall_rot_x_start and wall_rot_x_end else 0.0
    print(f"Results for ride_min_turns={ride_min_turns:.2f}, torque={along_torque_gain:.2f}, lift={lift_gain:.2f}:")
    print(f"  Wall Contact Time : {contact_steps / 100.0:.2f} s")
    print(f"  Turns Completed   : {cum_turn_rad / (2.0 * np.pi):.2f} full rotations ({cum_turn_rad:.2f} rad)")
    print(f"  Wall Distance     : {wall_dist_along:.2f} m along wall")
    print(f"  Max Height        : {max(h[2] for h in hist):.2f} m")
    print(f"  Final State       : x={pos[0]:.2f}, z={pos[2]:.2f}, phase={phase}")
    return cum_turn_rad / (2.0 * np.pi), contact_steps / 100.0, wall_dist_along

if __name__ == "__main__":
    for tg in [0.8, 1.2, 1.6, 2.0]:
        for lg in [0.5, 0.8, 1.2]:
            test_continuous_wall_roll(ride_min_turns=1.0, along_torque_gain=tg, lift_gain=lg)
