import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from omegaconf import OmegaConf

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_run import FOOT_BASE, _frame
from skills.locomotion import move, stop

def test_multistep_wall_run(
    n_wall_steps: int = 3,
    approach_angle: float = 20.0,
    speed: float = 4.5,
    seconds: float = 10.0,
):
    """Multi-step wall running: each wall touch provides upward lift and maintains proximity to the wall."""
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
    wall_touches = 0
    total_contact_steps = 0
    total_rot_rad = 0.0
    touch_in_progress = False
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
            total_contact_steps += 1
            total_rot_rad += float(np.linalg.norm(angvel)) * 0.01
            if not touch_in_progress:
                wall_touches += 1
                touch_in_progress = True
        else:
            touch_in_progress = False

        closing = float(np.dot(vel, n))

        if phase == "approach":
            head = t + np.tan(np.radians(approach_angle)) * n
            head = head / max(float(np.linalg.norm(head)), 1e-9)
            targets = move(quat, env.dirs_body, env.max_extend, np.array([head[0], head[1]]), speed=speed)
            along = float(np.linalg.norm(vel[:2]))
            if along >= 4.0 and wall_dist <= 1.05:
                phase = "launch"

        elif phase == "launch":
            want = 1.8 * n + 3.0 * np.array([0.0, 0.0, 1.0])
            want = want / max(float(np.linalg.norm(want)), 1e-9)
            aim = -(dirs_world @ want)
            short = max((1.8 - float(np.dot(vel, n))) / 0.60,
                        (3.0 - float(vel[2])) / 0.60)
            gain = float(np.clip(short, 0.0, 1.0))
            wave = np.clip((aim - 0.15) / 0.85, 0.0, 1.0) * gain
            wave[u_z > 0.15] = 0.0
            targets = (0.025 + (env.max_extend - 0.025) * wave).astype(np.float32)
            if not on_ground and pos[2] > 0.30:
                phase = "fly"

        elif phase == "fly":
            # Reach for the wall
            targets = np.full(n_bars, env.max_extend, dtype=np.float32)
            if wall_dist <= full_reach:
                phase = "wall_step"

        elif phase == "wall_step":
            # Active parkour stride against the wall:
            # - Trailing & bottom rods push DOWN and BACK against wall, creating UPWARD lift and forward speed
            # - Inward pressure to maintain grip
            facing = u_n > 0.05
            dot = np.maximum(u_n, 1e-6)
            reach = np.where(facing, wall_dist / dot - FOOT_BASE, env.max_extend)

            targets = np.full(n_bars, env.max_extend, dtype=np.float64)
            targets[facing] = np.clip(reach[facing], 0.0, env.max_extend)

            # Upward thrust off wall
            lower_trailing = facing & (u_z < 0.10)
            targets[lower_trailing] = np.clip(
                targets[lower_trailing] + 0.80 * env.max_extend,
                0.0, env.max_extend
            )

            # If moving away from wall or done with step
            if closing < -0.30 or wall_dist > full_reach:
                if wall_touches < n_wall_steps and pos[2] > 0.40:
                    phase = "wall_rebound"
                else:
                    phase = "push"

        elif phase == "wall_rebound":
            # Recoil and re-aim for next wall step
            targets = np.full(n_bars, env.max_extend, dtype=np.float32)
            if wall_dist <= full_reach:
                phase = "wall_step"
            elif pos[2] < 0.35:
                phase = "land"

        elif phase == "push":
            facing = u_n > 0.05
            targets = np.full(n_bars, env.max_extend, dtype=np.float64)
            targets[facing] = 0.70 * env.max_extend
            if wall_dist > FOOT_BASE + 0.80 * env.max_extend:
                phase = "land"

        elif phase == "land":
            targets = np.full(n_bars, env.max_extend, dtype=np.float32)
            if on_ground and abs(float(vel[2])) < 0.35:
                phase = "settle"

        elif phase == "settle":
            targets = stop(quat, env.dirs_body, env.max_extend, lin_vel=vel)

        targets = np.asarray(targets, dtype=np.float32)
        env.step(targets)
        hist.append((pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], phase, total_rot_rad / (2*np.pi), touching_wall))

    env.close()

    total_turns = total_rot_rad / (2.0 * np.pi)
    print(f"Multi-step Wall Run Result:")
    print(f"  Wall Touches    : {wall_touches}")
    print(f"  Total Contact   : {total_contact_steps / 100.0:.2f} s")
    print(f"  Total Rotations : {total_turns:.2f} full turns ({total_rot_rad:.2f} rad)")
    print(f"  Max Height      : {max(h[2] for h in hist):.2f} m")
    print(f"  Final pos       : x={pos[0]:.2f}, z={pos[2]:.2f}\n")
    return total_turns

if __name__ == "__main__":
    for steps in [1, 2, 3, 4]:
        print(f"Testing target {steps} wall steps:")
        test_multistep_wall_run(n_wall_steps=steps)
