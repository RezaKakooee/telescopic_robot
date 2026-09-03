import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from omegaconf import OmegaConf

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_run import FOOT_BASE, _frame
from skills.locomotion import move

def diagnose():
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

    for step in range(300):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        wall_dist = face_y - float(pos[1])
        dirs_world, n, t = _frame(quat, env.dirs_body, n_hat, travel)
        u_n = dirs_world @ n
        u_t = dirs_world @ t
        u_z = dirs_world[:, 2]

        touching_wall = any((mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom1) or "").startswith("wall_") or
                            (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom2) or "").startswith("wall_")
                            for c_i in range(env.data.ncon))

        if phase == "approach":
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
            if pos[2] > 0.30:
                phase = "fly"

        elif phase == "fly":
            targets = np.full(n_bars, env.max_extend, dtype=np.float32)
            if wall_dist <= full_reach:
                phase = "ride"

        elif phase == "ride":
            facing = u_n > 0.05
            dot = np.maximum(u_n, 1e-6)
            reach = np.where(facing, wall_dist / dot - FOOT_BASE, env.max_extend)
            targets = np.full(n_bars, env.max_extend, dtype=np.float64)
            # What happens if we command reach exactly?
            targets[facing] = np.clip(reach[facing], 0.0, env.max_extend)

        targets = np.asarray(targets, dtype=np.float32)
        env.step(targets)

        if 100 <= step <= 180:
            print(f"t={step/100:4.2f}s [{phase:4s}] x={pos[0]:5.2f} y={pos[1]:5.2f} z={pos[2]:5.2f} | vy={vel[1]:+5.2f} vz={vel[2]:+5.2f} | gap={wall_dist:5.3f} touch={touching_wall}")

    env.close()

if __name__ == "__main__":
    diagnose()
