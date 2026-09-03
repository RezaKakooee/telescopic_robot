import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.geometry import quat_to_rotmat

def main():
    cfg = load_config("configs/rl/circle_track.yaml")
    scenario = generate_scenario("motordrome", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    env.data.qpos[0] = 1.00
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.22
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    for step in range(80):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        z = float(pos[2])
        x = float(pos[0])
        y = float(pos[1])
        spd = float(np.linalg.norm(vel[:2]))
        r = float(np.hypot(x, y))
        theta = float(np.arctan2(y, x))

        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        r_hat = np.array([np.cos(theta), np.sin(theta), 0.0])
        theta_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])

        r_err = r - 1.00
        d_steer = theta_hat - 0.30 * r_err * r_hat
        d_steer /= np.linalg.norm(d_steer)

        u_long = dirs_world @ d_steer
        u_norm = dirs_world[:, 2]

        rear = np.clip((-u_long - 0.05) / 0.95, 0.0, 1.0)
        down = np.clip(1.0 - abs(u_norm + 0.35) / 0.85, 0.0, 1.0)
        wave = np.clip((rear ** 1.0) * down * 3.6, 0.0, 1.0)
        wave[u_long > -0.05] = 0.0
        wave[u_norm > 0.10] = 0.0

        targets = 0.025 + (env.max_extend - 0.025) * wave
        env.step(targets)

        if step >= 45:
            # Print contacts
            contacts = []
            for ci in range(env.data.ncon):
                c = env.data.contact[ci]
                g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
                g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
                contacts.append(f"{g1}<->{g2}(dist={c.dist:.3f})")
            print(f"Step {step:2d}: r={r:.2f}m | z={z:.3f}m | spd={spd:.2f}m/s | th={np.degrees(theta):.1f}° | ncon={env.data.ncon} | {contacts[:4]}")

if __name__ == "__main__":
    main()
