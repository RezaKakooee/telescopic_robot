"""Kinematic Flat-Plane Stance & Tangential Shear Controller Simulation & Rendering."""
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
import rootutils
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

rootutils.setup_root("/home/azureuser/telescopic_robot", pythonpath=True)

from radial_sphere import (
    MujocoSteeringEnv,
    generate_scenario,
    load_config_cli,
)
from radial_sphere.geometry import quat_to_rotmat

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/flat_plane_kinematics_benchmark")
renders_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")

model_dir = Path("/home/azureuser/telescopic_robot/storage_local/radial__20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
model_path = model_dir / "checkpoints" / "ppo_final.zip"
norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

cfg = load_config_cli(name="maze_level3_large_active_braking")
OmegaConf.set_struct(cfg, False)

cfg.robot.core_mass = 3.5  # Realistic payload mass
cfg.robot.kp = 500.0
cfg.robot.kv = 35.0

cfg.scenario.maze.level = 3
cfg.scenario.maze.random_endpoints = False
cfg.scenario.maze.random_start = False
cfg.scenario.maze.random_goal = False
cfg.scenario.maze.layout_seed = 42

sc = generate_scenario("maze", cfg, seed=42)


def flat_plane_kinematic_bar_targets(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    drive: float = 1.0,
    min_offset: float = 0.025,
    r_core: float = 0.20,
    r_foot: float = 0.015,
    h_nominal: float = 0.275,
    push_gain: float = 0.12,
) -> np.ndarray:
    """Compute per-bar targets constrained to a mathematically flat ground plane envelope.

    1. Stance Envelope: For all downward-pointing rods (u_z < -0.30), targets are calculated
       so that every rod tip touches the flat floor (z=0) at nominal core height h_nominal.
    2. Tangential Propulsion Shear: Trailing rods (u_long < 0) apply rearward horizontal thrust
       by expanding outward behind the contact patch.
    3. Clearance Envelope: Non-contact rods smoothly retract to min_offset.
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_z = dirs_world[:, 2]

    targets = np.full(len(dirs_body), min_offset, dtype=np.float32)

    # 1. Flat ground plane stance constraint: z_core + (r_core + u_k + r_foot)*u_z = r_foot
    # u_k = (h_nominal - r_foot) / (-u_z) - r_core - r_foot
    is_down = u_z < -0.20
    cos_down = np.maximum(-u_z[is_down], 0.20)
    stance_lengths = (h_nominal - r_foot) / cos_down - r_core - r_foot
    targets[is_down] = np.clip(stance_lengths, min_offset, max_extend)

    # 2. Tangential propulsion wave on trailing hemisphere
    is_trailing = (u_long < 0.0) & (u_z < 0.30)
    rear_factor = np.clip(-u_long[is_trailing], 0.0, 1.0)
    down_bias = 0.35 + 0.65 * np.clip(-u_z[is_trailing], 0.0, 1.0)
    push_extension = drive * push_gain * (rear_factor ** 1.2) * down_bias

    targets[is_trailing] = np.clip(targets[is_trailing] + push_extension, min_offset, max_extend)

    # 3. Smooth forward clearance tuck for front rods to avoid stubbing
    is_front = (u_long > 0.30) & (u_z > -0.70)
    targets[is_front] = np.clip(targets[is_front] * 0.75, min_offset, max_extend)

    return targets


class FlatPlaneKinematicEnv(MujocoSteeringEnv):
    def step(self, action):
        act = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        cmd_gf = act[:2]
        n = float(np.linalg.norm(cmd_gf))
        d_gf = cmd_gf / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
        drive = 1.0

        g = self._goal_dir(self._info["ball_xy"])
        raw_d_world = d_gf[0] * g + d_gf[1] * np.array([-g[1], g[0]], dtype=np.float32)

        if self._smoothed_cmd_world is None:
            self._smoothed_cmd_world = raw_d_world.copy()
        else:
            self._smoothed_cmd_world = 0.85 * self._smoothed_cmd_world + 0.15 * raw_d_world

        s_norm = float(np.linalg.norm(self._smoothed_cmd_world))
        d_world = self._smoothed_cmd_world / s_norm if s_norm > 1e-6 else raw_d_world

        total_r = 0.0
        term = False
        info = self._info

        prev_d = getattr(self, "_last_d_exec", d_world)
        self._last_d_exec = d_world.copy()

        for sub_i in range(self.k):
            alpha = (sub_i + 1.0) / float(self.k)
            d_sub = (1.0 - alpha) * prev_d + alpha * d_world
            d_sub /= max(np.linalg.norm(d_sub), 1e-6)

            targets = flat_plane_kinematic_bar_targets(
                quat=info["quat"],
                dirs_body=self.env.dirs_body,
                max_extend=self.env.max_extend,
                d_hat=d_sub,
                drive=drive,
                min_offset=0.025,
            )

            _sub_obs, sub_r, sub_term, _sub_trunc, info = self.env.step(targets)
            total_r += float(sub_r)
            if sub_term:
                term = True
                break

        self._info = info
        return self._observe(info), total_r, term, False, info


def main():
    vec_env = DummyVecEnv([lambda: FlatPlaneKinematicEnv(cfg, scenario=sc, randomize=False, max_steps=1500)])
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env, device="cpu")

    obs = vec_env.reset()
    raw_env = vec_env.envs[0]

    frames_dual = []
    frames_chase = []

    done = False
    step = 0
    wall_contacts = 0
    velocities = []
    z_heights = []
    angular_jerks = []
    prev_ang_vel = None

    print("Running Flat-Plane Kinematic Stance Simulation...", flush=True)

    while not done and step < 600:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)

        raw = raw_env.env if hasattr(raw_env, "env") else raw_env
        z_pos = float(raw.data.qpos[2])
        z_heights.append(z_pos)

        speed = float(np.linalg.norm(raw.data.qvel[:2]))
        velocities.append(speed)

        curr_ang_vel = raw.data.qvel[3:6].copy()
        if prev_ang_vel is not None:
            angular_jerks.append(float(np.linalg.norm(curr_ang_vel - prev_ang_vel)))
        prev_ang_vel = curr_ang_vel

        if infos[0].get("wall_contact", False):
            wall_contacts += 1

        frames_dual.append(raw_env.render(mode="dual"))
        frames_chase.append(raw_env.render(mode="chase"))

        done = dones[0]
        step += 1

    info = infos[0]
    final_dist = float(np.linalg.norm(raw.data.qpos[:2] - sc.goal[:2]))
    success = bool(final_dist < 0.50 or info.get("success", False))
    avg_speed = float(np.mean(velocities)) if velocities else 0.0
    z_bounce_std = float(np.std(z_heights)) * 1000.0
    z_ptp = float(np.ptp(z_heights)) * 1000.0
    mean_jerk = float(np.mean(angular_jerks)) if angular_jerks else 0.0

    print("\n=========================================================================")
    print("FLAT-PLANE KINEMATIC STANCE RESULTS")
    print("=========================================================================")
    print(f"Success: {success} | Steps: {step} | Wall Contacts: {wall_contacts} ({wall_contacts/step*100:.1f}%)")
    print(f"Z-Bounce Std: {z_bounce_std:.2f} mm | Peak-to-Peak Bounce: {z_ptp:.2f} mm")
    print(f"Angular Jerk: {mean_jerk:.4f} | Avg Speed: {avg_speed:.2f} m/s")

    out_dual = renders_dir / "flat_plane_kinematics_dual.mp4"
    out_chase = renders_dir / "flat_plane_kinematics_chase.mp4"
    imageio.mimsave(str(out_dual), frames_dual, fps=30)
    imageio.mimsave(str(out_chase), frames_chase, fps=30)

    thumb_dual = scratch_dir / "flat_plane_dual_thumb.png"
    thumb_chase = scratch_dir / "flat_plane_chase_thumb.png"
    imageio.imwrite(str(thumb_dual), frames_dual[len(frames_dual)//2])
    imageio.imwrite(str(thumb_chase), frames_chase[len(frames_chase)//2])

    print(f"\nSaved videos to:\n  - {out_dual}\n  - {out_chase}")
    vec_env.close()


if __name__ == "__main__":
    main()
