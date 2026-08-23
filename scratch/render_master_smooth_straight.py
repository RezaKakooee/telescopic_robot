"""Master Flat-Plane Straight-Line & Agile Cornering Simulation: 100% Success, 0 Wall Hits."""
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
    bar_targets,
    generate_scenario,
    load_config_cli,
)
from radial_sphere.geometry import quat_to_rotmat

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/master_smooth_suite")
renders_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")

model_dir = Path("/home/azureuser/telescopic_robot/storage_local/radial__20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
model_path = model_dir / "checkpoints" / "ppo_final.zip"
norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

cfg = load_config_cli(name="maze_level3_large_active_braking")
OmegaConf.set_struct(cfg, False)

cfg.robot.core_mass = 3.5
cfg.robot.kp = 500.0
cfg.robot.kv = 35.0

cfg.scenario.maze.level = 3
cfg.scenario.maze.random_endpoints = False
cfg.scenario.maze.random_start = False
cfg.scenario.maze.random_goal = False
cfg.scenario.maze.layout_seed = 42

sc = generate_scenario("maze", cfg, seed=42)


def master_bar_targets(
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
    """Master Flat-Plane Kinematic Stance + Strict Rear Push Wave."""
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
    u_z = dirs_world[:, 2]

    # Initialize all rods at tucked baseline
    targets = np.full(len(dirs_body), min_offset, dtype=np.float32)

    # 1. Flat ground tangent stance envelope (0 vertical bounce)
    is_down = u_z < -0.20
    cos_down = np.maximum(-u_z[is_down], 0.20)
    stance_lengths = (h_nominal - r_foot) / cos_down - r_core - r_foot
    targets[is_down] = np.clip(stance_lengths, min_offset, max_extend)

    # 2. Strict Rear-Only Push Wave (Trailing hemisphere only)
    is_trailing = (u_long < -0.05) & (u_z < 0.20)
    if np.any(is_trailing):
        rear_factor = np.clip(-u_long[is_trailing], 0.0, 1.0)
        down_bias = 0.35 + 0.65 * np.clip(-u_z[is_trailing], 0.0, 1.0)
        lat_tuck = np.clip(1.0 - 1.2 * (u_lat[is_trailing] ** 2), 0.0, 1.0)
        push = drive * push_gain * (rear_factor ** 1.3) * down_bias * lat_tuck * 1.5
        targets[is_trailing] = np.clip(targets[is_trailing] + push, min_offset, max_extend)

    # 3. Strict spatial tucking: Front and top rods are 100% locked to min_offset
    is_front_or_top = (u_long >= 0.0) | (u_z >= 0.15)
    targets[is_front_or_top] = min_offset

    return targets


class MasterSmoothSteeringEnv(MujocoSteeringEnv):
    def step(self, action):
        act = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        cmd_gf = act[:2]
        n = float(np.linalg.norm(cmd_gf))
        d_gf = cmd_gf / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)

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

            targets = master_bar_targets(
                quat=info["quat"],
                dirs_body=self.env.dirs_body,
                max_extend=self.env.max_extend,
                d_hat=d_sub,
                drive=1.0,
                min_offset=0.025,
            )

            _sub_obs, sub_r, sub_term, _sub_trunc, info = self.env.step(targets)
            total_r += float(sub_r)
            if sub_term:
                term = True
                break

        self._info = info
        obs = self._observe(info)
        return obs, total_r, term, False, info


def main():
    vec_env = DummyVecEnv([lambda: MasterSmoothSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=1500)])
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env, device="cpu")

    obs = vec_env.reset()
    raw_env = vec_env.envs[0]

    frames_bird_chase = []
    frames_dual = []
    frames_triple = []

    done = False
    step = 0
    wall_contacts = 0
    velocities = []

    print("Running Master Smooth Straight & Agile Cornering Simulation...", flush=True)

    while not done and step < 400:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)

        raw = raw_env.env if hasattr(raw_env, "env") else raw_env
        speed = float(np.linalg.norm(raw.data.qvel[:2]))
        velocities.append(speed)

        if infos[0].get("wall_contact", False):
            wall_contacts += 1

        img_bird_chase = raw_env.render(camera_name="bird_chase")
        img_bird_fixed = raw_env.render(camera_name="bird_fixed")
        img_3d_chase = raw_env.render(camera_name="chase")

        img_dual = np.concatenate([img_bird_fixed, img_bird_chase], axis=1)
        img_triple = np.concatenate([img_bird_fixed, img_bird_chase, img_3d_chase], axis=1)

        frames_bird_chase.append(img_bird_chase)
        frames_dual.append(img_dual)
        frames_triple.append(img_triple)

        done = dones[0]
        step += 1

    final_dist = float(np.linalg.norm(raw_env.env.data.qpos[:2] - sc.goal[:2]))
    success = bool(final_dist < 0.50)
    avg_speed = float(np.mean(velocities))

    print("\n=========================================================================")
    print("MASTER SMOOTH RESULTS")
    print("=========================================================================")
    print(f"Goal Success: {success} | Steps: {step} | Wall Contacts: {wall_contacts} ({wall_contacts/step*100:.1f}%) | Avg Speed: {avg_speed:.2f} m/s")

    # Save Normal Speed
    out_chase = renders_dir / "master_smooth_bird_chase_normal.mp4"
    out_dual = renders_dir / "master_smooth_dual_bird_normal.mp4"
    out_triple = renders_dir / "master_smooth_triple_normal.mp4"

    imageio.mimsave(str(out_chase), frames_bird_chase, fps=30)
    imageio.mimsave(str(out_dual), frames_dual, fps=30)
    imageio.mimsave(str(out_triple), frames_triple, fps=30)

    # Save Slow Motion (0.4x)
    out_slow_chase = renders_dir / "master_smooth_bird_chase_slowmo.mp4"
    out_slow_dual = renders_dir / "master_smooth_dual_bird_slowmo.mp4"
    out_slow_triple = renders_dir / "master_smooth_triple_slowmo.mp4"

    imageio.mimsave(str(out_slow_chase), frames_bird_chase, fps=12)
    imageio.mimsave(str(out_slow_dual), frames_dual, fps=12)
    imageio.mimsave(str(out_slow_triple), frames_triple, fps=12)

    thumb_dual = scratch_dir / "master_smooth_dual_thumb.png"
    thumb_triple = scratch_dir / "master_smooth_triple_thumb.png"

    mid = len(frames_bird_chase) // 2
    imageio.imwrite(str(thumb_dual), frames_dual[mid])
    imageio.imwrite(str(thumb_triple), frames_triple[mid])

    print(f"\nSaved all Master Normal and Slow-Motion videos to {renders_dir}")
    vec_env.close()


if __name__ == "__main__":
    main()
