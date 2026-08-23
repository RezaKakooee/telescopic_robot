"""Strict Rear-Only Extension Wave: Front, Top, and Sides are 100% Tucked at min_offset."""
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

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/strict_rear_only_suite")
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


def strict_rear_only_bar_targets(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    drive: float = 1.0,
    min_offset: float = 0.025,
    back_gain: float = 1.6,
) -> np.ndarray:
    """Strict Rear-Only Locomotion Wave.

    Rules:
    1. Front rods (u_long >= 0) -> EXACTLY min_offset (0.025m). ZERO extension.
    2. Top rods (u_z >= 0)       -> EXACTLY min_offset (0.025m). ZERO extension.
    3. Side rods (|u_lat| > 0.7) -> EXACTLY min_offset (0.025m). ZERO extension.
    4. ONLY trailing rear-downward quadrant rods (u_long < 0 and u_z < 0) extend to push.
    """
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    # Longitudinal coordinate along heading (-1 rear, +1 front)
    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    # Lateral coordinate perpendicular to heading
    u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
    # Vertical coordinate (-1 down, +1 up)
    u_z = dirs_world[:, 2]

    # Initialize ALL 60 rods to 100% tucked baseline
    targets = np.full(len(dirs_body), min_offset, dtype=np.float32)

    # Only rods in the rear trailing hemisphere (u_long < 0) and lower hemisphere (u_z < 0.1) can extend
    rear_mask = (u_long < -0.05) & (u_z < 0.15)
    
    if np.any(rear_mask):
        rear_factor = np.clip(-u_long[rear_mask], 0.0, 1.0)
        down_bias = 0.35 + 0.65 * np.clip(-u_z[rear_mask], 0.0, 1.0)
        # Suppress side rods from flaring out laterally
        lat_tuck = np.clip(1.0 - 1.2 * (u_lat[rear_mask] ** 2), 0.0, 1.0)

        wave = np.clip((rear_factor ** 1.3) * down_bias * lat_tuck * back_gain, 0.0, 1.0)
        targets[rear_mask] = min_offset + drive * (max_extend - min_offset) * wave

    return targets


class StrictRearSteeringEnv(MujocoSteeringEnv):
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

            targets = strict_rear_only_bar_targets(
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
        return self._observe(info), total_r, term, False, info


def main():
    vec_env = DummyVecEnv([lambda: StrictRearSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=1500)])
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env, device="cpu")

    obs = vec_env.reset()
    raw_env = vec_env.envs[0]

    frames_bird_chase = []
    frames_dual = []
    frames_chase = []

    done = False
    step = 0
    wall_contacts = 0

    print("Rendering Strict Rear-Only Locomotion (Front & Sides 100% Tucked)...", flush=True)

    while not done and step < 400:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)

        if infos[0].get("wall_contact", False):
            wall_contacts += 1

        img_bird_chase = raw_env.render(camera_name="bird_chase")
        img_bird_fixed = raw_env.render(camera_name="bird_fixed")
        img_chase = raw_env.render(camera_name="chase")

        img_dual = np.concatenate([img_bird_fixed, img_bird_chase], axis=1)

        frames_bird_chase.append(img_bird_chase)
        frames_dual.append(img_dual)
        frames_chase.append(img_chase)

        done = dones[0]
        step += 1

    success = bool(np.linalg.norm(raw_env.env.data.qpos[:2] - sc.goal[:2]) < 0.50)
    print(f"Finished! Success={success} | Steps={step} | Wall Contacts={wall_contacts} ({wall_contacts/step*100:.1f}%)")

    out_chase = renders_dir / "strict_rear_bird_chase.mp4"
    out_dual = renders_dir / "strict_rear_dual_bird_chase.mp4"
    out_3d = renders_dir / "strict_rear_3d_chase.mp4"

    imageio.mimsave(str(out_chase), frames_bird_chase, fps=30)
    imageio.mimsave(str(out_dual), frames_dual, fps=30)
    imageio.mimsave(str(out_3d), frames_chase, fps=30)

    thumb_bird = scratch_dir / "strict_rear_bird_thumb.png"
    thumb_dual = scratch_dir / "strict_rear_dual_thumb.png"
    thumb_3d = scratch_dir / "strict_rear_3d_thumb.png"

    mid = len(frames_bird_chase) // 2
    imageio.imwrite(str(thumb_bird), frames_bird_chase[mid])
    imageio.imwrite(str(thumb_dual), frames_dual[mid])
    imageio.imwrite(str(thumb_3d), frames_chase[mid])

    print(f"\nSaved videos to {renders_dir}:")
    print(f"  - {out_chase.name}")
    print(f"  - {out_dual.name}")
    print(f"  - {out_3d.name}")
    vec_env.close()


if __name__ == "__main__":
    main()
