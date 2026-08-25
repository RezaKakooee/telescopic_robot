"""Render 4 fixed outside edge cameras at 30 degrees and 2x2 Fixed Quad Multi-View."""
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

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260823_2258__fixed_outside_cameras_suite")
renders_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")

model_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
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
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    u_z = dirs_world[:, 2]

    targets = np.full(len(dirs_body), min_offset, dtype=np.float32)

    is_down = u_z < -0.20
    cos_down = np.maximum(-u_z[is_down], 0.20)
    stance_lengths = (h_nominal - r_foot) / cos_down - r_core - r_foot
    targets[is_down] = np.clip(stance_lengths, min_offset, max_extend)

    is_trailing = (u_long < 0.0) & (u_z < 0.30)
    rear_factor = np.clip(-u_long[is_trailing], 0.0, 1.0)
    down_bias = 0.35 + 0.65 * np.clip(-u_z[is_trailing], 0.0, 1.0)
    push_extension = drive * push_gain * (rear_factor ** 1.2) * down_bias
    targets[is_trailing] = np.clip(targets[is_trailing] + push_extension, min_offset, max_extend)

    is_front = (u_long > 0.30) & (u_z > -0.70)
    targets[is_front] = np.clip(targets[is_front] * 0.75, min_offset, max_extend)

    return targets


class FlatPlaneKinematicEnv(MujocoSteeringEnv):
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

            targets = flat_plane_kinematic_bar_targets(
                quat=info["quat"],
                dirs_body=self.env.dirs_body,
                max_extend=self.env.max_extend,
                d_hat=d_sub,
                drive=1.0,
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

    frames_quad = []
    frames_north = []
    frames_south = []
    frames_east = []
    frames_west = []

    done = False
    step = 0

    print("Rendering 4 Fixed Outside Edge Cameras (30 deg) & 2x2 Fixed Quad Video...", flush=True)

    while not done and step < 400:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)

        # 4 Fixed outside edge cameras looking inward at 30 degrees
        img_n = raw_env.render(camera_name="fixed_edge_north_30deg")
        img_s = raw_env.render(camera_name="fixed_edge_south_30deg")
        img_e = raw_env.render(camera_name="fixed_edge_east_30deg")
        img_w = raw_env.render(camera_name="fixed_edge_west_30deg")

        # 2x2 Fixed Quad Grid
        top_row = np.concatenate([img_w, img_n], axis=1)
        bot_row = np.concatenate([img_s, img_e], axis=1)
        img_quad = np.concatenate([top_row, bot_row], axis=0)

        frames_quad.append(img_quad)
        frames_north.append(img_n)
        frames_south.append(img_s)
        frames_east.append(img_e)
        frames_west.append(img_w)

        done = dones[0]
        step += 1

    out_quad = renders_dir / "fixed_quad_outside_30deg.mp4"
    out_n = renders_dir / "fixed_edge_north_30deg.mp4"
    out_s = renders_dir / "fixed_edge_south_30deg.mp4"
    out_e = renders_dir / "fixed_edge_east_30deg.mp4"
    out_w = renders_dir / "fixed_edge_west_30deg.mp4"

    imageio.mimsave(str(out_quad), frames_quad, fps=30)
    imageio.mimsave(str(out_n), frames_north, fps=30)
    imageio.mimsave(str(out_s), frames_south, fps=30)
    imageio.mimsave(str(out_e), frames_east, fps=30)
    imageio.mimsave(str(out_w), frames_west, fps=30)

    thumb_quad = scratch_dir / "fixed_quad_outside_30deg_thumb.png"
    thumb_n = scratch_dir / "fixed_edge_north_30deg_thumb.png"
    thumb_s = scratch_dir / "fixed_edge_south_30deg_thumb.png"
    thumb_e = scratch_dir / "fixed_edge_east_30deg_thumb.png"
    thumb_w = scratch_dir / "fixed_edge_west_30deg_thumb.png"

    mid = len(frames_quad) // 2
    imageio.imwrite(str(thumb_quad), frames_quad[mid])
    imageio.imwrite(str(thumb_n), frames_north[mid])
    imageio.imwrite(str(thumb_s), frames_south[mid])
    imageio.imwrite(str(thumb_e), frames_east[mid])
    imageio.imwrite(str(thumb_w), frames_west[mid])

    print(f"\nAll 4 Fixed Outside Edge Cameras Rendered successfully!")
    print(f"Saved to {renders_dir}:")
    print(f"  - {out_quad.name}")
    print(f"  - {out_n.name}")
    print(f"  - {out_s.name}")
    print(f"  - {out_e.name}")
    print(f"  - {out_w.name}")
    vec_env.close()


if __name__ == "__main__":
    main()
