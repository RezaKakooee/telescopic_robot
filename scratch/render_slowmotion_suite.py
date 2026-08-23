"""Render High-Detail Slow-Motion Video Suite (0.33x / 3x Slow Motion) capturing every sub-step."""
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

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/slowmotion_video_suite")
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


class SlowMoSteeringEnv(MujocoSteeringEnv):
    def step(self, action):
        act = np.array(action, copy=True).reshape(-1)
        if len(act) > 2:
            act[2] = 1.0  # Full drive
        return super().step(act)


def main():
    vec_env = DummyVecEnv([lambda: SlowMoSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=1500)])
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
    frames_3d = []

    done = False
    step = 0

    print("Rendering High-Resolution Slow-Motion Video Suite...", flush=True)

    while not done and step < 350:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)

        img_bird_chase = raw_env.render(camera_name="bird_chase")
        img_bird_fixed = raw_env.render(camera_name="bird_fixed")
        img_3d_chase = raw_env.render(camera_name="chase")

        img_dual = np.concatenate([img_bird_fixed, img_bird_chase], axis=1)
        img_triple = np.concatenate([img_bird_fixed, img_bird_chase, img_3d_chase], axis=1)

        frames_bird_chase.append(img_bird_chase)
        frames_dual.append(img_dual)
        frames_triple.append(img_triple)
        frames_3d.append(img_3d_chase)

        done = dones[0]
        step += 1

    # 1. Normal Real-Time Speed Videos (30 FPS)
    out_normal_bird = renders_dir / "bird_chase_normal_speed.mp4"
    out_normal_dual = renders_dir / "dual_bird_normal_speed.mp4"
    out_normal_triple = renders_dir / "triple_view_normal_speed.mp4"
    out_normal_3d = renders_dir / "3d_chase_normal_speed.mp4"

    imageio.mimsave(str(out_normal_bird), frames_bird_chase, fps=30)
    imageio.mimsave(str(out_normal_dual), frames_dual, fps=30)
    imageio.mimsave(str(out_normal_triple), frames_triple, fps=30)
    imageio.mimsave(str(out_normal_3d), frames_3d, fps=30)

    # 2. Cinematic Slow-Motion Videos (12 FPS = 0.40x / 2.5x Slow-Mo)
    out_slow_bird = renders_dir / "bird_chase_slow_motion_0_4x.mp4"
    out_slow_dual = renders_dir / "dual_bird_slow_motion_0_4x.mp4"
    out_slow_triple = renders_dir / "triple_view_slow_motion_0_4x.mp4"
    out_slow_3d = renders_dir / "3d_chase_slow_motion_0_4x.mp4"

    imageio.mimsave(str(out_slow_bird), frames_bird_chase, fps=12)
    imageio.mimsave(str(out_slow_dual), frames_dual, fps=12)
    imageio.mimsave(str(out_slow_triple), frames_triple, fps=12)
    imageio.mimsave(str(out_slow_3d), frames_3d, fps=12)

    thumb = scratch_dir / "slowmo_triple_thumb.png"
    imageio.imwrite(str(thumb), frames_triple[len(frames_triple)//2])

    print(f"\nBoth Normal and Slow-Motion Videos Saved Successfully!")
    print(f"Directory: {renders_dir}")
    print(f"  - {out_normal_bird.name} & {out_slow_bird.name}")
    print(f"  - {out_normal_dual.name} & {out_slow_dual.name}")
    print(f"  - {out_normal_triple.name} & {out_slow_triple.name}")
    print(f"  - {out_normal_3d.name} & {out_slow_3d.name}")
    vec_env.close()


if __name__ == "__main__":
    main()
