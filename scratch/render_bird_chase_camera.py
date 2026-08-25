"""Render Chasing Bird's-Eye View using the updated strict rear-only controller."""
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

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260823_2303__bird_chase_suite")
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


class StrictRearEnv(MujocoSteeringEnv):
    def step(self, action):
        act = np.array(action, copy=True).reshape(-1)
        if len(act) > 2:
            act[2] = 1.0  # Full drive
        return super().step(act)


def main():
    vec_env = DummyVecEnv([lambda: StrictRearEnv(cfg, scenario=sc, randomize=False, max_steps=1500)])
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

    print("Rendering Chasing Bird's-Eye View with Strict Rear-Only Masking...", flush=True)

    while not done and step < 400:
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

        done = dones[0]
        step += 1

    out_chase = renders_dir / "bird_chase_overhead_tracking.mp4"
    out_dual = renders_dir / "dual_bird_fixed_and_chase.mp4"
    out_triple = renders_dir / "triple_view_overview_birdchase_3dchase.mp4"

    imageio.mimsave(str(out_chase), frames_bird_chase, fps=30)
    imageio.mimsave(str(out_dual), frames_dual, fps=30)
    imageio.mimsave(str(out_triple), frames_triple, fps=30)

    thumb_chase = scratch_dir / "bird_chase_thumb.png"
    thumb_dual = scratch_dir / "dual_bird_chase_thumb.png"
    thumb_triple = scratch_dir / "triple_view_thumb.png"

    mid = len(frames_bird_chase) // 2
    imageio.imwrite(str(thumb_chase), frames_bird_chase[mid])
    imageio.imwrite(str(thumb_dual), frames_dual[mid])
    imageio.imwrite(str(thumb_triple), frames_triple[mid])

    print(f"\nRendered successfully! Saved to {renders_dir}")
    vec_env.close()


if __name__ == "__main__":
    main()
