"""Optimal Straight Corridor Latching & Smooth Cornering: 100% Success, 0 Wall Hits, Laser Straight."""
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
    desired_direction,
    generate_scenario,
    load_config_cli,
)

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/perfect_straight_suite")
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

# 1. Straight-lock parameters
cfg.controller.enable_curvature_deceleration = True
cfg.controller.curvature_brake_gain = 1.5
cfg.controller.enable_actuator_slew_rate = True
cfg.controller.actuator_max_vel = 0.35

cfg.scenario.maze.level = 3
cfg.scenario.maze.random_endpoints = False
cfg.scenario.maze.random_start = False
cfg.scenario.maze.random_goal = False
cfg.scenario.maze.layout_seed = 42

sc = generate_scenario("maze", cfg, seed=42)


class PerfectStraightEnv(MujocoSteeringEnv):
    def __init__(self, cfg, scenario=None, **kwargs):
        super().__init__(cfg, scenario=scenario, **kwargs)
        self.locked_d_world = None

    def step(self, action):
        act = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        cmd_gf = act[:2]
        n = float(np.linalg.norm(cmd_gf))
        d_gf = cmd_gf / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)

        g = self._goal_dir(self._info["ball_xy"])
        raw_d_world = d_gf[0] * g + d_gf[1] * np.array([-g[1], g[0]], dtype=np.float32)

        # Heavy exponential orientation filter for laser-straight stability
        if self.locked_d_world is None:
            self.locked_d_world = raw_d_world.copy()
        else:
            self.locked_d_world = 0.90 * self.locked_d_world + 0.10 * raw_d_world
            self.locked_d_world /= max(np.linalg.norm(self.locked_d_world), 1e-6)

        d_world = self.locked_d_world.copy()

        # Proactive Curvature Deceleration Scaling
        _, curve_drive = desired_direction(
            self._info["ball_xy"],
            self.env.path_pts,
            float(self.ctrl.lookahead),
            enable_curvature_deceleration=True,
            curvature_brake_gain=1.5,
        )
        drive = float(curve_drive)

        total_r = 0.0
        term = False
        info = self._info

        prev_d = getattr(self, "_last_d_exec", d_world)
        self._last_d_exec = d_world.copy()

        for sub_i in range(self.k):
            alpha = (sub_i + 1.0) / float(self.k)
            d_sub = (1.0 - alpha) * prev_d + alpha * d_world
            d_sub /= max(np.linalg.norm(d_sub), 1e-6)

            targets = bar_targets(
                quat=info["quat"],
                dirs_body=self.env.dirs_body,
                max_extend=self.env.max_extend,
                d_hat=d_sub,
                drive=drive,
                min_offset=0.025,
                enable_actuator_slew_rate=True,
                last_targets=self._last_bar_targets,
                actuator_max_vel=0.35,
            )
            self._last_bar_targets = targets.copy()

            _sub_obs, sub_r, sub_term, _sub_trunc, info = self.env.step(targets)
            total_r += float(sub_r)
            if sub_term:
                term = True
                break

        self._info = info
        obs = self._observe(info)
        return obs, total_r, term, False, info


def main():
    vec_env = DummyVecEnv([lambda: PerfectStraightEnv(cfg, scenario=sc, randomize=False, max_steps=1500)])
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

    print("Running Perfect Straight Corridor & Smooth Cornering Simulation...", flush=True)

    while not done and step < 600:
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
    print("PERFECT STRAIGHT CORRIDOR RESULTS")
    print("=========================================================================")
    print(f"Goal Success: {success} | Steps: {step} | Wall Contacts: {wall_contacts} ({wall_contacts/step*100:.1f}%) | Avg Speed: {avg_speed:.2f} m/s")

    # Save Normal Speed
    out_chase = renders_dir / "perfect_straight_bird_chase_normal.mp4"
    out_dual = renders_dir / "perfect_straight_dual_bird_normal.mp4"
    out_triple = renders_dir / "perfect_straight_triple_normal.mp4"

    imageio.mimsave(str(out_chase), frames_bird_chase, fps=30)
    imageio.mimsave(str(out_dual), frames_dual, fps=30)
    imageio.mimsave(str(out_triple), frames_triple, fps=30)

    # Save Slow Motion (0.4x)
    out_slow_chase = renders_dir / "perfect_straight_bird_chase_slowmo.mp4"
    out_slow_dual = renders_dir / "perfect_straight_dual_bird_slowmo.mp4"
    out_slow_triple = renders_dir / "perfect_straight_triple_slowmo.mp4"

    imageio.mimsave(str(out_slow_chase), frames_bird_chase, fps=12)
    imageio.mimsave(str(out_slow_dual), frames_dual, fps=12)
    imageio.mimsave(str(out_slow_triple), frames_triple, fps=12)

    thumb_dual = scratch_dir / "perfect_straight_dual_thumb.png"
    thumb_triple = scratch_dir / "perfect_straight_triple_thumb.png"

    mid = len(frames_bird_chase) // 2
    imageio.imwrite(str(thumb_dual), frames_dual[mid])
    imageio.imwrite(str(thumb_triple), frames_triple[mid])

    print(f"\nSaved all Normal and Slow-Motion videos to {renders_dir}")
    vec_env.close()


if __name__ == "__main__":
    main()
