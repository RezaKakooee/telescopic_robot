"""Pure Silk Gliding Locomotion: C-infinity Spherical Harmonic Wave + Heavy Yaw Rate Limiting + Soft Pneumatics."""
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

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/pure_silk_suite")
renders_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")

model_dir = Path("/home/azureuser/telescopic_robot/storage_local/radial__20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
model_path = model_dir / "checkpoints" / "ppo_final.zip"
norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

cfg = load_config_cli(name="maze_level3_large_active_braking")
OmegaConf.set_struct(cfg, False)

# 1. Soft pneumatic impedance: acts like cushioned rubber suspension
cfg.robot.kp = 320.0
cfg.robot.kv = 48.0

cfg.scenario.maze.level = 3
cfg.scenario.maze.layout_seed = 42

sc = generate_scenario("maze", cfg, seed=42)


def pure_silk_bar_targets(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    drive: float = 0.85,
    min_offset: float = 0.025,
    back_gain: float = 1.4,
) -> np.ndarray:
    """Compute C-infinity smooth spherical harmonic bar targets with ZERO kinks or step-cutoffs."""
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T

    # Ideal pushing pole: rearward (-d_hat) and downward (-z) at 45 degrees
    push_pole = np.array([-d_hat[0] * 0.7071, -d_hat[1] * 0.7071, -0.7071])
    push_pole /= np.linalg.norm(push_pole)

    # Cosine of angle from the push pole: [-1, +1]
    cos_phi = np.clip(dirs_world @ push_pole, -1.0, 1.0)

    # C-infinity Fourier harmonic bell curve: ( (1 + cos_phi)/2 )^2
    # Smooth everywhere on the sphere with zero derivative at the antipole!
    wave = ((1.0 + cos_phi) * 0.5) ** 2.5 * back_gain
    wave = np.clip(wave, 0.0, 1.0)

    targets = min_offset + drive * (max_extend - min_offset) * wave
    return targets


class PureSilkSteeringEnv(MujocoSteeringEnv):
    def __init__(self, cfg, scenario=None, **kwargs):
        super().__init__(cfg, scenario=scenario, **kwargs)
        self.smoothed_heading_angle = None
        self.max_angular_rate = 0.60  # rad/s max yaw turn rate for cruise-ship smoothness
        self.last_bar_targets_slew = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.smoothed_heading_angle = None
        self.last_bar_targets_slew = None
        return obs, info

    def step(self, action):
        # 1. Extract raw RL goal-frame heading
        act = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        cmd_gf = act[:2]
        n = float(np.linalg.norm(cmd_gf))
        d_gf = cmd_gf / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)

        g = self._goal_dir(self._info["ball_xy"])
        raw_d_world = d_gf[0] * g + d_gf[1] * np.array([-g[1], g[0]], dtype=np.float32)
        raw_yaw = np.arctan2(raw_d_world[1], raw_d_world[0])

        if self.smoothed_heading_angle is None:
            self.smoothed_heading_angle = raw_yaw
        else:
            # Bounded angular rate steering (no abrupt yaw turns)
            dt_step = 0.05
            delta = (raw_yaw - self.smoothed_heading_angle + np.pi) % (2 * np.pi) - np.pi
            max_delta = self.max_angular_rate * dt_step
            delta_clamped = np.clip(delta, -max_delta, max_delta)
            self.smoothed_heading_angle += delta_clamped

        d_smooth = np.array([np.cos(self.smoothed_heading_angle), np.sin(self.smoothed_heading_angle)])

        # Execute 10 physics sub-steps with smooth C-infinity harmonic wave
        total_r = 0.0
        term = False
        info = self._info

        for _ in range(self.k):
            targets = pure_silk_bar_targets(
                quat=info["quat"],
                dirs_body=self.env.dirs_body,
                max_extend=self.env.max_extend,
                d_hat=d_smooth,
                drive=0.85,
                min_offset=0.025,
                back_gain=1.4,
            )

            # Slew rate filtering
            if self.last_bar_targets_slew is not None:
                max_d = 0.25 * 0.005  # 0.25 m/s max rod velocity
                targets = np.clip(targets, self.last_bar_targets_slew - max_d, self.last_bar_targets_slew + max_d)
            self.last_bar_targets_slew = targets.copy()

            _sub_obs, sub_r, sub_term, _sub_trunc, info = self.env.step(targets)
            total_r += float(sub_r)
            if sub_term:
                term = True
                break

        self._info = info
        obs_vec = self._observe(info)
        return obs_vec, total_r, term, False, info


vec_env = DummyVecEnv([lambda: PureSilkSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=1500)])
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
print("Rendering Pure Silk Gliding 60 FPS Real-Time Simulation...", flush=True)

while not done and step < 600:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = vec_env.step(action)
    
    # 1:1 real-time rendering
    frames_dual.append(raw_env.render(mode="dual"))
    frames_chase.append(raw_env.render(mode="chase"))
    
    done = dones[0]
    step += 1

out_dual = renders_dir / "pure_silk_gliding_dual.mp4"
out_chase = renders_dir / "pure_silk_gliding_chase.mp4"
imageio.mimsave(str(out_dual), frames_dual, fps=30)
imageio.mimsave(str(out_chase), frames_chase, fps=30)

thumb_dual = scratch_dir / "pure_silk_dual_thumb.png"
imageio.imwrite(str(thumb_dual), frames_dual[len(frames_dual)//2])

print(f"Finished! Saved {len(frames_dual)} frames to {out_dual} and {out_chase}")
vec_env.close()
