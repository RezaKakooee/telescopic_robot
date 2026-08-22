"""Low-Level 60D End-to-End RL Environment for RadialSphere on Native MuJoCo.

Supports:
1. Pure 60D End-to-End Motor RL
2. Teacher / Cam Wave Imitation Warm-Start (Method 1)
3. Contact-Hemisphere Physical Force Shaping (Method 2)
4. CPG + 60D Residual Motor Control (Method 3)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from radial_sphere.controller import bar_targets, desired_direction
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario


class MujocoLowLevelEnv(gym.Env):
    """Gym wrapper for direct 60-actuator continuous control on native MuJoCo."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        cfg,
        scenario: Scenario | None = None,
        randomize: bool = False,
        max_steps: int = 2000,
    ):
        super().__init__()
        self.cfg = cfg
        self.scenario = scenario
        self._randomize = randomize
        self.max_steps = max_steps

        # Create underlying MuJoCo physics env
        self.env = MujocoRadialSphereEnv(
            cfg,
            scenario=scenario,
            randomize=randomize,
            max_steps=max_steps,
        )

        self.n_bars = self.env.n_bars
        self.max_extend = float(self.env.max_extend)
        self.base_offset = float(getattr(cfg.controller, "base", 0.025))

        # RL hyperparameters
        rl = cfg.rl
        self.smooth_alpha = float(getattr(rl, "smooth_alpha", 0.30))
        self.action_smoothness_cost = float(getattr(rl, "action_smoothness_cost", 0.01))
        self.energy_cost = float(getattr(rl, "energy_cost", 0.005))
        self.progress_weight = float(getattr(rl, "progress_weight", 3.0))
        self.collision_cost = float(getattr(cfg.reward, "collision_cost", 0.05))

        # Specialized methods
        self.imitation_weight = float(getattr(rl, "imitation_weight", 0.0))
        self.imitation_decay_steps = int(getattr(rl, "imitation_decay_steps", 300000))
        self.contact_shaping = bool(getattr(rl, "contact_shaping", False))
        self.rear_contact_reward = float(getattr(rl, "rear_contact_reward", 0.15))
        self.front_contact_penalty = float(getattr(rl, "front_contact_penalty", 0.15))
        self.cpg_residual = bool(getattr(rl, "cpg_residual", False))

        # LiDAR settings
        self.n_lidar = int(getattr(rl, "n_lidar", 24))
        self.lidar_range = float(getattr(rl, "lidar_range", 3.0))

        # Action Space
        if self.cpg_residual:
            # 6 CPG parameters + 60 residual trims
            self.action_space = Box(
                low=-1.0, high=1.0, shape=(6 + self.n_bars,), dtype=np.float32
            )
            act_dim = 6 + self.n_bars
        else:
            self.action_space = Box(
                low=-1.0, high=1.0, shape=(self.n_bars,), dtype=np.float32
            )
            act_dim = self.n_bars

        # Observation Space:
        # - Body orientation quat (4)
        # - Linear velocity in goal frame (3)
        # - Angular velocity (3)
        # - Current 60 joint positions (60)
        # - Previous action targets (act_dim)
        # - Relative goal vector (2) + distance (1)
        # - 24 LiDAR ray ranges (24)
        obs_dim = 4 + 3 + 3 + self.n_bars + act_dim + 3 + self.n_lidar
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._last_action = np.zeros(act_dim, dtype=np.float32)
        self._smoothed_action = np.zeros(act_dim, dtype=np.float32)
        self._info: dict = {}
        self.rl_step_count = 0
        self.total_env_steps = 0

    # ------------------------------------------------------------------
    # Observation Construction
    # ------------------------------------------------------------------
    def _goal_dir(self, ball_xy: np.ndarray) -> np.ndarray:
        g, _ = desired_direction(
            ball_xy, self.env.path_pts, float(self.cfg.controller.lookahead)
        )
        return g

    def _observe(self, info: dict) -> np.ndarray:
        g = self._goal_dir(info["ball_xy"])
        
        # 1. Orientation quaternion (4)
        quat = np.asarray(info["quat"], dtype=np.float32)
        
        # 2. Linear velocity transformed into goal frame (3)
        v = np.asarray(info["lin_vel"], dtype=np.float32)
        v_fwd = v[0] * g[0] + v[1] * g[1]
        v_lat = g[0] * v[1] - g[1] * v[0]
        v_gf = np.array([v_fwd, v_lat, v[2]], dtype=np.float32)
        
        # 3. Angular velocity (3)
        ang_vel = np.asarray(info["ang_vel"], dtype=np.float32)
        
        # 4. Joint positions normalized to [0, 1] (60)
        joint_pos = np.asarray(info["joint_pos"], dtype=np.float32) / self.max_extend
        
        # 5. Last smoothed action
        last_act = self._smoothed_action.copy()
        
        # 6. Relative goal vector and normalized distance (3)
        goal_xy = np.asarray(self.env.scenario.goal[:2], dtype=np.float32)
        ball_xy = np.asarray(info["ball_xy"][:2], dtype=np.float32)
        rel_goal = goal_xy - ball_xy
        norm_dist = np.array([info["distance"] / max(self.env.path_length, 1.0)], dtype=np.float32)
        
        # 7. LiDAR ranges in goal frame (24)
        lidar_ranges = self.env.raycast_lidar(
            n_rays=self.n_lidar,
            max_range=self.lidar_range,
            g=g,
        )

        parts = [quat, v_gf, ang_vel, joint_pos, last_act, rel_goal, norm_dist, lidar_ranges]
        return np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        obs_raw, info = self.env.reset(seed=seed, options=options)
        act_dim = self.action_space.shape[0]
        self._last_action = np.zeros(act_dim, dtype=np.float32)
        self._smoothed_action = np.zeros(act_dim, dtype=np.float32)
        self._info = info
        self.rl_step_count = 0
        return self._observe(info), info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        act = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        
        # Smooth action filter
        self._smoothed_action = (
            1.0 - self.smooth_alpha
        ) * self._smoothed_action + self.smooth_alpha * act

        # Action delta smoothness penalty
        action_delta = float(np.sum((act - self._last_action) ** 2))
        action_penalty = self.action_smoothness_cost * action_delta
        self._last_action = act.copy()

        g = self._goal_dir(self._info["ball_xy"])
        rotmat = quat_to_rotmat(self._info["quat"])

        # Decode physical targets
        if self.cpg_residual:
            cpg_params = self._smoothed_action[:6]
            residuals = self._smoothed_action[6:] * 0.35  # +/- 35% residual trim
            
            steer_xy = cpg_params[1:3]
            spin_yaw = float(cpg_params[3])  # [-1.0, 1.0] lateral yaw spin command
            
            s_norm = float(np.linalg.norm(steer_xy))
            d_hat = steer_xy / s_norm if s_norm > 1e-5 else np.array([g[0], g[1]], dtype=np.float32)
            d_world = d_hat[0] * g + d_hat[1] * np.array([-g[1], g[0]], dtype=np.float32)

            raw_drive = float(cpg_params[0])  # [-1.0, 1.0] full forward/reverse active braking
            if raw_drive >= 0.0:
                drive = raw_drive
                d_eff = d_world
            else:
                drive = -raw_drive
                d_eff = -d_world
            
            # Base longitudinal peristaltic rolling wave (forward or active reverse)
            base_targets = bar_targets(
                quat=self._info["quat"],
                dirs_body=self.env.dirs_body,
                max_extend=self.max_extend,
                d_hat=d_eff,
                drive=drive,
                min_offset=self.base_offset,
                back_gain=float(self.cfg.controller.back_gain),
            )
            
            # Multi-Axis Lateral Spin Wave: tangential differential push on ground-contacting rods
            dirs_world = self.env.dirs_body @ rotmat.T
            spin_wave = np.zeros(self.n_bars, dtype=np.float32)
            if abs(spin_yaw) > 1e-3:
                for k, (ux, uy, uz) in enumerate(dirs_world):
                    if uz < -0.05:  # Lower hemisphere near ground
                        # Tangential ground projection: (ux * -g[1] + uy * g[0])
                        tangent_proj = (ux * -g[1] + uy * g[0])
                        spin_wave[k] = spin_yaw * tangent_proj * (self.max_extend - self.base_offset) * 0.5
            
            targets = np.clip(
                base_targets + spin_wave + residuals * (self.max_extend - self.base_offset),
                self.base_offset,
                self.max_extend,
            )
            energy_penalty = self.energy_cost * float(np.sum(((targets - self.base_offset) / self.max_extend) ** 2))
        else:
            norm_ext = (self._smoothed_action + 1.0) * 0.5
            targets = self.base_offset + norm_ext * (self.max_extend - self.base_offset)
            energy_penalty = self.energy_cost * float(np.sum(norm_ext ** 2))

        # Advance physics simulation
        _sub_obs, sub_r, terminated, _sub_trunc, info = self.env.step(targets)

        # 1. Directional Movement & Cornering Re-direction Rewards
        v = info["lin_vel"][:2]
        v_norm = float(np.linalg.norm(v))
        v_fwd = float(v[0] * g[0] + v[1] * g[1])
        ang_z = float(abs(info["ang_vel"][2]))
        
        # Calculate alignment with goal heading g
        if v_norm > 0.05:
            alignment = v_fwd / v_norm
        else:
            alignment = 1.0

        # Dynamic cornering vs corridor speed mode:
        if alignment > 0.70:  # Well-aligned corridor travel: reward high forward velocity
            progress_reward = self.progress_weight * v_fwd
            alignment_bonus = 3.0 * alignment
            corner_bonus = 0.0
        else:  # Misaligned / approaching 90-deg turn: reward slowing down & fast yaw re-direction
            progress_reward = self.progress_weight * max(0.0, v_fwd)
            alignment_bonus = 0.0
            corner_bonus = 2.0 * ang_z + 1.5 * max(0.0, 1.2 - v_norm)

        # 2. Collision penalty
        collision_penalty = self.collision_cost if info.get("wall_contact", False) else 0.0

        # 3. Goal reached bonus
        goal_bonus = 50.0 if info.get("success", False) else 0.0

        # 4. Method 1: Teacher Imitation Warm-start Reward
        imitation_reward = 0.0
        if self.imitation_weight > 0.0:
            current_w = max(0.0, self.imitation_weight * (1.0 - self.total_env_steps / self.imitation_decay_steps))
            if current_w > 1e-4:
                cam_t = bar_targets(
                    quat=self._info["quat"],
                    dirs_body=self.env.dirs_body,
                    max_extend=self.max_extend,
                    d_hat=g,
                    drive=1.0,
                    min_offset=self.base_offset,
                    back_gain=float(self.cfg.controller.back_gain),
                )
                cam_norm = (cam_t - self.base_offset) / (self.max_extend - self.base_offset) * 2.0 - 1.0
                imit_error = float(np.mean((act[:self.n_bars] - cam_norm) ** 2))
                imitation_reward = current_w * np.exp(-3.0 * imit_error)

        # 5. Method 2: Contact-Hemisphere Force Shaping
        contact_shaping_reward = 0.0
        if self.contact_shaping:
            # Check foot ground contacts
            dirs_world = self.env.dirs_body @ rotmat.T
            for k, (ux, uy, uz) in enumerate(dirs_world):
                fid = self.env.model.geom(f"foot_{k}").id if hasattr(self.env.model, "geom") else -1
                if fid >= 0:
                    # dot product with heading g in xy
                    heading_proj = ux * g[0] + uy * g[1]
                    ext_k = targets[k]
                    # Rear quadrant (heading_proj < -0.1 and pointing down uz < 0)
                    if heading_proj < -0.15 and uz < -0.1:
                        if ext_k > self.base_offset + 0.04:
                            contact_shaping_reward += self.rear_contact_reward * (ext_k / self.max_extend)
                    # Front quadrant (heading_proj > 0.15 and pointing down uz < 0)
                    elif heading_proj > 0.15 and uz < -0.1:
                        if ext_k > self.base_offset + 0.04:
                            contact_shaping_reward -= self.front_contact_penalty * (ext_k / self.max_extend)

        total_reward = (
            progress_reward
            + alignment_bonus
            + corner_bonus
            - action_penalty
            - energy_penalty
            - collision_penalty
            + goal_bonus
            + imitation_reward
            + contact_shaping_reward
        )

        self.rl_step_count += 1
        self.total_env_steps += 1
        truncated = bool(self.rl_step_count >= self.max_steps)

        self._info = info
        obs = self._observe(info)
        return obs, float(total_reward), terminated, truncated, self._info

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self, mode: str = "chase") -> np.ndarray:
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
