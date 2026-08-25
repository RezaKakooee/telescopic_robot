"""High-level RL Steering Environment for Native MuJoCo.

The RL policy emits a 2D desired heading [cmd_x, cmd_y] (in the look-ahead goal frame),
and the open-loop bar controller computes the 60 radial telescoping extensions at
every physics step.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from ._gym import gym, spaces
from .controller import bar_targets, desired_direction
from .mujoco_env import MujocoRadialSphereEnv

class SensorNoiseModel:
    def __init__(self, accel_noise_std: float = 0.05, gyro_noise_std: float = 0.015, lidar_noise_std: float = 0.015, lidar_dropout_prob: float = 0.02, seed: int = 42):
        self.accel_noise_std, self.gyro_noise_std, self.lidar_noise_std, self.lidar_dropout_prob = accel_noise_std, gyro_noise_std, lidar_noise_std, lidar_dropout_prob
        self.rng = np.random.RandomState(seed)

    def corrupt_observation(self, obs: np.ndarray) -> np.ndarray:
        noisy_obs = obs.copy()
        noise = self.rng.normal(0.0, self.lidar_noise_std, size=obs.shape)
        return noisy_obs + noise

class RealisticActuatorModel:
    def __init__(self, n_bars: int = 60, v_max: float = 0.28, a_max: float = 3.5, f_max: float = 50.0, p_battery_max: float = 200.0, dt: float = 0.005):
        self.n_bars, self.v_max, self.a_max, self.f_max, self.p_battery_max, self.dt = n_bars, v_max, a_max, f_max, p_battery_max, dt
        self.current_pos = np.full(n_bars, 0.025, dtype=np.float32)
        self.current_vel = np.zeros(n_bars, dtype=np.float32)

    def apply_dynamics(self, target_pos: np.ndarray, actual_forces: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        desired_vel = (target_pos - self.current_pos) / self.dt
        effective_v_max = self.v_max
        if actual_forces is not None:
            load_factor = np.clip(np.abs(actual_forces) / self.f_max, 0.0, 1.0)
            effective_v_max = self.v_max * (1.0 - 0.5 * load_factor)

        clipped_accel = np.clip((desired_vel - self.current_vel) / self.dt, -self.a_max, self.a_max)
        achievable_vel = np.clip(self.current_vel + clipped_accel * self.dt, -effective_v_max, effective_v_max)

        if actual_forces is not None:
            mech_power = np.abs(actual_forces * achievable_vel)
            total_power = float(np.sum(mech_power))
            if total_power > self.p_battery_max:
                achievable_vel *= self.p_battery_max / total_power
        else:
            total_power = 0.0

        new_pos = np.clip(self.current_pos + achievable_vel * self.dt, 0.025, 0.160)
        self.current_vel = (new_pos - self.current_pos) / self.dt
        self.current_pos = new_pos.copy()
        return new_pos, {"total_power_w": total_power, "max_actuator_vel": float(np.max(np.abs(self.current_vel)))}


N_BASE_OBS = 7
N_ENDPOINT_OBS = 8
FAR_PILLAR = (20.0, 0.0, 20.0)


class MujocoSteeringEnv(gym.Env):
    """High-level 2D steering RL interface on native MuJoCo physics."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: DictConfig | None = None,
        decision_every: int | None = None,
        include_drive: bool | None = None,
        max_steps: int | None = None,
        env: MujocoRadialSphereEnv | None = None,
        **env_kwargs,
    ):
        super().__init__()
        # If max_steps is passed for steering, avoid early low-level truncation
        low_level_kwargs = dict(env_kwargs)
        if "max_steps" in low_level_kwargs:
            del low_level_kwargs["max_steps"]
        self.env = env if env is not None else MujocoRadialSphereEnv(config, max_steps=1000000, **low_level_kwargs)
        cfg = self.env.cfg
        rl = getattr(cfg, "rl", None)

        self.max_steps = int(max_steps if max_steps is not None else getattr(rl, "max_steps", cfg.env.max_steps))
        self.k = int(decision_every if decision_every is not None
                     else getattr(rl, "decision_every", 10))
        self.include_drive = bool(include_drive if include_drive is not None
                                  else getattr(rl, "include_drive", False))
        self.ctrl = cfg.controller

        self.k_obstacles = int(getattr(rl, "obs_obstacles", 3))
        self.n_lidar = int(getattr(rl, "obs_lidar", 16))
        self.lidar_range = float(getattr(rl, "lidar_range", 3.0))
        self.obs_endpoints = bool(getattr(rl, "obs_endpoints", True))
        self.smooth_alpha = float(getattr(rl, "smooth_alpha", 0.35))
        self.action_smoothness_cost = float(getattr(rl, "action_smoothness_cost", 0.02))
        self.lidar_clearance_cost = float(getattr(rl, "lidar_clearance_cost", 0.0))
        self.lidar_min_margin = float(getattr(rl, "lidar_min_margin", 0.35))

        n_act = 3 if self.include_drive else 2
        obs_dim = (N_BASE_OBS + 3 * self.k_obstacles + self.n_lidar +
                   (N_ENDPOINT_OBS if self.obs_endpoints else 0))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._last_cmd = np.array([1.0, 0.0], dtype=np.float32)
        self._last_raw_cmd = np.array([1.0, 0.0], dtype=np.float32)
        self._smoothed_cmd_world = None
        self._last_drive = 1.0
        self._info = None

        # --- Modular Sim2Real Dynamic Engine Setup ---
        s2r = getattr(cfg, "sim2real", None)
        s2r_dict = dict(s2r) if s2r is not None else {}
        self.enable_sim2real = bool(s2r_dict.get("enabled", False))
        self.enable_actuator_limits = bool(s2r_dict.get("enable_actuator_limits", False))
        self.enable_sensor_noise = bool(s2r_dict.get("enable_sensor_noise", False))
        self.enable_latency = bool(s2r_dict.get("enable_latency", False))
        
        self.sensor_noise = None
        if self.enable_sim2real and self.enable_sensor_noise:
            self.sensor_noise = SensorNoiseModel(
                accel_noise_std=float(s2r_dict.get("imu_accel_noise_std", 0.05)),
                gyro_noise_std=float(s2r_dict.get("imu_gyro_noise_std", 0.015)),
                lidar_noise_std=float(s2r_dict.get("lidar_noise_std", 0.015)),
                lidar_dropout_prob=float(s2r_dict.get("lidar_dropout_prob", 0.02)),
                seed=42
            )

        self.actuator_sim = None
        if self.enable_sim2real and self.enable_actuator_limits:
            self.actuator_sim = RealisticActuatorModel(
                n_bars=len(self.env.dirs_body),
                v_max=float(s2r_dict.get("actuator_v_max", 0.28)),
                a_max=float(s2r_dict.get("actuator_a_max", 3.5)),
                f_max=float(s2r_dict.get("actuator_force_limit", 50.0)),
                p_battery_max=float(s2r_dict.get("battery_p_max", 200.0)),
                dt=float(self.env.model.opt.timestep)
            )

        self.delay_steps = 0
        self.action_queue = None
        if self.enable_sim2real and self.enable_latency:
            latency_ms = float(s2r_dict.get("action_delay_ms", 25.0))
            self.delay_steps = max(1, int(np.round(latency_ms / (float(self.env.model.opt.timestep) * 1000.0))))
            self.action_queue = deque(maxlen=self.delay_steps + 1)
        
        self.power_history = []
        self.max_vel_history = []
        self._last_bar_targets = None

    # ------------------------------------------------------------------
    # Goal Frame Heading & Observation
    # ------------------------------------------------------------------
    def _goal_dir(self, ball_xy: np.ndarray) -> np.ndarray:
        """Unit look-ahead direction toward the goal/path (world frame)."""
        enable_spline_heading = bool(getattr(self.ctrl, "enable_spline_heading", False))
        spline_smoothing_weight = float(getattr(self.ctrl, "spline_smoothing_weight", 0.8))
        enable_curvature_deceleration = bool(getattr(self.ctrl, "enable_curvature_deceleration", False))
        curvature_brake_gain = float(getattr(self.ctrl, "curvature_brake_gain", 1.8))

        g, _drive = desired_direction(
            ball_xy,
            self.env.path_pts,
            float(self.ctrl.lookahead),
            enable_spline_heading=enable_spline_heading,
            spline_smoothing_weight=spline_smoothing_weight,
            enable_curvature_deceleration=enable_curvature_deceleration,
            curvature_brake_gain=curvature_brake_gain,
        )
        return g

    def _observe(self, info: dict) -> np.ndarray:
        g = self._goal_dir(info["ball_xy"])
        v = info["lin_vel"][:2]
        v_gf = np.array([v[0] * g[0] + v[1] * g[1],        # forward
                         g[0] * v[1] - g[1] * v[0]])       # lateral (left +)
        dist = info["distance"] / self.env.path_length
        parts = [v_gf, [info["ang_vel"][2]], [dist], self._last_cmd,
                 [self._last_drive]]

        if self.obs_endpoints:
            ball_xy = np.asarray(info["ball_xy"][:2], dtype=np.float32)
            goal_xy = np.asarray(self.env.scenario.goal[:2], dtype=np.float32)
            start_xy = np.asarray(self.env.scenario.spawn_xy[:2], dtype=np.float32)
            rel_goal = goal_xy - ball_xy
            parts.extend([ball_xy, goal_xy, start_xy, rel_goal])

        if self.k_obstacles > 0:
            parts.append(self._obstacle_obs(info["ball_xy"], g))

        # Raycast LiDAR (in goal frame)
        lidar_ranges = self.env.raycast_lidar(
            n_rays=self.n_lidar,
            max_range=self.lidar_range,
            g=g,
        )
        parts.append(lidar_ranges)

        return np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])

    def _obstacle_obs(self, ball_xy: np.ndarray, g: np.ndarray) -> np.ndarray:
        """K nearest pillars/blockers as (forward, lateral, surface gap) in the goal frame."""
        slots = np.tile(np.asarray(FAR_PILLAR, dtype=np.float32), (self.k_obstacles, 1))
        obstacles = getattr(self.env.scenario, "obstacles", None)
        if obstacles is None or len(obstacles) == 0:
            obstacles = getattr(self.env.scenario, "pillars", None)
        if obstacles is not None and len(obstacles) > 0:
            obs_arr = np.asarray(obstacles, dtype=np.float32)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(1, -1)
            rel = obs_arr[:, :2] - ball_xy[None, :]
            d = np.linalg.norm(rel, axis=1)
            order = np.argsort(d)[: self.k_obstacles]
            for s, j in enumerate(order):
                fwd = float(rel[j, 0] * g[0] + rel[j, 1] * g[1])
                lat = float(g[0] * rel[j, 1] - g[1] * rel[j, 0])
                rad = float(obs_arr[j, 2]) if obs_arr.shape[1] > 2 else 0.25
                slots[s] = (fwd, lat, d[j] - rad)
        return slots.reshape(-1)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_cmd = np.array([1.0, 0.0], dtype=np.float32)
        self._last_raw_cmd = np.array([1.0, 0.0], dtype=np.float32)
        self._smoothed_cmd_world = None
        self._last_drive = 1.0
        self._info = info
        self.rl_step_count = 0
        full_obs = self._observe(info)

        if self.enable_sim2real and self.enable_sensor_noise and self.sensor_noise is not None:
            self.sensor_noise.rng = np.random.RandomState(42)
            full_obs = self.sensor_noise.corrupt_observation(full_obs)

        if self.enable_sim2real and self.enable_actuator_limits and self.actuator_sim is not None:
            self.actuator_sim.current_pos = np.full(self.actuator_sim.n_bars, 0.025, dtype=np.float32)
            self.actuator_sim.current_vel = np.zeros(self.actuator_sim.n_bars, dtype=np.float32)
            self.power_history = []
            self.max_vel_history = []

        if self.enable_sim2real and self.enable_latency and self.action_queue is not None:
            self.action_queue.clear()
            for _ in range(self.delay_steps):
                self.action_queue.append(np.array([1.0, 0.0], dtype=np.float32))

        self._last_bar_targets = None
        return full_obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        act = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        cmd_gf = act[:2]
        n = float(np.linalg.norm(cmd_gf))
        d_gf = cmd_gf / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
        drive = float((act[2] + 1.0) * 0.5) if (self.include_drive and len(act) > 2) else 1.0

        # Action rate penalty for smoothness
        action_delta = float(np.sum((cmd_gf - self._last_raw_cmd) ** 2))
        action_penalty = self.action_smoothness_cost * action_delta
        self._last_raw_cmd = cmd_gf.copy()
        self._last_cmd = d_gf
        self._last_drive = drive

        # 1b. Apply Sim2Real Latency / Delay buffer
        if self.enable_sim2real and self.enable_latency and self.action_queue is not None:
            self.action_queue.append(cmd_gf.copy())
            delayed_cmd_gf = self.action_queue.popleft()
            cmd_gf = delayed_cmd_gf
            n = float(np.linalg.norm(cmd_gf))
            d_gf = cmd_gf / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)

        # Transform goal frame heading into world frame
        g = self._goal_dir(self._info["ball_xy"])
        raw_d_world = d_gf[0] * g + d_gf[1] * np.array([-g[1], g[0]], dtype=np.float32)

        if self._smoothed_cmd_world is None:
            self._smoothed_cmd_world = raw_d_world.copy()
        else:
            self._smoothed_cmd_world = (1.0 - self.smooth_alpha) * self._smoothed_cmd_world + self.smooth_alpha * raw_d_world

        s_norm = float(np.linalg.norm(self._smoothed_cmd_world))
        d_world = self._smoothed_cmd_world / s_norm if s_norm > 1e-6 else raw_d_world

        # Optional Curvature Deceleration Scaling
        enable_curvature_deceleration = bool(getattr(self.ctrl, "enable_curvature_deceleration", False))
        if enable_curvature_deceleration:
            _, curve_drive = desired_direction(
                self._info["ball_xy"],
                self.env.path_pts,
                float(self.ctrl.lookahead),
                enable_curvature_deceleration=True,
                curvature_brake_gain=float(getattr(self.ctrl, "curvature_brake_gain", 1.8)),
            )
            drive *= float(curve_drive)

        total_r = -action_penalty
        term, trunc = False, False
        info = self._info

        # Optional low-level enhancements
        enable_power_wave = bool(getattr(self.ctrl, "enable_power_wave", False))
        wave_power_exponent = float(getattr(self.ctrl, "wave_power_exponent", 1.4))
        enable_flank_retraction = bool(getattr(self.ctrl, "enable_flank_retraction", False))
        flank_retract_dist = float(getattr(self.ctrl, "flank_retract_dist", 0.45))
        flank_min_offset = float(getattr(self.ctrl, "flank_min_offset", 0.005))
        enable_camber_banking = bool(getattr(self.ctrl, "enable_camber_banking", False))
        camber_bank_gain = float(getattr(self.ctrl, "camber_bank_gain", 0.035))
        enable_contact_compliance = bool(getattr(self.ctrl, "enable_contact_compliance", False))
        compliance_gain = float(getattr(self.ctrl, "compliance_gain", 0.0005))
        max_contact_force = float(getattr(self.ctrl, "max_contact_force", 40.0))
        enable_anti_stall_reflex = bool(getattr(self.ctrl, "enable_anti_stall_reflex", False))
        anti_stall_speed_threshold = float(getattr(self.ctrl, "anti_stall_speed_threshold", 0.15))
        anti_stall_pulse_freq = float(getattr(self.ctrl, "anti_stall_pulse_freq", 10.0))
        anti_stall_pulse_amp = float(getattr(self.ctrl, "anti_stall_pulse_amp", 0.02))

        # Optional smooth maneuver enhancements
        enable_gaussian_stance = bool(getattr(self.ctrl, "enable_gaussian_stance", False))
        gaussian_stance_sigma = float(getattr(self.ctrl, "gaussian_stance_sigma", 0.38))
        enable_gyroscopic_damping = bool(getattr(self.ctrl, "enable_gyroscopic_damping", False))
        gyroscopic_damping_gain = float(getattr(self.ctrl, "gyroscopic_damping_gain", 0.025))
        enable_actuator_slew_rate = bool(getattr(self.ctrl, "enable_actuator_slew_rate", False))
        actuator_max_vel = float(getattr(self.ctrl, "actuator_max_vel", 0.35))

        lidar_ranges = None
        if enable_flank_retraction:
            lidar_ranges = self.env.raycast_lidar(n_rays=self.n_lidar, max_range=self.lidar_range, g=g)

        prev_d_world = self._last_executed_d_world if hasattr(self, "_last_executed_d_world") and self._last_executed_d_world is not None else d_world
        self._last_executed_d_world = d_world.copy()

        for sub_i in range(self.k):
            # Smooth continuous sub-step heading interpolation
            alpha_sub = (sub_i + 1.0) / float(self.k)
            curr_d_world = (1.0 - alpha_sub) * prev_d_world + alpha_sub * d_world
            norm_d = float(np.linalg.norm(curr_d_world))
            d_sub = curr_d_world / norm_d if norm_d > 1e-6 else d_world

            contact_forces = None
            if enable_contact_compliance and hasattr(self.env, "get_rod_contact_forces"):
                contact_forces = self.env.get_rod_contact_forces()

            forward_vel = float(info["lin_vel"][0] * d_sub[0] + info["lin_vel"][1] * d_sub[1])
            sim_time = float(self.env.data.time)

            enable_adaptive_grouping = bool(getattr(self.ctrl, "enable_adaptive_grouping", False))
            group_size = int(getattr(self.ctrl, "group_size", 10))
            enable_curb_vaulting = bool(getattr(self.ctrl, "enable_curb_vaulting", False))
            curb_boost_gain = float(getattr(self.ctrl, "curb_boost_gain", 2.4))
            enable_underbelly_contact = bool(getattr(self.ctrl, "enable_underbelly_contact", False))
            underbelly_stance_gain = float(getattr(self.ctrl, "underbelly_stance_gain", 0.55))
            underbelly_threshold_z = float(getattr(self.ctrl, "underbelly_threshold_z", -0.20))
            enable_active_suspension = bool(getattr(self.ctrl, "enable_active_suspension", False))

            ideal_targets = bar_targets(
                quat=info["quat"],
                dirs_body=self.env.dirs_body,
                max_extend=self.env.max_extend,
                d_hat=d_sub,
                drive=drive,
                min_offset=float(getattr(self.ctrl, "base", 0.025)),
                back_gain=float(self.ctrl.back_gain),
                enable_power_wave=enable_power_wave,
                wave_power_exponent=wave_power_exponent,
                enable_flank_retraction=enable_flank_retraction,
                lidar_ranges=lidar_ranges,
                lidar_max_range=self.lidar_range,
                flank_retract_dist=flank_retract_dist,
                flank_min_offset=flank_min_offset,
                enable_camber_banking=enable_camber_banking,
                yaw_rate=float(info["ang_vel"][2]),
                camber_bank_gain=camber_bank_gain,
                enable_contact_compliance=enable_contact_compliance,
                contact_forces=contact_forces,
                compliance_gain=compliance_gain,
                max_contact_force=max_contact_force,
                enable_anti_stall_reflex=enable_anti_stall_reflex,
                forward_vel=forward_vel,
                sim_time=sim_time,
                anti_stall_speed_threshold=anti_stall_speed_threshold,
                anti_stall_pulse_freq=anti_stall_pulse_freq,
                anti_stall_pulse_amp=anti_stall_pulse_amp,
                enable_gaussian_stance=enable_gaussian_stance,
                gaussian_stance_sigma=gaussian_stance_sigma,
                enable_gyroscopic_damping=enable_gyroscopic_damping,
                ang_vel=info.get("ang_vel", None),
                gyroscopic_damping_gain=gyroscopic_damping_gain,
                enable_actuator_slew_rate=enable_actuator_slew_rate,
                last_targets=self._last_bar_targets,
                actuator_max_vel=actuator_max_vel,
                enable_adaptive_grouping=enable_adaptive_grouping,
                group_size=group_size,
                enable_curb_vaulting=enable_curb_vaulting,
                curb_boost_gain=curb_boost_gain,
                enable_underbelly_contact=enable_underbelly_contact,
                underbelly_stance_gain=underbelly_stance_gain,
                underbelly_threshold_z=underbelly_threshold_z,
                enable_active_suspension=enable_active_suspension,
                core_z=float(self.env.data.qpos[2]),
                core_vz=float(self.env.data.qvel[2]),
                target_ride_height=float(getattr(self.ctrl, "target_ride_height", 0.28)),
                suspension_kp=float(getattr(self.ctrl, "suspension_kp", 0.65)),
                suspension_kd=float(getattr(self.ctrl, "suspension_kd", 0.12)),
                suspension_force_compliance=float(getattr(self.ctrl, "suspension_force_compliance", 0.0015)),
                nominal_support_force=float(getattr(self.ctrl, "nominal_support_force", 10.0)),
            )
            self._last_bar_targets = ideal_targets.copy()

            if self.enable_sim2real and self.enable_actuator_limits and self.actuator_sim is not None:
                forces = None
                if hasattr(self.env, "get_rod_contact_forces"):
                    forces = self.env.get_rod_contact_forces()
                phys_targets, act_metrics = self.actuator_sim.apply_dynamics(ideal_targets, actual_forces=forces)
                self.power_history.append(act_metrics["total_power_w"])
                self.max_vel_history.append(act_metrics["max_actuator_vel"])
                _sub_obs, sub_r, sub_term, _sub_trunc, info = self.env.step(phys_targets)
            else:
                _sub_obs, sub_r, sub_term, _sub_trunc, info = self.env.step(ideal_targets)

            total_r += float(sub_r)
            if sub_term:
                term = True
                break

        if self.lidar_clearance_cost > 0.0:
            lidar_dists = self.env.raycast_lidar(self.n_lidar, self.lidar_range, g=g)
            min_d = float(np.min(lidar_dists)) * self.lidar_range
            if min_d < self.lidar_min_margin:
                clearance_pen = self.lidar_clearance_cost * ((self.lidar_min_margin - min_d) / self.lidar_min_margin)
                total_r -= float(clearance_pen)

        self.rl_step_count += 1
        if self.rl_step_count >= self.max_steps:
            trunc = True

        self._info = info
        full_obs = self._observe(info)
        if self.enable_sim2real and self.enable_sensor_noise and self.sensor_noise is not None:
            full_obs = self.sensor_noise.corrupt_observation(full_obs)

        return full_obs, float(total_r), term, trunc, self._info

    def render(self, mode: str = "chase", camera_name: str | None = None) -> np.ndarray:
        return self.env.render(mode=mode, camera_name=camera_name)

    def close(self) -> None:
        self.env.close()
