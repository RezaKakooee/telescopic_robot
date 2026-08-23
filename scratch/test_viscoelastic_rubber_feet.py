"""Viscoelastic Rubber Footpads & Surface Friction Realism Suite.

Sim-to-Real Contact Mechanics:
1. Viscoelastic Compliant Footpads: solref="0.020 1.2", solimp="0.90 0.95 0.005" (modeling Shore 60A rubber deformation).
2. Realistic Surface Friction: friction="0.85 0.015 0.005" (realistic rubber-on-floor traction).
3. Realistic Actuator Engine: 50N force cap, 0.28 m/s speed limit, Back-EMF derating, 200W battery limit.
4. Soft-Cluster Footpad Controller: Smooth C1-continuous cluster handoffs.
"""
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

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/viscoelastic_rubber_suite")
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

# Sim-to-Real Actuator & Contact Parameters
cfg.robot.actuator_force_limit = 50.0   # 50 N max stall force
cfg.robot.actuator_v_max = 0.28         # 0.28 m/s max velocity
cfg.robot.actuator_a_max = 3.5          # 3.5 m/s^2 max acceleration
cfg.robot.battery_p_max = 200.0        # 200 W power limit

cfg.scenario.maze.level = 3
cfg.scenario.maze.random_endpoints = False
cfg.scenario.maze.random_start = False
cfg.scenario.maze.random_goal = False
cfg.scenario.maze.layout_seed = 42

sc = generate_scenario("maze", cfg, seed=42)


class RealisticActuatorModel:
    """Hardware-realistic physical linear actuator & battery model."""
    def __init__(
        self,
        n_bars: int = 60,
        v_max: float = 0.28,
        a_max: float = 3.5,
        f_max: float = 50.0,
        p_battery_max: float = 200.0,
        dt: float = 0.005,
    ):
        self.n_bars = n_bars
        self.v_max = v_max
        self.a_max = a_max
        self.f_max = f_max
        self.p_battery_max = p_battery_max
        self.dt = dt

        self.current_pos = np.full(n_bars, 0.025, dtype=np.float32)
        self.current_vel = np.zeros(n_bars, dtype=np.float32)

    def apply_dynamics(self, target_pos: np.ndarray, actual_forces: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        desired_vel = (target_pos - self.current_pos) / self.dt

        effective_v_max = self.v_max
        if actual_forces is not None:
            load_factor = np.clip(np.abs(actual_forces) / self.f_max, 0.0, 1.0)
            effective_v_max = self.v_max * (1.0 - 0.5 * load_factor)

        accel = (desired_vel - self.current_vel) / self.dt
        clipped_accel = np.clip(accel, -self.a_max, self.a_max)
        achievable_vel = np.clip(self.current_vel + clipped_accel * self.dt, -effective_v_max, effective_v_max)

        if actual_forces is not None:
            mech_power = np.abs(actual_forces * achievable_vel)
            total_power = float(np.sum(mech_power))
            if total_power > self.p_battery_max:
                power_scale = self.p_battery_max / total_power
                achievable_vel *= power_scale
        else:
            total_power = 0.0

        new_pos = np.clip(self.current_pos + achievable_vel * self.dt, 0.025, 0.160)
        self.current_vel = (new_pos - self.current_pos) / self.dt
        self.current_pos = new_pos.copy()

        metrics = {
            "total_power_w": total_power,
            "max_actuator_vel": float(np.max(np.abs(self.current_vel))),
        }
        return new_pos, metrics


class ViscoelasticRubberFeetEnv(MujocoSteeringEnv):
    def __init__(self, cfg, scenario=None, **kwargs):
        super().__init__(cfg, scenario=scenario, **kwargs)
        self.cfg = cfg

        # Apply Viscoelastic Rubber Footpad Parameters to MuJoCo Model Geoms
        # solref: [timeconst, dampratio] = [0.020, 1.20] (Shore 60A polyurethane compliance)
        # solimp: [dmin, dmax, width] = [0.90, 0.95, 0.005]
        # friction: [0.85, 0.015, 0.005] (realistic rubber traction)
        model = self.env.model
        for k in range(len(self.env.dirs_body)):
            geom_id = model.geom(f"foot_{k}").id
            model.geom_solref[geom_id] = np.array([0.020, 1.20], dtype=np.float64)
            model.geom_solimp[geom_id] = np.array([0.90, 0.95, 0.005, 0.5, 2.0], dtype=np.float64)
            model.geom_friction[geom_id] = np.array([0.85, 0.015, 0.005], dtype=np.float64)

        self.actuator_sim = RealisticActuatorModel(
            n_bars=len(self.env.dirs_body),
            v_max=float(getattr(cfg.robot, "actuator_v_max", 0.28)),
            a_max=float(getattr(cfg.robot, "actuator_a_max", 3.5)),
            f_max=float(getattr(cfg.robot, "actuator_force_limit", 50.0)),
            p_battery_max=float(getattr(cfg.robot, "battery_p_max", 200.0)),
            dt=float(self.env.model.opt.timestep),
        )
        self.power_history = []
        self.max_vel_history = []

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.actuator_sim = RealisticActuatorModel(
            n_bars=len(self.env.dirs_body),
            v_max=float(getattr(self.cfg.robot, "actuator_v_max", 0.28)),
            a_max=float(getattr(self.cfg.robot, "actuator_a_max", 3.5)),
            f_max=float(getattr(self.cfg.robot, "actuator_force_limit", 50.0)),
            p_battery_max=float(getattr(self.cfg.robot, "battery_p_max", 200.0)),
            dt=float(self.env.model.opt.timestep),
        )
        self.power_history = []
        self.max_vel_history = []
        return obs, info

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

        dirs_body = self.env.dirs_body
        max_extend = self.env.max_extend

        for sub_i in range(self.k):
            alpha = (sub_i + 1.0) / float(self.k)
            d_sub = (1.0 - alpha) * prev_d + alpha * d_world
            d_sub /= max(np.linalg.norm(d_sub), 1e-6)

            R = quat_to_rotmat(info["quat"])
            dirs_world = dirs_body @ R.T
            u_long = dirs_world[:, 0] * d_sub[0] + dirs_world[:, 1] * d_sub[1]
            u_lat = dirs_world[:, 0] * (-d_sub[1]) + dirs_world[:, 1] * d_sub[0]
            u_z = dirs_world[:, 2]

            ideal_targets = np.full(len(dirs_body), 0.025, dtype=np.float32)

            is_down = u_z < -0.15
            cos_down = np.maximum(-u_z[is_down], 0.15)
            stance_lengths = (0.275 - 0.015) / cos_down - 0.20 - 0.015
            ideal_targets[is_down] = np.clip(stance_lengths, 0.025, max_extend)

            ideal_push_dir = -0.707 * np.array([d_sub[0], d_sub[1], 0.0]) + np.array([0.0, 0.0, -0.707])
            ideal_push_dir /= np.linalg.norm(ideal_push_dir)
            alignment = dirs_world @ ideal_push_dir
            valid_mask = (u_long < -0.05) & (u_z < 0.20)

            if np.any(valid_mask):
                score_norm = (alignment[valid_mask] - 0.25) / 0.12
                sig_weights = 1.0 / (1.0 + np.exp(-np.clip(score_norm, -10.0, 10.0)))
                lat_tuck = np.clip(1.0 - 1.2 * (u_lat[valid_mask] ** 2), 0.0, 1.0)
                ideal_targets[valid_mask] = np.clip(ideal_targets[valid_mask] + 0.12 * sig_weights * lat_tuck * 1.6, 0.025, max_extend)

            ideal_targets[u_long >= 0.0] = 0.025
            ideal_targets[u_z >= 0.15] = 0.025

            forces = None
            if hasattr(self.env, "get_rod_contact_forces"):
                forces = self.env.get_rod_contact_forces()

            phys_targets, act_metrics = self.actuator_sim.apply_dynamics(ideal_targets, actual_forces=forces)
            self.power_history.append(act_metrics["total_power_w"])
            self.max_vel_history.append(act_metrics["max_actuator_vel"])

            _sub_obs, sub_r, sub_term, _sub_trunc, info = self.env.step(phys_targets)
            total_r += float(sub_r)
            if sub_term:
                term = True
                break

        self._info = info
        obs = self._observe(info)
        return obs, total_r, term, False, info


def main():
    vec_env = DummyVecEnv([lambda: ViscoelasticRubberFeetEnv(cfg, scenario=sc, randomize=False, max_steps=1500)])
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
    wall_contacts = 0
    velocities = []
    z_positions = []
    angular_jerks = []
    prev_ang_vel = None

    print("Benchmarking Viscoelastic Rubber Footpads & Surface Friction...", flush=True)

    while not done and step < 400:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)

        raw = raw_env.env if hasattr(raw_env, "env") else raw_env
        pos = raw.data.qpos[:3].copy()
        z_positions.append(pos[2])

        speed = float(np.linalg.norm(raw.data.qvel[:2]))
        velocities.append(speed)

        curr_ang_vel = raw.data.qvel[3:6].copy()
        if prev_ang_vel is not None:
            angular_jerks.append(float(np.linalg.norm(curr_ang_vel - prev_ang_vel)))
        prev_ang_vel = curr_ang_vel

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
        frames_3d.append(img_3d_chase)

        done = dones[0]
        step += 1

    final_dist = float(np.linalg.norm(raw_env.env.data.qpos[:2] - sc.goal[:2]))
    success = bool(final_dist < 0.50)
    avg_speed = float(np.mean(velocities))
    z_std_mm = float(np.std(z_positions) * 1000.0)
    mean_jerk = float(np.mean(angular_jerks))
    peak_act_vel = float(np.max(raw_env.max_vel_history)) if raw_env.max_vel_history else 0.0
    mean_power = float(np.mean(raw_env.power_history)) if raw_env.power_history else 0.0

    print("\n=========================================================================")
    print("VISCOELASTIC RUBBER FOOTPADS & FRICTION RESULTS")
    print("=========================================================================")
    print(f"Goal Success: {success} | Steps: {step} | Wall Contacts: {wall_contacts} ({wall_contacts/step*100:.1f}%)")
    print(f"Z-Bounce Std Dev: {z_std_mm:.2f} mm | Angular Jerk: {mean_jerk:.4f} | Avg Speed: {avg_speed:.2f} m/s")
    print(f"Peak Actuator Velocity: {peak_act_vel:.3f} m/s | Mean Power Draw: {mean_power:.2f} W")

    # 1. Normal Speed Videos (30 FPS)
    out_normal_bird = renders_dir / "rubber_feet_bird_chase_normal.mp4"
    out_normal_dual = renders_dir / "rubber_feet_dual_bird_normal.mp4"
    out_normal_triple = renders_dir / "rubber_feet_triple_normal.mp4"
    out_normal_3d = renders_dir / "rubber_feet_3d_chase_normal.mp4"

    imageio.mimsave(str(out_normal_bird), frames_bird_chase, fps=30)
    imageio.mimsave(str(out_normal_dual), frames_dual, fps=30)
    imageio.mimsave(str(out_normal_triple), frames_triple, fps=30)
    imageio.mimsave(str(out_normal_3d), frames_3d, fps=30)

    # 2. Slow Motion Videos (12 FPS = 0.4x)
    out_slow_bird = renders_dir / "rubber_feet_bird_chase_slowmo.mp4"
    out_slow_dual = renders_dir / "rubber_feet_dual_bird_slowmo.mp4"
    out_slow_triple = renders_dir / "rubber_feet_triple_slowmo.mp4"
    out_slow_3d = renders_dir / "rubber_feet_3d_chase_slowmo.mp4"

    imageio.mimsave(str(out_slow_bird), frames_bird_chase, fps=12)
    imageio.mimsave(str(out_slow_dual), frames_dual, fps=12)
    imageio.mimsave(str(out_slow_triple), frames_triple, fps=12)
    imageio.mimsave(str(out_slow_3d), frames_3d, fps=12)

    thumb_dual = scratch_dir / "rubber_feet_dual_thumb.png"
    thumb_triple = scratch_dir / "rubber_feet_triple_thumb.png"
    thumb_bird = scratch_dir / "rubber_feet_bird_thumb.png"

    mid = len(frames_bird_chase) // 2
    imageio.imwrite(str(thumb_dual), frames_dual[mid])
    imageio.imwrite(str(thumb_triple), frames_triple[mid])
    imageio.imwrite(str(thumb_bird), frames_bird_chase[mid])

    print(f"\nAll Viscoelastic Rubber Footpad Videos Saved to {renders_dir}!")
    vec_env.close()


if __name__ == "__main__":
    main()
