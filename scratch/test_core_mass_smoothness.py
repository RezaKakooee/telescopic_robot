"""Benchmark the physical smoothness of different core_mass values [0.5kg, 2.0kg, 4.0kg, 6.0kg]."""
import os
os.environ["MUJOCO_GL"] = "egl"
from multiprocessing import Pool
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

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260823_2245__core_mass_benchmark")
renders_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")

model_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
model_path = model_dir / "checkpoints" / "ppo_final.zip"
norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

mass_configs = [
    {"mass": 0.5, "label": "0.5kg (Baseline Light)"},
    {"mass": 2.0, "label": "2.0kg (Medium Mass)"},
    {"mass": 4.0, "label": "4.0kg (Battery Payload)"},
    {"mass": 6.0, "label": "6.0kg (Heavy Momentum)"},
]


class FullDriveSteeringEnv(MujocoSteeringEnv):
    def step(self, action):
        act = np.array(action, copy=True).reshape(-1)
        if len(act) > 2:
            act[2] = 1.0
        return super().step(act)


def run_mass_trial(spec: dict) -> dict:
    mass = spec["mass"]
    label = spec["label"]
    job_id = f"core_mass_{str(mass).replace('.', '_')}kg"

    cfg = load_config_cli(name="maze_level3_large_active_braking")
    OmegaConf.set_struct(cfg, False)

    cfg.robot.core_mass = float(mass)
    cfg.scenario.maze.level = 3
    cfg.scenario.maze.random_endpoints = False
    cfg.scenario.maze.random_start = False
    cfg.scenario.maze.random_goal = False
    cfg.scenario.maze.layout_seed = 42

    sc = generate_scenario("maze", cfg, seed=42)

    def make_env():
        return FullDriveSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=1500)

    vec_env = DummyVecEnv([make_env])
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
    wall_contacts = 0
    velocities = []
    z_heights = []
    angular_jerks = []
    prev_ang_vel = None

    print(f"[{job_id}] Running simulation for core_mass = {mass} kg...", flush=True)

    while not done and step < 800:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)

        raw = raw_env.env if hasattr(raw_env, "env") else raw_env
        z_pos = float(raw.data.qpos[2])
        z_heights.append(z_pos)

        speed = float(np.linalg.norm(raw.data.qvel[:2]))
        velocities.append(speed)

        curr_ang_vel = raw.data.qvel[3:6].copy()
        if prev_ang_vel is not None:
            angular_jerks.append(float(np.linalg.norm(curr_ang_vel - prev_ang_vel)))
        prev_ang_vel = curr_ang_vel

        if infos[0].get("wall_contact", False):
            wall_contacts += 1

        frames_dual.append(raw_env.render(mode="dual"))
        frames_chase.append(raw_env.render(mode="chase"))

        done = dones[0]
        step += 1

    info = infos[0]
    ball_pos = info.get("ball_xy", raw.data.qpos[:2])
    final_dist = float(np.linalg.norm(ball_pos - sc.goal[:2]))
    success = bool(final_dist < 0.50 or info.get("success", False))
    avg_speed = float(np.mean(velocities)) if velocities else 0.0
    z_bounce_std = float(np.std(z_heights)) * 1000.0  # mm
    z_peak_to_peak = float(np.ptp(z_heights)) * 1000.0  # mm
    mean_jerk = float(np.mean(angular_jerks)) if angular_jerks else 0.0

    vid_dual = renders_dir / f"{job_id}_dual.mp4"
    vid_chase = renders_dir / f"{job_id}_chase.mp4"
    imageio.mimsave(str(vid_dual), frames_dual, fps=30)
    imageio.mimsave(str(vid_chase), frames_chase, fps=30)

    thumb_dual = scratch_dir / f"{job_id}_dual_thumb.png"
    thumb_chase = scratch_dir / f"{job_id}_chase_thumb.png"
    imageio.imwrite(str(thumb_dual), frames_dual[len(frames_dual)//2])
    imageio.imwrite(str(thumb_chase), frames_chase[len(frames_chase)//2])

    vec_env.close()

    res = {
        "mass": mass,
        "label": label,
        "steps": step,
        "success": success,
        "wall_contacts": wall_contacts,
        "wall_pct": wall_contacts / max(step, 1) * 100.0,
        "avg_speed": avg_speed,
        "z_bounce_std_mm": z_bounce_std,
        "z_ptp_mm": z_peak_to_peak,
        "mean_jerk": mean_jerk,
        "dual_video": str(vid_dual),
        "chase_video": str(vid_chase),
        "thumb_dual": str(thumb_dual),
        "thumb_chase": str(thumb_chase),
    }
    print(f"[{job_id}] FINISHED! Steps={step} | Success={success} | Z-Bounce Std={z_bounce_std:.2f}mm | PTP={z_peak_to_peak:.2f}mm | Jerk={mean_jerk:.3f} | Speed={avg_speed:.2f}m/s", flush=True)
    return res


def main():
    print("=========================================================================================")
    print("CORE MASS SMOOTHNESS BENCHMARK [0.5kg, 2.0kg, 4.0kg, 6.0kg] (DUAL + CHASE)")
    print("=========================================================================================")

    with Pool(processes=4) as pool:
        results = pool.map(run_mass_trial, mass_configs)

    print("\n=========================================================================================")
    print("CORE MASS BENCHMARK RESULTS TABLE")
    print("=========================================================================================")
    print(f"{'Core Mass':25s} | {'Success':7s} | {'Steps':6s} | {'Z-Bounce Std':13s} | {'Z Peak-to-Peak':15s} | {'Angular Jerk':13s} | {'Avg Speed':10s}")
    print("-" * 108)
    for r in results:
        print(f"{r['label']:25s} | {str(r['success']):7s} | {r['steps']:6d} | {r['z_bounce_std_mm']:9.2f} mm | {r['z_ptp_mm']:11.2f} mm | {r['mean_jerk']:13.4f} | {r['avg_speed']:6.2f} m/s")

    import json
    with open(renders_dir / "core_mass_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
