"""Run 6 parallel evaluation jobs comparing baseline vs. each of the 5 smooth maneuver enhancements."""
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

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/six_smooth_jobs_benchmark")
renders_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")

model_dir = Path("/home/azureuser/telescopic_robot/storage_local/radial__20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
model_path = model_dir / "checkpoints" / "ppo_final.zip"
norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

job_specs = [
    {
        "id": "job_0_baseline",
        "name": "Job 0: Baseline Normal (All 5 Smoothing Off)",
        "params": {},
    },
    {
        "id": "job_1_spline_heading",
        "name": "Job 1: Continuous Spline Heading Only",
        "params": {
            "enable_spline_heading": True,
            "spline_smoothing_weight": 0.8,
        },
    },
    {
        "id": "job_2_curvature_glide",
        "name": "Job 2: Curvature-Adaptive Deceleration Glide Only",
        "params": {
            "enable_curvature_deceleration": True,
            "curvature_lookahead_dist": 1.2,
            "curvature_brake_gain": 1.8,
        },
    },
    {
        "id": "job_3_actuator_slew_rate",
        "name": "Job 3: Actuator Slew-Rate Limiter Only",
        "params": {
            "enable_actuator_slew_rate": True,
            "actuator_max_vel": 0.35,
        },
    },
    {
        "id": "job_4_gaussian_stance",
        "name": "Job 4: Gaussian Stance Handoff Only",
        "params": {
            "enable_gaussian_stance": True,
            "gaussian_stance_sigma": 0.38,
        },
    },
    {
        "id": "job_5_gyroscopic_damping",
        "name": "Job 5: Gyroscopic Precession Damping Only",
        "params": {
            "enable_gyroscopic_damping": True,
            "gyroscopic_damping_gain": 0.025,
        },
    },
]


class FullDriveSteeringEnv(MujocoSteeringEnv):
    def step(self, action):
        act = np.array(action, copy=True).reshape(-1)
        if len(act) > 2:
            act[2] = 1.0  # Lock full 100% drive throttle for full visible extension
        return super().step(act)


def run_single_job(job_info: dict) -> dict:
    job_id = job_info["id"]
    job_name = job_info["name"]
    params = job_info["params"]

    cfg = load_config_cli(name="maze_level3_large_active_braking")
    OmegaConf.set_struct(cfg, False)

    cfg.scenario.maze.level = 3
    cfg.scenario.maze.random_endpoints = False
    cfg.scenario.maze.random_start = False
    cfg.scenario.maze.random_goal = False
    cfg.scenario.maze.layout_seed = 42

    for k, v in params.items():
        setattr(cfg.controller, k, v)

    sc = generate_scenario("maze", cfg, seed=42)

    def make_env():
        return FullDriveSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=2000)

    vec_env = DummyVecEnv([make_env])
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env, device="cpu")
    obs = vec_env.reset()
    raw_env = vec_env.envs[0]

    frames_dual = [raw_env.render(mode="dual")]
    frames_chase = [raw_env.render(mode="chase")]

    done = False
    step = 0
    total_reward = 0.0
    wall_contacts = 0
    velocities = []
    prev_joints = None
    joint_efforts = []
    prev_ang_vel = None
    angular_jerks = []

    print(f"[{job_id}] Started execution...", flush=True)

    while not done and step < 1200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        total_reward += float(reward[0])

        raw = raw_env.env if hasattr(raw_env, "env") else raw_env
        curr_joints = raw.data.qpos[7:7 + raw.n_bars].copy()
        if prev_joints is not None:
            joint_efforts.append(float(np.sum(np.abs(curr_joints - prev_joints))))
        prev_joints = curr_joints

        curr_ang_vel = raw.data.qvel[3:6].copy()
        if prev_ang_vel is not None:
            angular_jerks.append(float(np.linalg.norm(curr_ang_vel - prev_ang_vel)))
        prev_ang_vel = curr_ang_vel

        speed = float(np.linalg.norm(raw.data.qvel[:2]))
        velocities.append(speed)

        if infos[0].get("wall_contact", False):
            wall_contacts += 1

        if step % 2 == 0:
            frames_dual.append(raw_env.render(mode="dual"))
            frames_chase.append(raw_env.render(mode="chase"))

        done = dones[0]
        step += 1

    info = infos[0]
    ball_pos = info.get("ball_xy", raw.data.qpos[:2])
    final_dist = float(np.linalg.norm(ball_pos - sc.goal[:2]))
    success = bool(final_dist < 0.50 or info.get("success", False))
    avg_speed = float(np.mean(velocities)) if velocities else 0.0
    peak_speed = float(np.max(velocities)) if velocities else 0.0
    total_effort = float(np.sum(joint_efforts)) if joint_efforts else 0.0
    mean_jerk = float(np.mean(angular_jerks)) if angular_jerks else 0.0

    vid_dual = renders_dir / f"{job_id}_smooth_dual.mp4"
    vid_chase = renders_dir / f"{job_id}_smooth_chase.mp4"
    imageio.mimsave(str(vid_dual), frames_dual, fps=25)
    imageio.mimsave(str(vid_chase), frames_chase, fps=25)

    thumb_dual = scratch_dir / f"{job_id}_smooth_dual_thumb.png"
    thumb_chase = scratch_dir / f"{job_id}_smooth_chase_thumb.png"
    imageio.imwrite(str(thumb_dual), frames_dual[len(frames_dual)//2])
    imageio.imwrite(str(thumb_chase), frames_chase[len(frames_chase)//2])

    vec_env.close()

    res = {
        "id": job_id,
        "name": job_name,
        "steps": step,
        "success": success,
        "final_dist": final_dist,
        "wall_contacts": wall_contacts,
        "wall_pct": wall_contacts / max(step, 1) * 100.0,
        "avg_speed": avg_speed,
        "peak_speed": peak_speed,
        "total_effort": total_effort,
        "mean_jerk": mean_jerk,
        "total_reward": total_reward,
        "dual_video": str(vid_dual),
        "chase_video": str(vid_chase),
        "thumb_dual": str(thumb_dual),
        "thumb_chase": str(thumb_chase),
    }
    print(f"[{job_id}] FINISHED! Steps={step} | Success={success} | WallHits={wall_contacts} ({res['wall_pct']:.1f}%) | Jerk={mean_jerk:.3f} | AvgSpeed={avg_speed:.2f}m/s", flush=True)
    return res


def main():
    print("=========================================================================================")
    print("RUNNING 6 SMOOTH MANEUVER ENHANCEMENT JOBS IN PARALLEL (DUAL + CHASE)")
    print("=========================================================================================")

    with Pool(processes=6) as pool:
        results = pool.map(run_single_job, job_specs)

    print("\n=========================================================================================")
    print("PARALLEL 6-JOB SMOOTH MANEUVER BENCHMARK RESULTS TABLE")
    print("=========================================================================================")
    print(f"{'Job Name':48s} | {'Success':7s} | {'Steps':6s} | {'Wall Hits (%)':15s} | {'Jerk (Smoothness)':18s} | {'Avg Speed':10s} | {'Effort':8s}")
    print("-" * 125)
    for r in results:
        wall_str = f"{r['wall_contacts']:3d} ({r['wall_pct']:4.1f}%)"
        print(f"{r['name']:48s} | {str(r['success']):7s} | {r['steps']:6d} | {wall_str:15s} | {r['mean_jerk']:18.4f} | {r['avg_speed']:6.2f} m/s | {r['total_effort']:7.1f}")

    import json
    with open(renders_dir / "benchmark_smooth_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
