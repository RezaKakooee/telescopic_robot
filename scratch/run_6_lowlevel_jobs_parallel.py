"""Run all 6 low-level enhancement jobs in parallel with full drive=1.0 storing both DUAL and CHASE video views."""
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

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260823_1540__six_lowlevel_jobs_benchmark")
renders_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")

model_dir = Path("/home/azureuser/telescopic_robot/storage_local/20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
model_path = model_dir / "checkpoints" / "ppo_final.zip"
norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

job_specs = [
    {
        "id": "job_0_baseline",
        "name": "Job 0: Baseline (All 5 Off)",
        "params": {},
    },
    {
        "id": "job_1_power_wave",
        "name": "Job 1: Asymmetric Power-Cosine Wave Only",
        "params": {
            "enable_power_wave": True,
            "wave_power_exponent": 1.4,
        },
    },
    {
        "id": "job_2_flank_retraction",
        "name": "Job 2: Active Flank Retraction Only",
        "params": {
            "enable_flank_retraction": True,
            "flank_retract_dist": 0.45,
            "flank_min_offset": 0.005,
        },
    },
    {
        "id": "job_3_camber_banking",
        "name": "Job 3: Dynamic Camber Banking Only",
        "params": {
            "enable_camber_banking": True,
            "camber_bank_gain": 0.035,
        },
    },
    {
        "id": "job_4_force_compliance",
        "name": "Job 4: Contact Force Compliance Only",
        "params": {
            "enable_contact_compliance": True,
            "compliance_gain": 0.0005,
            "max_contact_force": 40.0,
        },
    },
    {
        "id": "job_5_anti_stall_reflex",
        "name": "Job 5: Anti-Stall Reflex Only",
        "params": {
            "enable_anti_stall_reflex": True,
            "anti_stall_speed_threshold": 0.15,
            "anti_stall_pulse_freq": 10.0,
            "anti_stall_pulse_amp": 0.02,
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

    print(f"[{job_id}] Started execution with full drive=1.0...", flush=True)

    while not done and step < 1200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        total_reward += float(reward[0])

        raw = raw_env.env if hasattr(raw_env, "env") else raw_env
        curr_joints = raw.data.qpos[7:7 + raw.n_bars].copy()
        if prev_joints is not None:
            joint_efforts.append(float(np.sum(np.abs(curr_joints - prev_joints))))
        prev_joints = curr_joints

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

    vid_dual = renders_dir / f"{job_id}_full_drive_dual.mp4"
    vid_chase = renders_dir / f"{job_id}_full_drive_chase.mp4"
    imageio.mimsave(str(vid_dual), frames_dual, fps=25)
    imageio.mimsave(str(vid_chase), frames_chase, fps=25)

    thumb_dual = scratch_dir / f"{job_id}_full_drive_dual_thumb.png"
    thumb_chase = scratch_dir / f"{job_id}_full_drive_chase_thumb.png"
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
        "total_reward": total_reward,
        "dual_video": str(vid_dual),
        "chase_video": str(vid_chase),
        "thumb_dual": str(thumb_dual),
        "thumb_chase": str(thumb_chase),
    }
    print(f"[{job_id}] FINISHED! Steps={step} | Success={success} | WallHits={wall_contacts} ({res['wall_pct']:.1f}%) | AvgSpeed={avg_speed:.2f}m/s | PeakSpeed={peak_speed:.2f}m/s", flush=True)
    return res


def main():
    print("=========================================================================================")
    print("RUNNING ALL 6 LOW-LEVEL ENHANCEMENT JOBS IN PARALLEL (FULL DRIVE = 1.0, DUAL + CHASE)")
    print("=========================================================================================")

    with Pool(processes=6) as pool:
        results = pool.map(run_single_job, job_specs)

    print("\n=========================================================================================")
    print("PARALLEL 6-JOB BENCHMARK RESULTS TABLE (FULL VISIBLE EXTENSIONS)")
    print("=========================================================================================")
    print(f"{'Job Name':42s} | {'Success':7s} | {'Steps':6s} | {'Wall Hits (%)':15s} | {'Avg Speed':10s} | {'Peak Speed':10s} | {'Effort':8s}")
    print("-" * 110)
    for r in results:
        wall_str = f"{r['wall_contacts']:3d} ({r['wall_pct']:4.1f}%)"
        print(f"{r['name']:42s} | {str(r['success']):7s} | {r['steps']:6d} | {wall_str:15s} | {r['avg_speed']:6.2f} m/s | {r['peak_speed']:6.2f} m/s | {r['total_effort']:7.1f}")

    import json
    with open(renders_dir / "benchmark_full_drive_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
