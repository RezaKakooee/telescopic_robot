"""Evaluate Trained RL Policy on Complex 7x6 Maze with Industrial Blockers and Random Start/Goal.

Features:
1. Tests multiple episodes with completely randomized start and goal pairs.
2. 4 Heavy Cast-Steel Safety Bollards stationed inside corridor intersections.
3. Renders high-definition Dual, Ground-Level Underbelly, and Quad 4-Side Tracking videos.
"""
import datetime
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.mujoco_steering import MujocoSteeringEnv
from radial_sphere.scenario import generate_scenario


def evaluate_policy(run_dir: Path | None = None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    eval_dir = Path(f"storage_local/{timestamp}__eval_complex_maze_blockers")
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Evaluating Trained RL Agent on Complex Maze with Blockers -> {eval_dir} ===")

    # Find latest checkpoint
    if run_dir is None:
        candidates = sorted(Path("storage_local").glob("*train_mujoco_rl*maze_complex_blockers*"))
        if candidates:
            run_dir = candidates[-1]
        else:
            raise FileNotFoundError("No training run directory found!")

    ckpt_dir = run_dir / "checkpoints"
    model_files = sorted(ckpt_dir.glob("ppo_*.zip"))
    if not model_files:
        raise FileNotFoundError(f"No model checkpoints found in {ckpt_dir}")
    model_path = model_files[-1]
    norm_path = ckpt_dir / "vecnormalize_final.pkl"
    if not norm_path.exists():
        norm_files = sorted(ckpt_dir.glob("vecnormalize_*.pkl"))
        if norm_files:
            norm_path = norm_files[-1]

    print(f"Loading Model Checkpoint: {model_path}")
    print(f"Loading VecNormalize: {norm_path if norm_path.exists() else 'None'}")

    cfg = load_config("configs/rl/maze_complex_blockers_random_endpoints.yaml")
    
    # Run 5 evaluation episodes with randomized start & goal
    n_episodes = 5
    results = []

    model = PPO.load(str(model_path), device="cpu")

    for ep in range(n_episodes):
        ep_seed = 1000 + ep * 37
        print(f"\n--- Episode {ep + 1}/{n_episodes} (Seed: {ep_seed}) ---")
        scenario = generate_scenario("maze", cfg, seed=ep_seed)
        env = MujocoSteeringEnv(cfg, scenario=scenario, randomize=False, max_steps=3000)
        
        obs, info = env.reset(seed=ep_seed)
        spawn_pt = info["ball_xy"]
        goal_pt = env.env.scenario.goal
        path_len = float(env.env.scenario.path_length)

        print(f"  Spawn: ({spawn_pt[0]:.2f}, {spawn_pt[1]:.2f}) -> Goal: ({goal_pt[0]:.2f}, {goal_pt[1]:.2f})")
        print(f"  Route Length: {path_len:.2f} meters across corridors with 4 industrial bollards")

        v_dual = eval_dir / f"eval_ep{ep + 1}_dual_map_and_chase.mp4"
        v_ground = eval_dir / f"eval_ep{ep + 1}_ground_level_underbelly.mp4"
        v_quad = eval_dir / f"eval_ep{ep + 1}_quad_4_views.mp4"

        w_dual = imageio.get_writer(str(v_dual), fps=24, codec="libx264")
        w_ground = imageio.get_writer(str(v_ground), fps=24, codec="libx264")
        w_quad = imageio.get_writer(str(v_quad), fps=24, codec="libx264")

        step = 0
        total_rew = 0.0
        success = False
        min_dist = info["distance"]
        frames_dual = []

        while True:
            # Deterministic action inference
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            step += 1
            total_rew += float(rew)
            min_dist = min(min_dist, info["distance"])

            f_dual = env.render(camera_name="dual")
            f_ground = env.render(camera_name="underbelly_dual")
            f_quad = env.render(camera_name="quad")

            w_dual.append_data(f_dual)
            w_ground.append_data(f_ground)
            w_quad.append_data(f_quad)
            frames_dual.append(f_dual)

            if step % 100 == 0:
                print(f"    Step {step:4d}: Dist to Goal={info['distance']:.2f}m, Min Dist={min_dist:.2f}m")

            if term or trunc or info["distance"] < 0.45:
                success = info.get("success", False) or info["distance"] < 0.45
                print(f"  Episode {ep + 1} Done at Step {step}! Success={success}, Final Dist={info['distance']:.3f}m")
                break

            if step >= 1500:
                print(f"  Reached step limit at Step {step}. Final Dist={info['distance']:.2f}m")
                break

        w_dual.close()
        w_ground.close()
        w_quad.close()
        env.close()

        results.append({
            "episode": ep + 1,
            "success": success,
            "steps": step,
            "path_length": path_len,
            "min_dist": min_dist,
            "video_dual": v_dual,
            "video_ground": v_ground,
            "video_quad": v_quad,
            "frames_dual": frames_dual,
        })

    # Summary
    successes = sum(r["success"] for r in results)
    print("\n=======================================================")
    print(f"Evaluation Summary ({n_episodes} Randomized Episodes):")
    print(f"  - Success Rate: {successes}/{n_episodes} ({successes / n_episodes * 100:.1f}%)")
    print(f"  - Mean Steps: {np.mean([r['steps'] for r in results]):.1f}")
    print(f"  - Output Video Directory: {eval_dir}")
    print("=======================================================")

    # Save preview image from first successful episode
    if len(results[0]["frames_dual"]) > 50:
        p_img = Path("docs/project_journey/assets/eval_complex_maze_blockers_random_preview.png")
        Image.fromarray(results[0]["frames_dual"][len(results[0]["frames_dual"]) // 2]).save(p_img)
        print(f"Saved evaluation preview still -> {p_img}")

    return results


if __name__ == "__main__":
    evaluate_policy()
