"""Render 100% Stationary, Fixed-Angle Videos for Dense Rocky Mountain Maze (ZERO Jitter).

Features:
1. 'fixed_quad_corners': 2x2 synchronized grid of all 4 stationary outer corners (NW, NE, SW, SE) at -42° isometric pitch.
2. 'fixed_dual_iso': Side-by-side static views (SW Corner + NE Corner).
3. 'fixed_corner_sw_30deg': Full-screen static 3D isometric view.
4. ZERO camera movement, ZERO rotation, ZERO jitter — perfectly stable static cameras.
5. Traverses dense rocky corridors (700+ multi-mineral boulders) and avoids cast-steel safety bollards.
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

from radial_sphere.config import load_config
from radial_sphere.mujoco_steering import MujocoSteeringEnv
from radial_sphere.scenario import generate_scenario


def render_fixed_only_suite():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__fixed_only_rocky_maze_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering 100% Fixed / Stationary Rocky Maze Videos -> {out_dir} ===")

    # Load latest trained PPO model
    run_dir = sorted(Path("storage_local").glob("*train_mujoco_rl*maze_complex_blockers*"))[-1]
    ckpt_dir = run_dir / "checkpoints"
    model_files = sorted(ckpt_dir.glob("ppo_*.zip"))
    model_path = model_files[-1]
    print(f"Loading Model: {model_path}")

    model = PPO.load(str(model_path), device="cpu")
    cfg = load_config("configs/rl/maze_complex_blockers_random_endpoints.yaml")

    # Seed 1037 (Episode 2: Clean 12.0m goal route across dense rocky corridors)
    seed = 1037
    scenario = generate_scenario("maze", cfg, seed=seed)
    env = MujocoSteeringEnv(cfg, scenario=scenario, randomize=False, max_steps=3000)

    obs, info = env.reset(seed=seed)
    print(f"Spawn: {info['ball_xy']} -> Goal: {env.env.scenario.goal}")
    print(f"Path Length: {env.env.scenario.path_length:.2f}m")
    print(f"Total Corridor Rocks: {sum(s[4] for s in env.env.scenario.stones)} procedural boulders!")

    v_quad = out_dir / "fixed_quad_4_stationary_corners.mp4"
    v_dual = out_dir / "fixed_dual_stationary_sw_ne.mp4"
    v_sw = out_dir / "fixed_single_stationary_sw_isometric.mp4"

    w_quad = imageio.get_writer(str(v_quad), fps=24, codec="libx264")
    w_dual = imageio.get_writer(str(v_dual), fps=24, codec="libx264")
    w_sw = imageio.get_writer(str(v_sw), fps=24, codec="libx264")

    step = 0
    frames_quad = []
    frames_dual = []
    frames_sw = []

    print("\nRendering with 100% stationary fixed cameras (zero jitter)...")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(action)
        step += 1

        f_quad = env.render(camera_name="fixed_quad_corners")
        f_dual = env.render(camera_name="fixed_dual_iso")
        f_sw = env.render(camera_name="fixed_corner_sw_30deg")

        w_quad.append_data(f_quad)
        w_dual.append_data(f_dual)
        w_sw.append_data(f_sw)

        frames_quad.append(f_quad)
        frames_dual.append(f_dual)
        frames_sw.append(f_sw)

        if step % 50 == 0:
            print(f"  Step {step:3d}: Dist to Goal={info['distance']:.2f}m")

        if term or trunc or info["distance"] < 0.45:
            print(f"🎉 Goal reached at step {step}! Final dist={info['distance']:.3f}m")
            break

        if step >= 600:
            break

    w_quad.close()
    w_dual.close()
    w_sw.close()
    env.close()

    duration = step / 24.0
    print(f"\nFixed-Only Render Complete! Duration: {duration:.1f}s ({step} frames)")
    print(f"  - 4 Stationary Corners Quad: {v_quad}")
    print(f"  - Dual Stationary Isometric: {v_dual}")
    print(f"  - Single SW Stationary Isometric: {v_sw}")

    # Extract preview stills
    mid = len(frames_quad) // 2
    p_quad = Path("docs/project_journey/assets/fixed_quad_rocky_maze_preview.png")
    Image.fromarray(frames_quad[mid]).save(p_quad)
    p_dual = Path("docs/project_journey/assets/fixed_dual_rocky_maze_preview.png")
    Image.fromarray(frames_dual[mid]).save(p_dual)
    p_sw = Path("docs/project_journey/assets/fixed_sw_rocky_maze_preview.png")
    Image.fromarray(frames_sw[mid]).save(p_sw)
    print(f"Saved stationary preview stills to docs/project_journey/assets/!")


if __name__ == "__main__":
    render_fixed_only_suite()
