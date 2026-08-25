"""Render Cinematic 3D Multi-Perspective Videos for Complex Maze with Blockers.

Features:
1. 'cinematic_chase_3d': Elevated -42° rear tracking camera that looks down into the corridor with zero wall clipping.
2. 'iso_3d_overview': 3D Isometric View of the entire maze showing 3D walls, shadows, and all 4 safety bollards.
3. 'cinematic_dual': Side-by-side composite showing both isometric arena overview and corridor chase.
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


def render_cinematic_suite():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__cinematic_maze_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering Cinematic 3D Maze Suite -> {out_dir} ===")

    # Load latest trained PPO model
    run_dir = sorted(Path("storage_local").glob("*train_mujoco_rl*maze_complex_blockers*"))[-1]
    ckpt_dir = run_dir / "checkpoints"
    model_files = sorted(ckpt_dir.glob("ppo_*.zip"))
    model_path = model_files[-1]
    print(f"Loading Model: {model_path}")

    model = PPO.load(str(model_path), device="cpu")
    cfg = load_config("configs/rl/maze_complex_blockers_random_endpoints.yaml")

    # Seed 1037 (Episode 2: Clean 12.0m goal route across corridor bollards)
    seed = 1037
    scenario = generate_scenario("maze", cfg, seed=seed)
    env = MujocoSteeringEnv(cfg, scenario=scenario, randomize=False, max_steps=3000)

    obs, info = env.reset(seed=seed)
    print(f"Spawn: {info['ball_xy']} -> Goal: {env.env.scenario.goal}")
    print(f"Path Length: {env.env.scenario.path_length:.2f}m")

    v_dual = out_dir / "cinematic_dual_3d_composite.mp4"
    v_chase = out_dir / "cinematic_chase_3d_corridor.mp4"
    v_iso = out_dir / "cinematic_iso_3d_arena.mp4"

    w_dual = imageio.get_writer(str(v_dual), fps=24, codec="libx264")
    w_chase = imageio.get_writer(str(v_chase), fps=24, codec="libx264")
    w_iso = imageio.get_writer(str(v_iso), fps=24, codec="libx264")

    step = 0
    frames_dual = []
    frames_chase = []
    frames_iso = []

    print("\nRendering trajectory...")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(action)
        step += 1

        f_dual = env.render(camera_name="cinematic_dual")
        f_chase = env.render(camera_name="cinematic_chase_3d")
        f_iso = env.render(camera_name="iso_3d_overview")

        w_dual.append_data(f_dual)
        w_chase.append_data(f_chase)
        w_iso.append_data(f_iso)

        frames_dual.append(f_dual)
        frames_chase.append(f_chase)
        frames_iso.append(f_iso)

        if step % 50 == 0:
            print(f"  Step {step:3d}: Dist to Goal={info['distance']:.2f}m")

        if term or trunc or info["distance"] < 0.45:
            print(f"🎉 Goal reached at step {step}! Final dist={info['distance']:.3f}m")
            break

        if step >= 600:
            break

    w_dual.close()
    w_chase.close()
    w_iso.close()
    env.close()

    duration = step / 24.0
    print(f"\nRender Complete! Duration: {duration:.1f}s ({step} frames)")
    print(f"  - Dual 3D Composite: {v_dual}")
    print(f"  - 3D Corridor Chase: {v_chase}")
    print(f"  - 3D Isometric Arena: {v_iso}")

    # Extract preview stills
    mid = len(frames_dual) // 2
    p_dual = Path("docs/project_journey/assets/cinematic_3d_dual_preview.png")
    Image.fromarray(frames_dual[mid]).save(p_dual)
    p_chase = Path("docs/project_journey/assets/cinematic_3d_chase_preview.png")
    Image.fromarray(frames_chase[mid]).save(p_chase)
    p_iso = Path("docs/project_journey/assets/cinematic_3d_iso_preview.png")
    Image.fromarray(frames_iso[mid]).save(p_iso)
    print(f"Saved preview stills to docs/project_journey/assets/!")


if __name__ == "__main__":
    render_cinematic_suite()
