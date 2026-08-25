"""Render Complex 7x6 Maze using our 4 Fixed Outside Perimeter Edge Cameras at 30°.

Renders:
1. Fixed West Edge 30° camera (looking East)
2. Fixed South Edge 30° camera (looking North)
3. Fixed East Edge 30° camera (looking West)
4. Fixed North Edge 30° camera (looking South)
5. 2x2 Fixed Quad Multi-View Grid (all 4 fixed outside cameras simultaneously)
"""
import datetime
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from radial_sphere import MujocoSteeringEnv, generate_scenario, load_config_cli


def run_fixed_side_cameras_renders():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__fixed_side_cameras_suite")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering Complex Maze with 4 Fixed Side Cameras at 30° -> {out_dir} ===")

    model_dir = Path("storage_local/20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
    model_path = model_dir / "checkpoints" / "ppo_final.zip"
    norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

    cfg = load_config_cli(name="maze_level3_large_blockers")
    cfg.scenario.maze.layout_seed = 42
    sc = generate_scenario("maze", cfg, seed=42)

    def make_env():
        return MujocoSteeringEnv(cfg, scenario=sc, max_steps=800)

    venv = DummyVecEnv([make_env])
    if norm_path.exists():
        venv = VecNormalize.load(str(norm_path), venv)
        venv.training = False
        venv.norm_reward = False

    policy = PPO.load(str(model_path))
    raw_env = venv.envs[0]
    obs = venv.reset()

    # Video Writers for the 4 Fixed Outside Edge Cameras & 2x2 Fixed Quad
    writers = {
        "fixed_edge_west_30deg": imageio.get_writer(str(out_dir / "fixed_edge_west_30deg.mp4"), fps=24, codec="libx264"),
        "fixed_edge_south_30deg": imageio.get_writer(str(out_dir / "fixed_edge_south_30deg.mp4"), fps=24, codec="libx264"),
        "fixed_edge_east_30deg": imageio.get_writer(str(out_dir / "fixed_edge_east_30deg.mp4"), fps=24, codec="libx264"),
        "fixed_edge_north_30deg": imageio.get_writer(str(out_dir / "fixed_edge_north_30deg.mp4"), fps=24, codec="libx264"),
        "fixed_quad": imageio.get_writer(str(out_dir / "fixed_quad_4_views_2x2.mp4"), fps=24, codec="libx264"),
    }

    print(f"Complex Maze: 7 cols x 6 rows (layout_seed=42)")
    print(f"Spawn: {raw_env.env.scenario.spawn_xy}, Goal: {raw_env.env.scenario.goal}")
    print(f"Bollards: {len(raw_env.env.scenario.obstacles) if raw_env.env.scenario.obstacles is not None else 0}")

    frames_fixed_quad = []

    for step in range(300):
        action, _ = policy.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)

        if step % 2 == 0:
            for cam_name in ["fixed_edge_west_30deg", "fixed_edge_south_30deg", "fixed_edge_east_30deg", "fixed_edge_north_30deg", "fixed_quad"]:
                img = raw_env.env.render(camera_name=cam_name)
                writers[cam_name].append_data(img)
                if cam_name == "fixed_quad":
                    frames_fixed_quad.append(img)

        if dones[0]:
            print(f"Episode reached goal at step {step + 1}!")
            break

    for w in writers.values():
        w.close()
    venv.close()

    print(f"\nAll 4 Fixed Side Camera Videos & 2x2 Fixed Quad Multi-View saved to:\n  {out_dir}")

    # Extract Fixed Quad snapshot
    if len(frames_fixed_quad) > 30:
        preview_path = Path("docs/project_journey/assets/complex_maze_fixed_quad_side_views.png")
        Image.fromarray(frames_fixed_quad[30]).save(preview_path)
        print(f"Saved Fixed Quad Preview still -> {preview_path}")


if __name__ == "__main__":
    run_fixed_side_cameras_renders()
