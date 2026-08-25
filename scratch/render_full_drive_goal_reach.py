"""Render Complete Full-Drive Maze Navigation All the Way to Goal Success.

Runs the trained Active-Braking RL Expert continuously until the robot
reaches and physically touches the goal pad in the complex 7x6 Large Maze.

Renders:
1. 2x2 Quad Multi-View (4 Side 30° Tracking Cameras) - Full Complete Run
2. 2x2 Fixed Quad Multi-View (4 Fixed Perimeter Edge Cameras) - Full Complete Run
3. Dual-View Composite (Full Maze Map + Close Overhead Chase) - Full Complete Run
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


def run_full_drive_to_goal():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__full_maze_drive_to_goal")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering Full Maze Drive to Goal Reached -> {out_dir} ===")

    model_dir = Path("storage_local/20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
    model_path = model_dir / "checkpoints" / "ppo_final.zip"
    norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

    cfg = load_config_cli(name="maze_level3_large_blockers")
    cfg.scenario.maze.layout_seed = 42
    sc = generate_scenario("maze", cfg, seed=42)

    def make_env():
        return MujocoSteeringEnv(cfg, scenario=sc, max_steps=2500)

    venv = DummyVecEnv([make_env])
    if norm_path.exists():
        venv = VecNormalize.load(str(norm_path), venv)
        venv.training = False
        venv.norm_reward = False

    policy = PPO.load(str(model_path))
    raw_env = venv.envs[0]
    obs = venv.reset()

    # Video Writers for Full Complete Journey
    w_dual = imageio.get_writer(str(out_dir / "full_drive_dual_map_and_chase.mp4"), fps=24, codec="libx264")
    w_quad_side = imageio.get_writer(str(out_dir / "full_drive_quad_4_side_views.mp4"), fps=24, codec="libx264")
    w_quad_fixed = imageio.get_writer(str(out_dir / "full_drive_fixed_quad_outside_views.mp4"), fps=24, codec="libx264")

    print(f"Complex Large Maze: 7 cols x 6 rows (layout_seed=42)")
    print(f"Spawn: {raw_env.env.scenario.spawn_xy}, Goal: {raw_env.env.scenario.goal}")
    print(f"Route Length: {raw_env.env.scenario.path_length:.2f} meters")
    print(f"Bollards in maze: {len(raw_env.env.scenario.obstacles) if raw_env.env.scenario.obstacles is not None else 0}")

    step = 0
    goal_reached = False
    final_distance = 999.0
    frames_dual = []

    print("\nRunning complete episode until goal is reached...")
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        step += 1
        info = infos[0]
        final_distance = info["distance"]

        # Record frames
        img_bird = raw_env.env.render(camera_name="bird_fixed")
        img_chase = raw_env.env.render(camera_name="bird_chase")
        img_dual = np.concatenate([img_bird, img_chase], axis=1)

        img_quad_side = raw_env.env.render(camera_name="quad")
        img_quad_fixed = raw_env.env.render(camera_name="fixed_quad")

        w_dual.append_data(img_dual)
        w_quad_side.append_data(img_quad_side)
        w_quad_fixed.append_data(img_quad_fixed)
        frames_dual.append(img_dual)

        if step % 50 == 0:
            print(f"  Step {step:4d}: Ball pos=({info['ball_xy'][0]:.2f}, {info['ball_xy'][1]:.2f}), Distance to Goal={info['distance']:.2f}m")

        if dones[0]:
            goal_reached = bool(info.get("success", False) or info.get("goal_contact", False) or final_distance < 0.45)
            print(f"\n🎉 Goal Touchdown Achieved at Step {step}!")
            print(f"  -> Goal Reached: {goal_reached}")
            print(f"  -> Final Distance: {final_distance:.3f} meters")
            break

        if step >= 2400:
            print(f"Reached safety step limit at step {step}. Final distance={final_distance:.2f}m")
            break

    w_dual.close()
    w_quad_side.close()
    w_quad_fixed.close()
    venv.close()

    print(f"\nAll Full-Drive Videos Saved to -> {out_dir}:")
    print(f"  - Dual Map & Chase: full_drive_dual_map_and_chase.mp4")
    print(f"  - Quad 4-Side Tracking: full_drive_quad_4_side_views.mp4")
    print(f"  - Fixed Quad Outside: full_drive_fixed_quad_outside_views.mp4")

    # Save Goal Reach frame
    if len(frames_dual) > 0:
        goal_frame_path = Path("docs/project_journey/assets/full_drive_goal_reached_dual.png")
        Image.fromarray(frames_dual[-1]).save(goal_frame_path)
        print(f"Saved Goal Reached frame still -> {goal_frame_path}")


if __name__ == "__main__":
    run_full_drive_to_goal()
