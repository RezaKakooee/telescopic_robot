"""Render Transparent Glass Pipe In-Pipe Inspection with agent moving inside the pipe."""
import datetime
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.controller import bar_targets, desired_direction
from radial_sphere.scenario import generate_scenario


def main():
    print(f"\n=======================================================")
    print(f"=== Simulating In-Pipe Crawling: TRANSPARENT GLASS PIPE ===")
    print(f"=======================================================")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__in_pipe_crawling_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/rl/terrain_transparent_glass_pipe.yaml")
    scenario = generate_scenario("glass_pipe", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=1500)

    obs, info = env.reset(seed=42)
    print(f"Spawn (inside pipe): {info['ball_xy']} -> Goal: {env.scenario.goal}")
    print(f"Pipe Traversal Distance: {env.scenario.path_length:.2f}m")

    v_dual = out_dir / "in_pipe_dual_close_view.mp4"
    v_overview = out_dir / "in_pipe_stationary_overview.mp4"

    w_dual = imageio.get_writer(str(v_dual), fps=24, codec="libx264")
    w_over = imageio.get_writer(str(v_overview), fps=24, codec="libx264")

    ctrl = env.cfg.controller
    step = 0
    frames_dual = []

    while True:
        ball_pos = env.data.qpos[0:3]
        ball_vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]

        # Compute rotation matrix
        w, x, y, z = quat
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        dirs_world = env.dirs_body @ R.T

        # Cross-track centering towards pipe centerline y=0
        cross_track_y = float(ball_pos[1])
        target_heading_y = np.clip(-3.5 * cross_track_y, -0.6, 0.6)
        d_hat = np.array([np.sqrt(max(1.0 - target_heading_y**2, 0.1)), target_heading_y])
        d_hat /= np.linalg.norm(d_hat)

        u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
        u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
        u_z = dirs_world[:, 2]

        # Ideal rear-downward push vector (45-deg rear-down thrust)
        ideal_push = -0.707 * np.array([d_hat[0], d_hat[1], 0.0]) + np.array([0.0, 0.0, -0.707])
        ideal_push /= np.linalg.norm(ideal_push)
        align = dirs_world @ ideal_push

        # Dynamic peristaltic wave with lateral flank tucking
        wave = np.clip((align ** 2) * 2.8, 0.0, 1.0)
        wave = wave * np.clip(1.0 - 2.0 * (u_lat ** 2), 0.0, 1.0)
        # Leading rods (front) and top rods strictly retract to prevent clamping
        wave[u_long > -0.05] = 0.0
        wave[u_z > 0.05] = 0.0

        targets = env.max_extend * wave

        obs, rew, term, trunc, info = env.step(targets)
        step += 1

        f_dual = env.render(camera_name="fixed_close_dual")
        f_over = env.render(camera_name="fixed_corner_sw_30deg")

        w_dual.append_data(f_dual)
        w_over.append_data(f_over)
        frames_dual.append(f_dual)

        if step % 25 == 0 or term or trunc or info["distance"] < 0.45:
            print(f"Step {step:4d}: Ball pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.2f}), vx={ball_vel[0]:.2f}m/s, Dist to Goal={info['distance']:.2f}m")

        if term or trunc or info["distance"] < 0.45:
            print(f"🎉 Goal reached inside/exiting pipe at step {step}! Final dist={info['distance']:.3f}m")
            break

        if step >= 1000:
            break

    w_dual.close()
    w_over.close()
    env.close()

    duration = step / 24.0
    print(f"\nCompleted In-Pipe Evaluation: Duration {duration:.1f}s ({step} frames)")
    print(f"  - Dual Close Video: {v_dual}")
    print(f"  - Overview Video: {v_overview}")

    if len(frames_dual) > 20:
        mid = len(frames_dual) // 2
        p_img = Path("docs/project_journey/assets/terrain_glass_pipe_preview.png")
        Image.fromarray(frames_dual[mid]).save(p_img)
        Image.fromarray(frames_dual[mid]).save("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/terrain_glass_pipe_preview.png")
        print(f"Saved preview still -> {p_img}")


if __name__ == "__main__":
    main()
