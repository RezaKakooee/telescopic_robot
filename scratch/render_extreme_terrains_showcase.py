"""Extreme Multi-Terrain Showcase: Slopes, Staircases, Transparent Glass Pipe, and Gauntlet.

Features:
1. 'slopes': Uphill +18° ramp incline, elevated plateau, and -18° downhill descent.
2. 'stairs': 5 ascending steps (rise 5cm), elevated landing, and 5 descending steps.
3. 'glass_pipe': 8.5m transparent glass pipe (rgba transparency + specular reflection) with in-pipe crawling.
4. 'extreme_gauntlet': Full 28-meter adventure combining slopes, rocky boulders, stairs, and glass pipe.
5. 100% stationary and fixed-orientation zero-jitter cameras.
"""
import argparse
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


def run_terrain_simulation(scenario_kind: str, out_dir: Path):
    print(f"\n=======================================================")
    print(f"=== Simulating Extreme Terrain: {scenario_kind.upper()} ===")
    print(f"=======================================================")

    cfg_map = {
        "slopes": "configs/rl/terrain_slopes_and_ramps.yaml",
        "stairs": "configs/rl/terrain_staircase_steps.yaml",
        "glass_pipe": "configs/rl/terrain_transparent_glass_pipe.yaml",
        "extreme_gauntlet": "configs/rl/terrain_extreme_gauntlet.yaml",
    }
    cfg_file = cfg_map.get(scenario_kind, "configs/rl/terrain_extreme_gauntlet.yaml")
    cfg = load_config(cfg_file)

    scenario = generate_scenario(scenario_kind, cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=2500)

    obs, info = env.reset(seed=42)
    print(f"Spawn: {info['ball_xy']} -> Goal: {env.scenario.goal}")
    print(f"Course Length: {env.scenario.path_length:.2f}m")

    v_dual = out_dir / f"terrain_{scenario_kind}_dual_close_view.mp4"
    v_overview = out_dir / f"terrain_{scenario_kind}_stationary_overview.mp4"

    w_dual = imageio.get_writer(str(v_dual), fps=24, codec="libx264")
    w_over = imageio.get_writer(str(v_overview), fps=24, codec="libx264")

    ctrl = env.cfg.controller
    step = 0
    frames_dual = []

    while True:
        ball_pos = env.data.qpos[0:3]
        ball_vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]

        forces = None
        if hasattr(env, "get_rod_contact_forces"):
            forces = env.get_rod_contact_forces()

        # Compute rotation matrix
        w, x, y, z = quat
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        dirs_world = env.dirs_body @ R.T

        cross_track_y = float(ball_pos[1])
        target_heading_y = np.clip(-3.5 * cross_track_y, -0.6, 0.6)
        d_hat = np.array([np.sqrt(max(1.0 - target_heading_y**2, 0.1)), target_heading_y])
        d_hat /= np.linalg.norm(d_hat)

        u_long = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
        u_lat = dirs_world[:, 0] * (-d_hat[1]) + dirs_world[:, 1] * d_hat[0]
        u_z = dirs_world[:, 2]

        rear_factor = np.clip((-u_long - 0.10) / 0.90, 0.0, 1.0)
        down_factor = np.clip(1.0 - abs(u_z + 0.35) / 0.85, 0.0, 1.0)
        gain = 3.2 if scenario_kind in ["slopes", "stairs", "extreme_gauntlet"] else 2.8
        wave = (rear_factor ** 1.1) * down_factor * gain
        wave = np.clip(wave * np.clip(1.0 - 1.8 * (u_lat ** 2), 0.0, 1.0), 0.0, 1.0)

        # Low-profile underbelly support
        is_underbelly = (u_z < -0.30) & (u_long <= -0.05)
        depth_frac = np.clip((-u_z - 0.30) / 0.70, 0.0, 1.0)
        support_stance = depth_frac * 0.25 * np.clip(1.0 - 1.5 * (u_lat ** 2), 0.0, 1.0)
        wave = np.where(is_underbelly, np.maximum(wave, support_stance), wave)

        # Strictly retract forward & top rods
        wave[u_long > -0.05] = 0.0
        wave[u_z > 0.10] = 0.0

        targets = env.max_extend * wave

        obs, rew, term, trunc, info = env.step(targets)
        step += 1

        f_dual = env.render(camera_name="fixed_close_dual")
        f_over = env.render(camera_name="fixed_corner_sw_30deg")

        w_dual.append_data(f_dual)
        w_over.append_data(f_over)
        frames_dual.append(f_dual)

        if step % 50 == 0:
            print(f"  Step {step:4d}: Ball pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.2f}), Dist to Goal={info['distance']:.2f}m")

        if term or trunc or info["distance"] < 0.45:
            print(f"🎉 Goal reached at step {step}! Final dist={info['distance']:.3f}m")
            break

        if step >= 1500:
            break

    w_dual.close()
    w_over.close()
    env.close()

    duration = step / 24.0
    print(f"Completed {scenario_kind}: Duration {duration:.1f}s ({step} frames)")
    print(f"  - Dual Close Video: {v_dual}")
    print(f"  - Overview Video: {v_overview}")

    # Extract preview still
    if len(frames_dual) > 20:
        mid = len(frames_dual) // 2
        p_img = Path(f"docs/project_journey/assets/terrain_{scenario_kind}_preview.png")
        Image.fromarray(frames_dual[mid]).save(p_img)
        print(f"Saved preview still -> {p_img}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "slopes", "stairs", "glass_pipe", "extreme_gauntlet"])
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__extreme_terrains_showcase")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Extreme Terrains Showcase Output Directory -> {out_dir} ===")

    scenarios = ["slopes", "stairs", "glass_pipe", "extreme_gauntlet"] if args.scenario == "all" else [args.scenario]
    for sc in scenarios:
        run_terrain_simulation(sc, out_dir)


if __name__ == "__main__":
    main()
