"""Evaluation and Comparison of Ground-Contacting Underbelly Stance Strategy.

Tests:
1. Baseline: Group movement / rear-only push (rods underneath core are tucked if u_long >= 0).
2. New Strategy: Ground-Contacting Underbelly Stance (downward rods directly beneath the core
   always extend to maintain solid ground/rock contact, while rear rods provide propulsion).

Measures:
- Average number of rods touching the ground underneath the core.
- Core pitch/roll tilt variance (stability indicator).
- Forward navigation across the 260-rock mountainous floor.
"""
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


def evaluate_underbelly_strategy(enable_underbelly: bool, out_dir: Path, label: str):
    print(f"\n--- Running Evaluation: {label} (enable_underbelly_contact={enable_underbelly}) ---")
    cfg = load_config("configs/rl/rocky_mountain_terrain.yaml")
    cfg.controller.enable_underbelly_contact = enable_underbelly
    cfg.controller.enable_adaptive_grouping = False
    cfg.controller.enable_gaussian_stance = False
    cfg.controller.underbelly_stance_gain = 0.50
    cfg.controller.underbelly_threshold_z = -0.20

    env = MujocoRadialSphereEnv(cfg, max_steps=400)
    obs, info = env.reset(seed=42)

    video_side_path = out_dir / f"rocky_{label}_side_profile.mp4"
    video_quad_path = out_dir / f"rocky_{label}_quad_4_views.mp4"

    w_side = imageio.get_writer(str(video_side_path), fps=24, codec="libx264")
    w_quad = imageio.get_writer(str(video_quad_path), fps=24, codec="libx264")

    ctrl = env.cfg.controller
    tilt_angles = []
    underbelly_contacts_count = []
    frames_side = []
    frames_quad = []

    for step in range(250):
        ball_pos = env.data.qpos[0:3]
        quat = env.data.qpos[3:7]

        # Measure tilt angle from world vertical (0 deg = perfectly upright)
        # quat is [w, x, y, z]
        w, x, y, z = quat
        tilt = 2.0 * np.arccos(np.clip(abs(w), 0.0, 1.0))
        tilt_angles.append(np.degrees(tilt))

        # Count active ground contact forces on downward pointing rods
        R = np.zeros((3, 3))
        # Simple rotation matrix from quat
        R[0, 0] = 1 - 2 * (y**2 + z**2)
        R[0, 1] = 2 * (x*y - z*w)
        R[0, 2] = 2 * (x*z + y*w)
        R[1, 0] = 2 * (x*y + z*w)
        R[1, 1] = 1 - 2 * (x**2 + z**2)
        R[1, 2] = 2 * (y*z - x*w)
        R[2, 0] = 2 * (x*z - y*w)
        R[2, 1] = 2 * (y*z + x*w)
        R[2, 2] = 1 - 2 * (x**2 + y**2)

        dirs_world = env.dirs_body @ R.T
        downward_mask = dirs_world[:, 2] < -0.20
        active_contacts = np.sum(downward_mask)
        underbelly_contacts_count.append(active_contacts)

        d_hat, drive = desired_direction(ball_pos[:2], env.path_pts, lookahead=float(ctrl.lookahead))
        targets = bar_targets(
            quat,
            env.dirs_body,
            env.max_extend,
            d_hat,
            drive=drive,
            min_offset=float(ctrl.base),
            back_gain=float(ctrl.back_gain),
            enable_curb_vaulting=bool(getattr(ctrl, "enable_curb_vaulting", True)),
            curb_boost_gain=float(getattr(ctrl, "curb_boost_gain", 2.6)),
            enable_underbelly_contact=enable_underbelly,
            underbelly_stance_gain=float(getattr(ctrl, "underbelly_stance_gain", 0.70)),
            underbelly_threshold_z=float(getattr(ctrl, "underbelly_threshold_z", -0.18)),
        )

        obs, rew, term, trunc, info = env.step(targets)

        if step % 2 == 0:
            frame_side = env.render(camera_name="side_right_30deg")
            frame_quad = env.render(camera_name="quad")
            w_side.append_data(frame_side)
            w_quad.append_data(frame_quad)
            frames_side.append(frame_side)
            frames_quad.append(frame_quad)

        if term or trunc:
            print(f"Goal reached at step {step + 1}! Final distance={info['distance']:.3f}m")
            break

    w_side.close()
    w_quad.close()
    env.close()

    mean_contacts = np.mean(underbelly_contacts_count)
    std_tilt = np.std(tilt_angles)

    print(f"Results for {label}:")
    print(f"  - Underbelly Downward Rods in Stance: {mean_contacts:.1f} rods active")
    print(f"  - Final Distance to Goal: {info['distance']:.3f}m")
    print(f"  - Video (Side View): {video_side_path}")
    print(f"  - Video (4-Side Quad): {video_quad_path}")

    return {
        "frames_side": frames_side,
        "frames_quad": frames_quad,
        "mean_contacts": mean_contacts,
        "final_dist": info["distance"],
        "video_side": video_side_path,
        "video_quad": video_quad_path,
    }


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__underbelly_ground_contact_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Testing Ground-Contacting Underbelly Stance -> {out_dir} ===")

    # 1. Baseline without underbelly touching
    res_baseline = evaluate_underbelly_strategy(False, out_dir, "baseline_tucked_belly")

    # 2. New Underbelly Ground Touching Strategy
    res_underbelly = evaluate_underbelly_strategy(True, out_dir, "new_underbelly_touching")

    # Extract comparison preview stills
    if len(res_underbelly["frames_side"]) > 25:
        f_base = res_baseline["frames_side"][25]
        f_under = res_underbelly["frames_side"][25]
        comparison = np.concatenate([f_base, f_under], axis=1)

        comp_path = Path("docs/project_journey/assets/underbelly_contact_comparison.png")
        Image.fromarray(comparison).save(comp_path)

        quad_path = Path("docs/project_journey/assets/underbelly_ground_touching_quad.png")
        Image.fromarray(res_underbelly["frames_quad"][25]).save(quad_path)
        print(f"\nSaved comparison images to docs/project_journey/assets/!")


if __name__ == "__main__":
    main()
