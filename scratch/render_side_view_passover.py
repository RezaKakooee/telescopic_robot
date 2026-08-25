"""Render High-Definition Side-Profile View of the Wooden Plank Passover.

Renders:
1. Close-up lateral horizontal tracking view showing ground contacts and rear push rod extension.
2. Stationary side view focused right on the wooden curb.
3. Dual side-profile composite video.
"""
import datetime
from pathlib import Path
import numpy as np
import imageio
from PIL import Image

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.controller import bar_targets, desired_direction


def render_side_view_passover():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__wood_plank_side_view")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering Close-up Side Profile Passover -> {out_dir} ===")

    cfg = load_config("configs/rl/obstacle_wood_plank.yaml")
    env = MujocoRadialSphereEnv(cfg, max_steps=800)
    obs, info = env.reset(seed=42)

    # 1. Lateral Side Tracking Profile Video
    side_track_path = out_dir / "wood_plank_side_profile_track.mp4"
    # 2. Stationary Side Curb Video
    side_curb_path = out_dir / "wood_plank_side_curb_fixed.mp4"
    # 3. Dual Side + Overhead Chase Video
    side_dual_path = out_dir / "wood_plank_side_dual_composite.mp4"

    w_track = imageio.get_writer(str(side_track_path), fps=24, codec="libx264")
    w_curb = imageio.get_writer(str(side_curb_path), fps=24, codec="libx264")
    w_dual = imageio.get_writer(str(side_dual_path), fps=24, codec="libx264")

    ctrl = env.cfg.controller
    frames_track = []
    frames_curb = []

    print("Running episode across wooden plank with side cameras...")
    for step in range(250):
        ball_pos = env.data.qpos[0:3]
        quat = env.data.qpos[3:7]

        d_hat, drive = desired_direction(ball_pos[:2], env.path_pts, lookahead=float(ctrl.lookahead))
        targets = bar_targets(
            quat,
            env.dirs_body,
            env.max_extend,
            d_hat,
            drive=drive,
            min_offset=float(ctrl.base),
            back_gain=float(ctrl.back_gain),
            enable_gaussian_stance=bool(getattr(ctrl, "enable_gaussian_stance", False)),
            enable_curb_vaulting=bool(getattr(ctrl, "enable_curb_vaulting", True)),
            curb_boost_gain=float(getattr(ctrl, "curb_boost_gain", 2.6)),
        )

        obs, rew, term, trunc, info = env.step(targets)

        if step % 2 == 0:
            frame_side = env.render(camera_name="side_profile")
            frame_fixed = env.render(camera_name="fixed_curb_side")
            frame_chase = env.render(camera_name="bird_chase")

            composite = np.concatenate([frame_side, frame_chase], axis=1)

            w_track.append_data(frame_side)
            w_curb.append_data(frame_fixed)
            w_dual.append_data(composite)

            frames_track.append(frame_side)
            frames_curb.append(frame_fixed)

        if term or trunc:
            break

    w_track.close()
    w_curb.close()
    w_dual.close()
    env.close()

    # Save key preview stills
    if len(frames_track) > 55:
        Image.fromarray(frames_track[53]).save("docs/project_journey/assets/wood_plank_side_view_vaulting.png")
        Image.fromarray(frames_curb[53]).save("docs/project_journey/assets/wood_plank_side_fixed_vaulting.png")
        Image.fromarray(np.concatenate([frames_track[53], frames_curb[53]], axis=1)).save("docs/project_journey/assets/wood_plank_side_dual_preview.png")

    print(f"\nRender Complete:")
    print(f"  - Side Tracking Video: {side_track_path}")
    print(f"  - Side Fixed Curb Video: {side_curb_path}")
    print(f"  - Dual Composite Video: {side_dual_path}")
    print(f"  - Preview image: docs/project_journey/assets/wood_plank_side_dual_preview.png")


if __name__ == "__main__":
    render_side_view_passover()
