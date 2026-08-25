"""Evaluation and Video Recording Suite for Multi-Hazard Obstacle Gauntlet.

Tests & Renders:
1. Floor Hole / Ground Pit Chasm (20 cm wide, 12 cm deep) bridged by multi-bar stance.
2. Tall Wooden Timber Blocker (7.5 cm height) vaulted by high-step rear rod extension.
3. Stone Boulder Field (18 rocks) & Granular Sand Patch.
4. Close-up side-profile + dual-view video recordings.
"""
import datetime
from pathlib import Path
import numpy as np
import imageio
from PIL import Image

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.controller import bar_targets, desired_direction


def test_hazard_gauntlet():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__hazard_gauntlet_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Testing Multi-Hazard Gauntlet (Tall 7.5cm Blocker + Floor Hole) -> {out_dir} ===")

    cfg = load_config("configs/rl/obstacle_hazard_gauntlet.yaml")
    env = MujocoRadialSphereEnv(cfg, max_steps=1200)
    obs, info = env.reset(seed=42)

    video_dual_path = out_dir / "hazard_gauntlet_dual_composite.mp4"
    video_side_path = out_dir / "hazard_gauntlet_side_profile.mp4"

    w_dual = imageio.get_writer(str(video_dual_path), fps=24, codec="libx264")
    w_side = imageio.get_writer(str(video_side_path), fps=24, codec="libx264")

    print(f"Spawn: {info['ball_xy']}, Goal: {env.scenario.goal}")
    print("Hazards present in scenario:")
    if env.scenario.gaps is not None and len(env.scenario.gaps) > 0:
        g = env.scenario.gaps[0]
        print(f"  - Floor Hole / Pit: center=({g[0]:.2f}, {g[1]:.2f}), width={g[2]*2*100:.1f}cm, depth={g[4]*100:.1f}cm")
    if env.scenario.steps is not None and len(env.scenario.steps) > 0:
        s = env.scenario.steps[0]
        print(f"  - Tall Wooden Blocker: center=({s[0]:.2f}, {s[1]:.2f}), width={s[2]*2*100:.1f}cm, height={s[4]*100:.1f}cm")
    if env.scenario.stones is not None and len(env.scenario.stones) > 0:
        st = env.scenario.stones[0]
        print(f"  - Stone Field: center=({st[0]:.2f}, {st[1]:.2f}), rocks={st[4]}")
    if env.scenario.sand_patches is not None and len(env.scenario.sand_patches) > 0:
        sp = env.scenario.sand_patches[0]
        print(f"  - Sand Patch: center=({sp[0]:.2f}, {sp[1]:.2f})")

    ctrl = env.cfg.controller
    z_history = []
    x_history = []
    frames_dual = []
    frames_side = []

    hole_crossed = False
    tall_curb_vaulted = False

    print("Running episode through multi-hazard course...")
    for step in range(400):
        ball_pos = env.data.qpos[0:3]
        quat = env.data.qpos[3:7]
        x_history.append(ball_pos[0])
        z_history.append(ball_pos[2])

        # Detect milestones
        if not hole_crossed and ball_pos[0] >= 1.35:
            hole_crossed = True
            print(f"  -> Successfully bridged Floor Hole at step {step} (x={ball_pos[0]:.3f}m, z={ball_pos[2]:.3f}m)")

        if not tall_curb_vaulted and ball_pos[0] >= 2.30:
            tall_curb_vaulted = True
            print(f"  -> Successfully vaulted 7.5cm Tall Blocker at step {step} (x={ball_pos[0]:.3f}m, z={ball_pos[2]:.3f}m)")

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
            curb_boost_gain=float(getattr(ctrl, "curb_boost_gain", 2.8)),
        )

        obs, rew, term, trunc, info = env.step(targets)

        if step % 2 == 0:
            frame_side = env.render(camera_name="side_profile")
            frame_chase = env.render(camera_name="bird_chase")
            composite = np.concatenate([frame_side, frame_chase], axis=1)

            w_side.append_data(frame_side)
            w_dual.append_data(composite)
            frames_side.append(frame_side)
            frames_dual.append(composite)

        if term or trunc:
            print(f"Episode completed at step {step + 1}. Success={info.get('success', False)}")
            break

    w_side.close()
    w_dual.close()
    env.close()

    z_arr = np.array(z_history)
    print(f"\nMulti-Hazard Gauntlet Results:")
    print(f"  - Dual Video: {video_dual_path}")
    print(f"  - Side Video: {video_side_path}")
    print(f"  - Baseline z: {z_arr[0]:.4f} m")
    print(f"  - Peak Elevation (on 7.5cm timber): {np.max(z_arr):.4f} m (+{(np.max(z_arr) - z_arr[0])*100:.1f} cm)")
    print(f"  - Floor Hole Crossed: {hole_crossed}")
    print(f"  - Tall Timber Vaulted: {tall_curb_vaulted}")
    print(f"  - Goal Distance: {info['distance']:.3f} m")
    print(f"  - Success: {info.get('success', False)}")

    # Extract key stage stills
    n_f = len(frames_dual)
    if n_f > 80:
        # 1. Crossing Floor Hole
        Image.fromarray(frames_dual[28]).save("docs/project_journey/assets/hazard_1_floor_hole_dual.png")
        # 2. Vaulting Tall 7.5cm Blocker
        Image.fromarray(frames_dual[62]).save("docs/project_journey/assets/hazard_2_tall_blocker_dual.png")
        # 3. Rolling Over Stone Field & Sand
        Image.fromarray(frames_dual[95]).save("docs/project_journey/assets/hazard_3_stones_sand_dual.png")
        print("Saved hazard stage preview images in docs/project_journey/assets/!")


if __name__ == "__main__":
    test_hazard_gauntlet()
