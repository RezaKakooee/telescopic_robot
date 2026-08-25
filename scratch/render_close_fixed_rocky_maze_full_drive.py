"""Render Close-Up Fixed-Angle Videos for Full Maze Drive across Dense Rocky Corridors.

Features:
1. 'fixed_angle_close_3d': Close macro tracking camera (distance = 1.30m) with 100% CONSTANT orientation.
   Pure translation with the ball only: NEVER rotates or turns when the robot turns -> ZERO JITTER!
2. 'fixed_angle_side_close': Close low-angle lateral view (elevation = -12°) with constant orientation.
3. Active locomotion with underbelly ground-touching stance & terrain suspension.
4. Continuous active navigation from spawn across boulders and bollards to goal touchdown!
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
from radial_sphere.scenario import generate_scenario


def render_close_fixed_full_drive():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__close_fixed_rocky_maze_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering Close-Up Fixed-Angle Rocky Maze Videos -> {out_dir} ===")

    cfg = load_config("configs/rl/maze_complex_blockers_random_endpoints.yaml")
    cfg.controller.enable_underbelly_contact = True
    cfg.controller.underbelly_stance_gain = 0.42
    cfg.controller.underbelly_threshold_z = -0.20
    cfg.controller.enable_active_suspension = True
    cfg.controller.target_ride_height = 0.28
    cfg.controller.suspension_kp = 0.75
    cfg.controller.suspension_kd = 0.15
    cfg.controller.enable_curb_vaulting = True
    cfg.controller.curb_boost_gain = 2.8
    cfg.controller.back_gain = 2.2

    # Seed 1037: 12.0m corridor route with 746 rocks and 4 safety bollards
    seed = 1037
    scenario = generate_scenario("maze", cfg, seed=seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=3000)

    obs, info = env.reset(seed=seed)
    print(f"Spawn: {info['ball_xy']} -> Goal: {env.scenario.goal}")
    print(f"Course Length: {env.scenario.path_length:.2f}m")
    print(f"Total Corridor Rocks: {sum(s[4] for s in env.scenario.stones)} boulders")

    v_close_3d = out_dir / "fixed_angle_close_3d.mp4"
    v_close_side = out_dir / "fixed_angle_side_close.mp4"
    v_close_dual = out_dir / "fixed_close_dual_composite.mp4"
    v_overview = out_dir / "fixed_sw_arena_overview.mp4"

    w_3d = imageio.get_writer(str(v_close_3d), fps=24, codec="libx264")
    w_side = imageio.get_writer(str(v_close_side), fps=24, codec="libx264")
    w_dual = imageio.get_writer(str(v_close_dual), fps=24, codec="libx264")
    w_over = imageio.get_writer(str(v_overview), fps=24, codec="libx264")

    ctrl = env.cfg.controller
    step = 0
    frames_dual = []

    print("\nSimulating full-drive active navigation across rocky maze...")
    while True:
        ball_pos = env.data.qpos[0:3]
        ball_vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]

        forces = None
        if hasattr(env, "get_rod_contact_forces"):
            forces = env.get_rod_contact_forces()

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
            curb_boost_gain=float(getattr(ctrl, "curb_boost_gain", 2.8)),
            enable_underbelly_contact=True,
            underbelly_stance_gain=float(getattr(ctrl, "underbelly_stance_gain", 0.42)),
            underbelly_threshold_z=float(getattr(ctrl, "underbelly_threshold_z", -0.20)),
            enable_active_suspension=True,
            core_z=float(ball_pos[2]),
            core_vz=float(ball_vel[2]),
            target_ride_height=float(getattr(ctrl, "target_ride_height", 0.28)),
            suspension_kp=float(getattr(ctrl, "suspension_kp", 0.75)),
            suspension_kd=float(getattr(ctrl, "suspension_kd", 0.15)),
            suspension_force_compliance=float(getattr(ctrl, "suspension_force_compliance", 0.0018)),
            nominal_support_force=float(getattr(ctrl, "nominal_support_force", 10.0)),
            contact_forces=forces,
        )

        obs, rew, term, trunc, info = env.step(targets)
        step += 1

        f_3d = env.render(camera_name="fixed_angle_close_3d")
        f_side = env.render(camera_name="fixed_angle_side_close")
        f_dual = np.concatenate([f_3d, f_side], axis=1)
        f_over = env.render(camera_name="fixed_corner_sw_30deg")

        w_3d.append_data(f_3d)
        w_side.append_data(f_side)
        w_dual.append_data(f_dual)
        w_over.append_data(f_over)

        frames_dual.append(f_dual)

        if step % 50 == 0:
            print(f"  Step {step:4d}: Ball pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}), Dist to Goal={info['distance']:.2f}m")

        if term or trunc or info["distance"] < 0.45:
            print(f"\n🎉 Reached Goal at Step {step}! Final distance={info['distance']:.3f}m")
            break

        if step >= 1500:
            break

    w_3d.close()
    w_side.close()
    w_dual.close()
    w_over.close()
    env.close()

    duration = step / 24.0
    print(f"\nRender Complete! Duration: {duration:.1f}s ({step} frames)")
    print(f"  - Close 3D Fixed-Angle Video: {v_close_3d}")
    print(f"  - Close Side Low-Angle Video: {v_close_side}")
    print(f"  - Dual Close-Up Composite Video: {v_close_dual}")
    print(f"  - Static Arena Overview: {v_overview}")

    # Extract preview stills
    if len(frames_dual) > 50:
        mid = len(frames_dual) // 2
        p_dual = Path("docs/project_journey/assets/fixed_close_dual_rocky_maze_preview.png")
        Image.fromarray(frames_dual[mid]).save(p_dual)
        print(f"Saved preview still -> {p_dual}")


if __name__ == "__main__":
    render_close_fixed_full_drive()
