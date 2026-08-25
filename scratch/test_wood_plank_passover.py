"""Evaluation and Video Recording Suite for Traversable Wooden Timber Plank on the Ground.

Validates:
1. Realistic rectangular wooden plank on the floor with natural timber materials and anchor brackets.
2. The 60-bar radial sphere rolling forward, encountering the 4.0cm wooden plank, climbing/passing over it.
3. Recording dual-view MP4 video showing the peristaltic rod suspension mechanics during passover.
"""
import datetime
from pathlib import Path
import numpy as np
import imageio

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.controller import bar_targets, desired_direction


def test_wood_plank_passover():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__wood_plank_passover_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Testing Wooden Plank Passover Locomotion -> {out_dir} ===")

    cfg = load_config("configs/rl/obstacle_wood_plank.yaml")
    env = MujocoRadialSphereEnv(cfg, max_steps=800)
    obs, info = env.reset(seed=42)

    video_path = out_dir / "wood_plank_passover_dual_view.mp4"
    writer = imageio.get_writer(str(video_path), fps=24, codec="libx264")

    print(f"Spawn: {info['ball_xy']}, Goal: {env.scenario.goal}")
    steps_data = getattr(env.scenario, "steps", [])
    print(f"Wooden Planks on ground: {len(steps_data)}")
    for i, p in enumerate(steps_data):
        print(f"  Plank {i}: center=({p[0]:.2f}, {p[1]:.2f}), width={p[2]*2:.2f}m, length={p[3]*2:.2f}m, height={p[4]*100:.1f}cm")

    ctrl = env.cfg.controller
    z_history = []
    x_history = []
    climbed = False
    passed_over = False

    print("Running episode across wooden plank...")
    for step in range(350):
        ball_pos = env.data.qpos[0:3]
        quat = env.data.qpos[3:7]
        x_history.append(ball_pos[0])
        z_history.append(ball_pos[2])

        # Detect climbing event
        if not climbed and ball_pos[0] >= 1.65:
            climbed = True
            print(f"  -> Reached wooden plank at step {step} (x={ball_pos[0]:.3f}m, z={ball_pos[2]:.3f}m)")

        if not passed_over and ball_pos[0] >= 2.05:
            passed_over = True
            print(f"  -> Successfully passed over wooden plank at step {step} (x={ball_pos[0]:.3f}m, z={ball_pos[2]:.3f}m)")

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
        )

        obs, rew, term, trunc, info = env.step(targets)

        if step % 2 == 0:
            frame = env.render(mode="dual_bird_chase")
            writer.append_data(frame)

        if term or trunc:
            print(f"Episode finished at step {step + 1}. Success={info.get('success', False)}")
            break

    writer.close()
    env.close()

    z_arr = np.array(z_history)
    print(f"\nPassover Results Summary:")
    print(f"  - Video saved: {video_path}")
    print(f"  - Baseline ground z: {z_arr[0]:.4f} m")
    print(f"  - Peak elevation on plank z: {np.max(z_arr):.4f} m (+{(np.max(z_arr) - z_arr[0])*100:.1f} cm)")
    print(f"  - Passed over plank: {passed_over}")
    print(f"  - Final Distance to Goal: {info['distance']:.3f} m")
    print(f"  - Goal Success: {info.get('success', False)}")


if __name__ == "__main__":
    test_wood_plank_passover()
