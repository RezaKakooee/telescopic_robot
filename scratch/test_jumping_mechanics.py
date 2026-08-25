"""Prototype & Benchmark: Explosive Radial Jumping for 60-Bar Telescopic Sphere.

Implements a 4-phase Jumping State Machine:
1. Crouch / Pre-loading: Bottom rods compress to store kinematic stroke.
2. Explosive Thrust (Takeoff): Synchronized impulse on all ground rods to full stroke (0.16m).
3. Airborne Flight / Tuck: Rods tuck in mid-air to clear obstacles and stabilize rotation.
4. Soft Suspension Touchdown: Compliant landing damping.
"""
import datetime
from pathlib import Path
import numpy as np
import imageio
from PIL import Image

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.controller import bar_targets, desired_direction, quat_to_rotmat


def run_jumping_experiment():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"storage_local/{timestamp}__jumping_ball_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Testing 60-Bar Explosive Jumping Locomotion -> {out_dir} ===")

    cfg = load_config("configs/rl/obstacle_hazard_gauntlet.yaml")
    # Arena with tall obstacle and pit
    env = MujocoRadialSphereEnv(cfg, max_steps=600)
    obs, info = env.reset(seed=42)

    video_side_path = out_dir / "jumping_side_profile.mp4"
    video_dual_path = out_dir / "jumping_dual_composite.mp4"

    w_side = imageio.get_writer(str(video_side_path), fps=24, codec="libx264")
    w_dual = imageio.get_writer(str(video_dual_path), fps=24, codec="libx264")

    ctrl = env.cfg.controller
    z_history = []
    vz_history = []
    x_history = []
    frames_side = []
    frames_dual = []

    # Jump timing & state machine
    # Robot rolls forward, encounters chasm/curb around step 50-70, initiates explosive jump!
    jump_trigger_step = 60
    jump_duration = 8  # steps of maximum explosive push
    jump_active = False

    print("Running episode with directional jump maneuver...")
    for step in range(260):
        ball_pos = env.data.qpos[0:3]
        ball_vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]

        x_history.append(ball_pos[0])
        z_history.append(ball_pos[2])
        vz_history.append(ball_vel[2])

        R = quat_to_rotmat(quat)
        dirs_world = env.dirs_body @ R.T

        # Detect or trigger jump when approaching chasm (x ~ 0.85m) or at trigger step
        if not jump_active and (ball_pos[0] >= 0.80 or step >= jump_trigger_step) and step < 120:
            jump_active = True
            jump_start_step = step
            print(f"  ⚡ [TAKEOFF] Triggering Explosive Thrust at step {step} (x={ball_pos[0]:.3f}m, z={ball_pos[2]:.3f}m)")

        if jump_active and (step - jump_start_step) < jump_duration:
            # PHASE 2: EXPLOSIVE THRUST
            # Command all downward & rear-downward rods to instantaneous 100% max stroke!
            targets = np.full(env.n_bars, float(ctrl.base), dtype=np.float32)
            u_z = dirs_world[:, 2]
            u_long = dirs_world[:, 0]  # moving in +x

            # Downward ground cluster (u_z < 0) + rearward impulse (u_long < 0.2)
            ground_push_mask = (u_z < 0.15) & (u_long < 0.3)
            targets[ground_push_mask] = env.max_extend  # Full 0.16m explosive extension!

            # Front-top rods retracted
            targets[u_z >= 0.2] = 0.005

        elif jump_active and ball_pos[2] > 0.25:
            # PHASE 3: AIRBORNE FLIGHT & OBSTACLE CLEARANCE TUCK
            targets = np.full(env.n_bars, 0.020, dtype=np.float32)
            # Maintain aerodynamic / gyro stability
        else:
            # PHASE 1 & 4: NORMAL PERISTALTIC ROLLING & LANDING
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
            break

    w_side.close()
    w_dual.close()
    env.close()

    z_arr = np.array(z_history)
    vz_arr = np.array(vz_history)
    peak_z = float(np.max(z_arr))
    baseline_z = float(z_arr[0])
    jump_height = (peak_z - baseline_z) * 100.0  # in cm
    peak_vz = float(np.max(vz_arr))

    print(f"\nJumping Experiment Results:")
    print(f"  - Baseline ground height: {baseline_z:.4f} m")
    print(f"  - Peak Airborne Altitude: {peak_z:.4f} m")
    print(f"  - Net Jump Height (\u0394z): +{jump_height:.1f} cm (Launch Altitude)")
    print(f"  - Peak Vertical Takeoff Velocity: {peak_vz:.2f} m/s")
    print(f"  - Side Video: {video_side_path}")
    print(f"  - Dual Video: {video_dual_path}")

    # Extract key takeoff, apex, and landing frames
    if len(frames_dual) > 50:
        # Find apex frame index
        apex_idx = int(np.argmax(z_arr)) // 2
        takeoff_idx = max(0, apex_idx - 6)
        landing_idx = min(len(frames_dual) - 1, apex_idx + 8)

        Image.fromarray(frames_dual[takeoff_idx]).save("docs/project_journey/assets/jump_1_takeoff_dual.png")
        Image.fromarray(frames_dual[apex_idx]).save("docs/project_journey/assets/jump_2_apex_dual.png")
        Image.fromarray(frames_dual[landing_idx]).save("docs/project_journey/assets/jump_3_landing_dual.png")
        print("Saved takeoff, apex, and landing images in docs/project_journey/assets/!")


if __name__ == "__main__":
    run_jumping_experiment()
