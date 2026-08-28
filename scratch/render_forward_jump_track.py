import os
os.environ["MUJOCO_GL"] = "egl"
import datetime
from pathlib import Path
import imageio
import numpy as np
from PIL import Image

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

OUTPUT_ROOT = Path("storage_local")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
EXP_DIR = OUTPUT_ROOT / f"{TIMESTAMP}__forward_jump_track_eval"
EXP_DIR.mkdir(parents=True, exist_ok=True)
print(f"=== Forward Jump Track Showcase -> {EXP_DIR} ===")


def run_forward_jump_track():
    cfg = load_config("configs/rl/standing_jump_showcase.yaml")
    scenario = generate_scenario("jump_track", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=800)
    obs, info = env.reset(seed=42)

    dirs_body = env.dirs_body
    max_extend = env.max_extend

    video_dual_path = EXP_DIR / "forward_jump_toward_camera_dual.mp4"
    video_front_path = EXP_DIR / "forward_jump_front_toward_view.mp4"
    video_lateral_path = EXP_DIR / "forward_jump_lateral_track_view.mp4"

    w_dual = imageio.get_writer(str(video_dual_path), fps=24, codec="libx264")
    w_front = imageio.get_writer(str(video_front_path), fps=24, codec="libx264")
    w_lat = imageio.get_writer(str(video_lateral_path), fps=24, codec="libx264")

    x_hist, z_hist, vx_hist, vz_hist = [], [], [], []
    frames_dual, frames_front, frames_lat = [], [], []

    # EXACT SAME proven sequence from render_forward_jump.py (the working one):
    # 1. Standing Rest (steps 1-30)
    # 2. Standing Forward Jump: Crouch step 30-50, Takeoff step 50-62
    # 3. Sprint Rollout (steps 120-175, vx ~ 2.0-2.5 m/s) using rear-pusher drive
    # 4. Running Long Hurdle Leap: Dip step 175-182, Explosive Leap step 182-194
    # 5. Touchdown Landing & Steady Stop (steps 240-320)
    total_steps = 320

    print("Executing PROVEN Forward Jump Routine on Marked Track...")

    for step in range(1, total_steps + 1):
        ball_pos = env.data.qpos[0:3]
        ball_vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]

        x_hist.append(ball_pos[0])
        z_hist.append(ball_pos[2])
        vx_hist.append(ball_vel[0])
        vz_hist.append(ball_vel[2])

        # Rotation matrix
        w, x, y, z = quat
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        dirs_world = dirs_body @ R.T

        u_long = dirs_world[:, 0]
        u_lat = dirs_world[:, 1]
        u_z = dirs_world[:, 2]

        targets = np.zeros(env.n_bars, dtype=np.float32)

        # EXACT state machine from the WORKING render_forward_jump.py
        is_jump1_crouch = (30 <= step < 50)
        is_jump1_launch = (50 <= step < 62)
        is_jump2_run = (120 <= step < 175)
        is_jump2_dip = (175 <= step < 182)
        is_jump2_launch = (182 <= step < 195)

        if is_jump1_crouch:
            # Jump 1: Deep Crouch Preload
            targets[:] = 0.00
            phase_name = "JUMP1 CROUCH"
        elif is_jump1_launch:
            # Jump 1: Directional Forward Impulse
            ground_mask = (u_z < 0.10)
            forward_bias = np.clip(1.0 - 0.85 * np.maximum(u_long, -0.3), 0.35, 1.0)
            targets[ground_mask] = max_extend * forward_bias[ground_mask]
            targets[u_long > 0.15] = 0.0
            targets[u_z > 0.15] = 0.0
            phase_name = "🚀 JUMP1 LAUNCH"
        elif is_jump2_run:
            # Running Stride (Building vx ~ 2.2 m/s) — EXACT same rear-pusher drive
            rear_pusher = (u_long < -0.10) & (u_z < 0.10)
            targets[rear_pusher] = 0.16
            targets[u_long > -0.05] = 0.0
            targets[u_z > 0.10] = 0.0
            phase_name = "FORWARD SPRINT"
        elif is_jump2_dip:
            # Pre-Leap Kinematic Dip (Compressing ground rods)
            targets[:] = 0.00
            phase_name = "PRE-LEAP DIP"
        elif is_jump2_launch:
            # Jump 2: EXPLOSIVE HIGH-SPEED FORWARD HURDLE LEAP
            ground_mask = (u_z < 0.10)
            targets[ground_mask] = max_extend
            targets[u_z > 0.15] = 0.0
            phase_name = "🚀 JUMP2 HURDLE LEAP"
        elif ball_pos[2] > 0.28:
            # Airborne Parabolic Flight (Mid-air tuck)
            targets[:] = 0.015
            phase_name = "✈️ AIRBORNE LEAP"
        elif (62 <= step < 120) or (195 <= step < 245):
            # Soft Compliant Touchdown & Rollout
            bottom_mask = (u_z < -0.20)
            targets[bottom_mask] = 0.055
            rear_pusher = (u_long < -0.15) & (u_z < 0.0)
            targets[rear_pusher] = 0.10
            phase_name = "🛬 TOUCHDOWN"
        else:
            # Stationary Standing Rest
            bottom_mask = (u_z < -0.30)
            targets[bottom_mask] = 0.045
            phase_name = "STAND"

        obs, rew, term, trunc, info = env.step(targets)

        # Render Front-Facing (Toward Camera) and Trackside Lateral views
        f_front = env.render(camera_name="jump_front_view")
        f_lat = env.render(camera_name="jump_trackside_lateral")
        f_dual = np.concatenate([f_front, f_lat], axis=1)

        w_dual.append_data(f_dual)
        w_front.append_data(f_front)
        w_lat.append_data(f_lat)

        frames_dual.append(f_dual)
        frames_front.append(f_front)
        frames_lat.append(f_lat)

        if step % 25 == 0 or phase_name.startswith("🚀"):
            print(f"Step {step:3d} [{phase_name:22s}]: pos=(x={ball_pos[0]:.3f}m, z={ball_pos[2]:.3f}m), vx={ball_vel[0]:+5.2f}m/s, vz={ball_vel[2]:+5.2f}m/s")

    w_dual.close()
    w_front.close()
    w_lat.close()
    env.close()

    x_arr = np.array(x_hist)
    z_arr = np.array(z_hist)
    vx_arr = np.array(vx_hist)
    vz_arr = np.array(vz_hist)

    # Jump 1 metrics
    j1_takeoff_x = float(x_arr[50])
    j1_peak_z = float(np.max(z_arr[50:120]))
    j1_landing_x = float(x_arr[105])
    j1_dist = j1_landing_x - j1_takeoff_x

    # Jump 2 metrics (Running Hurdle Leap)
    j2_takeoff_x = float(x_arr[182])
    j2_peak_z = float(np.max(z_arr[182:240]))
    j2_landing_x = float(x_arr[230])
    j2_dist = j2_landing_x - j2_takeoff_x

    hurdle1_idx = int(np.argmin(np.abs(x_arr - 1.45)))

    print(f"\n=======================================================")
    print(f"=== Forward Hurdle Jump Evaluation Summary ===")
    print(f"=======================================================")
    print(f"  - Jump 1 (Standing Forward Jump): Peak z = {j1_peak_z:.3f}m (+{(j1_peak_z-z_arr[10])*100:.1f}cm), Distance = {j1_dist:.2f}m")
    print(f"  - Jump 2 (High-Speed Hurdle Leap): Peak z = {j2_peak_z:.3f}m (+{(j2_peak_z-z_arr[10])*100:.1f}cm), Distance = {j2_dist:.2f}m")
    print(f"  - Z when passing hurdle x=1.45m: z = {z_arr[hurdle1_idx]:.3f}m")
    print(f"  - Max Forward Velocity:            vx = {np.max(vx_arr):+.2f} m/s")
    print(f"  - Max Vertical Velocity:           vz = {np.max(vz_arr):+.2f} m/s")
    print(f"  - Dual Video:                      {video_dual_path}")
    print(f"  - Front View Video:                {video_front_path}")
    print(f"  - Lateral View Video:              {video_lateral_path}")

    # Extract preview stills
    j1_apex_idx = 50 + int(np.argmax(z_arr[50:120]))
    j2_apex_idx = 182 + int(np.argmax(z_arr[182:240]))

    Image.fromarray(frames_dual[40]).save("docs/project_journey/assets/forward_jump_track_1_stand.png")
    Image.fromarray(frames_dual[j1_apex_idx]).save("docs/project_journey/assets/forward_jump_track_2_jump1_apex.png")
    Image.fromarray(frames_dual[160]).save("docs/project_journey/assets/forward_jump_track_3_sprint.png")
    Image.fromarray(frames_dual[j2_apex_idx]).save("docs/project_journey/assets/forward_jump_track_4_hurdle_apex.png")
    landing2_idx = min(len(frames_dual)-1, j2_apex_idx + 22)
    Image.fromarray(frames_dual[landing2_idx]).save("docs/project_journey/assets/forward_jump_track_5_landing.png")
    Image.fromarray(frames_dual[j2_apex_idx]).save("docs/project_journey/assets/forward_jump_track_preview.png")

    print(f"Saved preview stills -> docs/project_journey/assets/forward_jump_track_*.png")


if __name__ == "__main__":
    run_forward_jump_track()
