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
EXP_DIR = OUTPUT_ROOT / f"{TIMESTAMP}__forward_jump_eval"
EXP_DIR.mkdir(parents=True, exist_ok=True)
print(f"=== Forward Jump Showcase Output Directory -> {EXP_DIR} ===")


def run_forward_jump_routine():
    cfg = load_config("configs/rl/standing_jump_showcase.yaml")
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=800)
    obs, info = env.reset(seed=42)

    dirs_body = env.dirs_body
    max_extend = env.max_extend

    video_dual_path = EXP_DIR / "forward_jump_dual_close_view.mp4"
    video_overview_path = EXP_DIR / "forward_jump_stationary_overview.mp4"

    w_dual = imageio.get_writer(str(video_dual_path), fps=24, codec="libx264")
    w_over = imageio.get_writer(str(video_overview_path), fps=24, codec="libx264")

    z_history = []
    x_history = []
    vx_history = []
    vz_history = []
    frames_dual = []
    frames_over = []

    # Sequence:
    # 1. Standing Rest (steps 1-30)
    # 2. Standing Forward Jump: Crouch step 30, Takeoff step 50
    # 3. Sprint Rollout (steps 120-175, vx ~ 2.0 m/s)
    # 4. Running Long Hurdle Leap: Dip step 175-182, Explosive Leap step 182-194 (Apex z ~ 0.65m, flying > 1.8m in air!)
    # 5. Touchdown Landing & Steady Stop (steps 240-300)
    total_steps = 320

    print("Executing Complete Forward Jump Routine (Standing Forward Jump + Running Long Leap)...")

    for step in range(1, total_steps + 1):
        ball_pos = env.data.qpos[0:3]
        ball_vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]

        x_history.append(ball_pos[0])
        z_history.append(ball_pos[2])
        vx_history.append(ball_vel[0])
        vz_history.append(ball_vel[2])

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

        # State evaluation
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
            # Running Stride (Building vx ~ 2.2 m/s)
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

        f_dual = env.render(camera_name="fixed_close_dual")
        f_over = env.render(camera_name="fixed_corner_sw_30deg")

        w_dual.append_data(f_dual)
        w_over.append_data(f_over)
        frames_dual.append(f_dual)
        frames_over.append(f_over)

        if step % 25 == 0 or phase_name.startswith("🚀"):
            print(f"Step {step:3d} [{phase_name:22s}]: pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, z={ball_pos[2]:.3f}m), vx={ball_vel[0]:+5.2f}m/s, vz={ball_vel[2]:+5.2f}m/s")

    w_dual.close()
    w_over.close()
    env.close()

    x_arr = np.array(x_history)
    z_arr = np.array(z_history)
    vx_arr = np.array(vx_history)
    vz_arr = np.array(vz_history)

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

    print(f"\n=======================================================")
    print(f"=== Forward Jump Evaluation Summary ===")
    print(f"=======================================================")
    print(f"  - Jump 1 (Standing Forward Jump): Peak z = {j1_peak_z:.3f}m (+{(j1_peak_z-z_arr[10])*100:.1f}cm), Distance = {j1_dist:.2f}m")
    print(f"  - Jump 2 (High-Speed Hurdle Leap): Peak z = {j2_peak_z:.3f}m (+{(j2_peak_z-z_arr[10])*100:.1f}cm), Distance = {j2_dist:.2f}m")
    print(f"  - Max Forward Velocity:            vx = {np.max(vx_arr):+.2f} m/s")
    print(f"  - Max Vertical Velocity:           vz = {np.max(vz_arr):+.2f} m/s")
    print(f"  - Dual Close Video:                {video_dual_path}")
    print(f"  - Overview Video:                  {video_overview_path}")

    # Extract preview stills
    j1_apex_idx = 50 + int(np.argmax(z_arr[50:120]))
    j2_apex_idx = 182 + int(np.argmax(z_arr[182:240]))

    Image.fromarray(frames_dual[40]).save("docs/project_journey/assets/forward_jump_1_crouch.png")
    Image.fromarray(frames_dual[j1_apex_idx]).save("docs/project_journey/assets/forward_jump_2_apex1.png")
    Image.fromarray(frames_dual[160]).save("docs/project_journey/assets/forward_jump_3_sprint.png")
    Image.fromarray(frames_dual[j2_apex_idx]).save("docs/project_journey/assets/forward_jump_4_hurdle_apex.png")
    Image.fromarray(frames_dual[min(len(frames_dual)-1, j2_apex_idx + 22)]).save("docs/project_journey/assets/forward_jump_5_landing.png")
    Image.fromarray(frames_dual[j2_apex_idx]).save("docs/project_journey/assets/forward_jump_preview.png")

    print(f"Saved preview stills -> docs/project_journey/assets/forward_jump_*.png")


if __name__ == "__main__":
    run_forward_jump_routine()
