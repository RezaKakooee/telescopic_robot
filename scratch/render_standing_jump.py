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
EXP_DIR = OUTPUT_ROOT / f"{TIMESTAMP}__standing_jump_eval"
EXP_DIR.mkdir(parents=True, exist_ok=True)
print(f"=== Standing Vertical Jump Showcase -> {EXP_DIR} ===")


def run_standing_jump_routine():
    cfg = load_config("configs/rl/standing_jump_showcase.yaml")
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=800)
    obs, info = env.reset(seed=42)

    dirs_body = env.dirs_body
    max_extend = env.max_extend

    video_dual_path = EXP_DIR / "standing_jump_dual_close_view.mp4"
    video_overview_path = EXP_DIR / "standing_jump_stationary_overview.mp4"

    w_dual = imageio.get_writer(str(video_dual_path), fps=24, codec="libx264")
    w_over = imageio.get_writer(str(video_overview_path), fps=24, codec="libx264")

    z_history = []
    vz_history = []
    frames_dual = []
    frames_over = []

    # Sequence of 2 standing jumps:
    # Jump 1: Crouch step 40, Takeoff step 60 (apex ~ step 90) -> Settle step 140
    # Jump 2 (Mega Jump): Crouch step 185, Takeoff step 205 (apex ~ step 245) -> Settle step 300
    total_steps = 360

    print("Executing Realistic Standing Vertical Jump Routine (2 Consecutive Jumps)...")

    takeoff_step_1 = 55
    takeoff_step_2 = 205

    for step in range(1, total_steps + 1):
        ball_pos = env.data.qpos[0:3]
        ball_vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]

        z_history.append(ball_pos[2])
        vz_history.append(ball_vel[2])

        # Rotation matrix
        w, x, y, z = quat
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        dirs_world = dirs_body @ R.T
        u_z = dirs_world[:, 2]

        targets = np.zeros(env.n_bars, dtype=np.float32)

        # State evaluation
        is_crouch_1 = (35 <= step < takeoff_step_1)
        is_takeoff_1 = (takeoff_step_1 <= step < takeoff_step_1 + 13)
        is_crouch_2 = (185 <= step < takeoff_step_2)
        is_takeoff_2 = (takeoff_step_2 <= step < takeoff_step_2 + 13)

        if is_crouch_1 or is_crouch_2:
            # PHASE 1: DEEP CROUCH PRELOAD
            # Retract bottom rods so core drops to minimum ground height
            targets[:] = 0.00
            phase_name = "CROUCH"
        elif is_takeoff_1 or is_takeoff_2:
            # PHASE 2: EXPLOSIVE VERTICAL TAKEOFF IMPULSE
            # Fire all downward-facing ground rods with 100% full stroke!
            ground_mask = (u_z < 0.10)
            targets[ground_mask] = max_extend
            targets[u_z > 0.15] = 0.0
            phase_name = "🚀 TAKEOFF"
        elif ball_pos[2] > 0.28:
            # PHASE 3: AIRBORNE APEX FLIGHT & MID-AIR TUCK
            # Hold sleek aerodynamic spherical profile in mid-air
            targets[:] = 0.015
            phase_name = "✈️ AIRBORNE"
        elif (takeoff_step_1 + 13 <= step < 130) or (takeoff_step_2 + 13 <= step < 280):
            # PHASE 4: SOFT COMPLIANT TOUCHDOWN LANDING
            bottom_mask = (u_z < -0.20)
            # Compliant landing extension to absorb ground impact smoothly
            targets[bottom_mask] = 0.055
            phase_name = "🛬 LANDING"
        else:
            # PHASE 0 & 5: STABLE STATIONARY STANDING REST
            bottom_mask = (u_z < -0.30)
            targets[bottom_mask] = 0.045
            phase_name = "STAND"

        obs, rew, term, trunc, info = env.step(targets)

        # Render dual close-up and stationary overview
        f_dual = env.render(camera_name="fixed_close_dual")
        f_over = env.render(camera_name="fixed_corner_sw_30deg")

        w_dual.append_data(f_dual)
        w_over.append_data(f_over)
        frames_dual.append(f_dual)
        frames_over.append(f_over)

        if step % 25 == 0 or phase_name.startswith("🚀"):
            print(f"Step {step:3d} [{phase_name:12s}]: z={ball_pos[2]:.3f}m, vz={ball_vel[2]:+5.2f}m/s, pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f})")

    w_dual.close()
    w_over.close()
    env.close()

    z_arr = np.array(z_history)
    vz_arr = np.array(vz_history)

    jump1_peak = float(np.max(z_arr[50:150]))
    jump2_peak = float(np.max(z_arr[200:300]))
    baseline = float(z_arr[10])

    print(f"\n=======================================================")
    print(f"=== Standing Vertical Jump Evaluation Summary ===")
    print(f"=======================================================")
    print(f"  - Standing Baseline Altitude: {baseline:.3f} m")
    print(f"  - Jump 1 Peak Altitude:       {jump1_peak:.3f} m (Net Lift: +{(jump1_peak-baseline)*100:.1f} cm)")
    print(f"  - Jump 2 Peak Altitude:       {jump2_peak:.3f} m (Net Lift: +{(jump2_peak-baseline)*100:.1f} cm)")
    print(f"  - Max Vertical Velocity:      {np.max(vz_arr):+.2f} m/s")
    print(f"  - Dual Close Video:           {video_dual_path}")
    print(f"  - Overview Video:             {video_overview_path}")

    # Extract key preview stills: Standing, Crouch, Takeoff, Airborne Apex, Landing
    stand_idx = 20
    crouch2_idx = 198
    takeoff2_idx = 208
    apex2_idx = 200 + int(np.argmax(z_arr[200:300]))
    landing2_idx = min(len(frames_dual) - 1, apex2_idx + 22)

    Image.fromarray(frames_dual[stand_idx]).save("docs/project_journey/assets/standing_jump_1_stand.png")
    Image.fromarray(frames_dual[crouch2_idx]).save("docs/project_journey/assets/standing_jump_2_crouch.png")
    Image.fromarray(frames_dual[takeoff2_idx]).save("docs/project_journey/assets/standing_jump_3_takeoff.png")
    Image.fromarray(frames_dual[apex2_idx]).save("docs/project_journey/assets/standing_jump_4_apex.png")
    Image.fromarray(frames_dual[landing2_idx]).save("docs/project_journey/assets/standing_jump_5_landing.png")
    Image.fromarray(frames_dual[apex2_idx]).save("docs/project_journey/assets/standing_jump_preview.png")

    print(f"Saved preview stills -> docs/project_journey/assets/standing_jump_*.png")


if __name__ == "__main__":
    run_standing_jump_routine()
