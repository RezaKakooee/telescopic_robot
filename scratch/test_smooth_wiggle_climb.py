import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from pathlib import Path
import imageio
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.run_id import build_run_id
from radial_sphere.snapshot import make_run_dir
from skills.overlay import annotate

def smooth_blend(t_curr, t_total):
    tau = np.clip(t_curr / max(t_total, 1), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * tau))

def main():
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = generate_scenario("chimney", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    obs, info = env.reset(seed=42)

    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.5
    mujoco.mj_forward(env.model, env.data)

    print("Testing Smooth Wiggle Climb with Continuous Overlap Hand-off...")

    run_dir = make_run_dir(build_run_id("smooth_wiggle", "continuous_control"))
    out_video = Path(run_dir) / "renders" / "smooth_wiggle_climb.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    # Group rods
    left_down = []
    right_down = []
    left_horiz = []
    right_horiz = []

    for i, d in enumerate(env.dirs_body):
        if abs(d[0]) > 0.3:
            continue
        if d[1] > 0.5 and d[2] < -0.3:
            left_down.append(i)
        elif d[1] < -0.5 and d[2] < -0.3:
            right_down.append(i)
        elif d[1] > 0.8 and abs(d[2]) <= 0.3:
            left_horiz.append(i)
        elif d[1] < -0.8 and abs(d[2]) <= 0.3:
            right_horiz.append(i)

    print(f"Rod count - LH: {len(left_horiz)}, RH: {len(right_horiz)}, LD: {len(left_down)}, RD: {len(right_down)}")

    # Target tracker
    current_targets = np.full(60, 0.01, dtype=np.float32)
    
    # Initialize horizontal clamp
    for i in left_horiz + right_horiz:
        current_targets[i] = 0.15

    # Simulation schedule: sequence of smooth transition segments
    # Each sub-phase defines duration (steps), start setpoint dict, end setpoint dict
    segments = [
        # 1. Settle in initial horizontal clamp (50 steps)
        {
            "name": "Init Clamp Settle",
            "steps": 40,
            "H_L": (0.15, 0.15), "H_R": (0.15, 0.15),
            "D_L": (0.01, 0.01), "D_R": (0.01, 0.01)
        },
        # 2. Cycle 1 - Phase A: Engage diagonals while clamping
        {
            "name": "C1: Engage Diagonals",
            "steps": 30,
            "H_L": (0.15, 0.15), "H_R": (0.15, 0.15),
            "D_L": (0.01, 0.12), "D_R": (0.01, 0.12)
        },
        # 3. Cycle 1 - Phase B: Disengage horizontal clamp
        {
            "name": "C1: Disengage Clamp",
            "steps": 25,
            "H_L": (0.15, 0.01), "H_R": (0.15, 0.01),
            "D_L": (0.12, 0.12), "D_R": (0.12, 0.12)
        },
        # 4. Cycle 1 - Phase C: Right push + Left accommodate
        {
            "name": "C1: Push Right -> Left",
            "steps": 60,
            "H_L": (0.01, 0.01), "H_R": (0.01, 0.01),
            "D_L": (0.12, 0.08), "D_R": (0.12, 0.25)
        },
        # 5. Cycle 1 - Phase D: Re-engage horizontal clamp at new shifted stance
        {
            "name": "C1: Re-engage Clamp",
            "steps": 30,
            "H_L": (0.01, 0.12), "H_R": (0.01, 0.18),
            "D_L": (0.08, 0.08), "D_R": (0.25, 0.25)
        },
        # 6. Cycle 1 - Phase E: Disengage diagonals
        {
            "name": "C1: Relax Diagonals",
            "steps": 25,
            "H_L": (0.12, 0.12), "H_R": (0.18, 0.18),
            "D_L": (0.08, 0.01), "D_R": (0.25, 0.01)
        },
        # 7. Cycle 1 - Phase F: Re-center clamp stance
        {
            "name": "C1: Center Clamp",
            "steps": 30,
            "H_L": (0.12, 0.15), "H_R": (0.18, 0.15),
            "D_L": (0.01, 0.01), "D_R": (0.01, 0.01)
        },
        # 8. Cycle 2 - Phase A: Engage diagonals
        {
            "name": "C2: Engage Diagonals",
            "steps": 30,
            "H_L": (0.15, 0.15), "H_R": (0.15, 0.15),
            "D_L": (0.01, 0.12), "D_R": (0.01, 0.12)
        },
        # 9. Cycle 2 - Phase B: Disengage horizontal clamp
        {
            "name": "C2: Disengage Clamp",
            "steps": 25,
            "H_L": (0.15, 0.01), "H_R": (0.15, 0.01),
            "D_L": (0.12, 0.12), "D_R": (0.12, 0.12)
        },
        # 10. Cycle 2 - Phase C: Left push + Right accommodate
        {
            "name": "C2: Push Left -> Right",
            "steps": 60,
            "H_L": (0.01, 0.01), "H_R": (0.01, 0.01),
            "D_L": (0.12, 0.25), "D_R": (0.12, 0.08)
        },
        # 11. Cycle 2 - Phase D: Re-engage clamp at shifted stance
        {
            "name": "C2: Re-engage Clamp",
            "steps": 30,
            "H_L": (0.01, 0.18), "H_R": (0.01, 0.12),
            "D_L": (0.25, 0.25), "D_R": (0.08, 0.08)
        },
        # 12. Cycle 2 - Phase E: Relax diagonals
        {
            "name": "C2: Relax Diagonals",
            "steps": 25,
            "H_L": (0.18, 0.18), "H_R": (0.12, 0.12),
            "D_L": (0.25, 0.01), "D_R": (0.08, 0.01)
        },
        # 13. Cycle 2 - Phase F: Center clamp stance
        {
            "name": "C2: Center Clamp",
            "steps": 30,
            "H_L": (0.18, 0.15), "H_R": (0.12, 0.15),
            "D_L": (0.01, 0.01), "D_R": (0.01, 0.01)
        },
    ]

    # Repeat for 3 full iterations to observe net trend
    full_schedule = [segments[0]] + (segments[1:] * 3)

    step_global = 0
    max_z = 0.5
    min_z = 0.5

    for seg_idx, seg in enumerate(full_schedule):
        n_steps = seg["steps"]
        seg_name = seg["name"]

        h_l_start, h_l_end = seg["H_L"]
        h_r_start, h_r_end = seg["H_R"]
        d_l_start, d_l_end = seg["D_L"]
        d_r_start, d_r_end = seg["D_R"]

        for s in range(n_steps):
            w = smooth_blend(s, n_steps)

            # Interpolated targets
            t_hl = h_l_start + w * (h_l_end - h_l_start)
            t_hr = h_r_start + w * (h_r_end - h_r_start)
            t_dl = d_l_start + w * (d_l_end - d_l_start)
            t_dr = d_r_start + w * (d_r_end - d_r_start)

            targets = np.full(60, 0.01, dtype=np.float32)
            for i in left_horiz: targets[i] = t_hl
            for i in right_horiz: targets[i] = t_hr
            for i in left_down: targets[i] = t_dl
            for i in right_down: targets[i] = t_dr

            # Keep planar constraint (zero out X and pitch/yaw drift)
            env.data.qpos[0] = 0.0
            env.data.qvel[0] = 0.0
            env.data.qpos[3:7] = [1, 0, 0, 0]
            env.data.qvel[3:6] = [0, 0, 0]

            env.step(targets)
            pos = env.data.qpos[:3].copy()
            max_z = max(max_z, pos[2])
            min_z = min(min_z, pos[2])

            # Video recording
            if step_global % 4 == 0:
                if env.renderer is None:
                    env.render(camera_name="fixed_angle_close_3d")

                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = env.core_body_id
                cam.distance = 1.35
                cam.elevation = -5.0
                cam.azimuth = 180.0
                env.renderer.update_scene(env.data, camera=cam)
                frame = env.renderer.render()

                annotated_frame = annotate(
                    frame,
                    "Smooth Wiggle Climb (S-Curve & Overlap)",
                    [
                        f"Phase: {seg_name}",
                        f"Z Height: {pos[2]:.3f} m (Net: {pos[2]-0.5:+.3f}m)",
                        f"Y Offset: {pos[1]:.3f} m",
                        f"Targets: HL={t_hl:.2f} HR={t_hr:.2f} DL={t_dl:.2f} DR={t_dr:.2f}"
                    ],
                    margin=14,
                )
                writer.append_data(annotated_frame)

            if step_global % 30 == 0:
                print(f"Step {step_global:4d} [{seg_name:22s}]: z={pos[2]:.3f}m, y={pos[1]:.3f}m | HL={t_hl:.2f} HR={t_hr:.2f} DL={t_dl:.2f} DR={t_dr:.2f}")

            step_global += 1

    writer.close()
    env.close()

    print(f"\nFinished! Total steps: {step_global}")
    print(f"Initial Z: 0.500m | Final Z: {pos[2]:.3f}m | Max Z: {max_z:.3f}m | Min Z: {min_z:.3f}m")
    print(f"Video saved to: {out_video}")

    # Copy to artifacts directory
    artifact_dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/smooth_wiggle_climb.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(artifact_dst))
    print(f"Copied to artifact path: {artifact_dst}")

if __name__ == "__main__":
    main()
