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

def smooth_step(t, t_total):
    tau = np.clip(t / max(t_total, 1), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * tau))

def main():
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = generate_scenario("chimney", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    # Spawn high in the chimney
    start_z = 2.50
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = start_z
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print("Testing Zero-Slip Stepping Descent (Lift-off & Plant without Wall Drag)...")

    run_dir = make_run_dir(build_run_id("zero_slip_descent", "stepping"))
    out_video = Path(run_dir) / "renders" / "zero_slip_stepping_descent.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    # Rod groups:
    # Set A: Near-horizontal lateral clamp rods (Mid-tier)
    # Set B: Downward diagonal lateral rods (Lower-tier)
    # Set C: Upward diagonal lateral rods (Upper-tier)
    set_horiz_L = []
    set_horiz_R = []
    set_down_L = []
    set_down_R = []
    set_up_L = []
    set_up_R = []

    for i, d in enumerate(env.dirs_body):
        if abs(d[0]) > 0.35: continue
        # Left wall (y > 0)
        if d[1] > 0.65:
            if abs(d[2]) <= 0.30: set_horiz_L.append(i)
            elif d[2] < -0.30: set_down_L.append(i)
            elif d[2] > +0.30: set_up_L.append(i)
        # Right wall (y < 0)
        elif d[1] < -0.65:
            if abs(d[2]) <= 0.30: set_horiz_R.append(i)
            elif d[2] < -0.30: set_down_R.append(i)
            elif d[2] > +0.30: set_up_R.append(i)

    print(f"Rods: Horiz L={len(set_horiz_L)} R={len(set_horiz_R)} | Down L={len(set_down_L)} R={len(set_down_R)} | Up L={len(set_up_L)} R={len(set_up_R)}")

    # Stepping sequence:
    # 1. Stance on Horiz (HL, HR clamped at 0.15m) -> holds core statically.
    # 2. Plant Downward rods (DL, DR extend from 0.01m to 0.18m in free air until touching wall below core).
    # 3. Transfer load: DL, DR hold stance firmly. Retract HL, HR inwards to 0.01m (lifting feet off the wall!).
    # 4. Yield DL, DR smoothly from 0.18m to 0.08m: core lowers statically while DL, DR tips stay fixed on wall.
    # 5. Plant HL, HR at new lowered core height (extend 0.01m -> 0.15m through free air until touching wall).
    # 6. Transfer load back to HL, HR: HL, HR clamp firmly. Retract DL, DR to 0.01m (lifting feet off the wall!).
    # 7. Repeat! Each step lowers the core cleanly with ZERO sliding along the wall!

    substeps = [
        # Phase 1: Hold static clamp on Horiz
        {"name": "Hold Clamp (Horiz)", "steps": 30, "HL": (0.15, 0.15), "HR": (0.15, 0.15), "DL": (0.01, 0.01), "DR": (0.01, 0.01)},
        # Phase 2: Plant Downward feet lower on wall through air
        {"name": "Plant Downward Feet", "steps": 30, "HL": (0.15, 0.15), "HR": (0.15, 0.15), "DL": (0.01, 0.18), "DR": (0.01, 0.18)},
        # Phase 3: Lift Horiz feet off wall (retract to 0.01m)
        {"name": "Lift Horiz Feet Off Wall", "steps": 25, "HL": (0.15, 0.01), "HR": (0.15, 0.01), "DL": (0.18, 0.18), "DR": (0.18, 0.18)},
        # Phase 4: Core lowers as Downward rods yield
        {"name": "Lower Core on Downward Feet", "steps": 40, "HL": (0.01, 0.01), "HR": (0.01, 0.01), "DL": (0.18, 0.08), "DR": (0.18, 0.08)},
        # Phase 5: Plant Horiz feet at new lowered position
        {"name": "Plant Horiz Feet at New Level", "steps": 30, "HL": (0.01, 0.15), "HR": (0.01, 0.15), "DL": (0.08, 0.08), "DR": (0.08, 0.08)},
        # Phase 6: Lift Downward feet off wall
        {"name": "Lift Downward Feet Off Wall", "steps": 25, "HL": (0.15, 0.15), "HR": (0.15, 0.15), "DL": (0.08, 0.01), "DR": (0.08, 0.01)},
    ]

    # Repeat sequence for 8 cycles
    full_cycle = [substeps[0]] + (substeps[1:] * 8)

    step_global = 0
    max_z = start_z

    for seg in full_cycle:
        n_steps = seg["steps"]
        seg_name = seg["name"]
        hl_s, hl_e = seg["HL"]
        hr_s, hr_e = seg["HR"]
        dl_s, dl_e = seg["DL"]
        dr_s, dr_e = seg["DR"]

        for s in range(n_steps):
            w = smooth_step(s, n_steps)
            t_hl = hl_s + w * (hl_e - hl_s)
            t_hr = hr_s + w * (hr_e - hr_s)
            t_dl = dl_s + w * (dl_e - dl_s)
            t_dr = dr_s + w * (dr_e - dr_s)

            targets = np.full(60, 0.01, dtype=np.float32)
            for i in set_horiz_L: targets[i] = t_hl
            for i in set_horiz_R: targets[i] = t_hr
            for i in set_down_L: targets[i] = t_dl
            for i in set_down_R: targets[i] = t_dr

            # Check near floor for landing gear
            pos = env.data.qpos[:3].copy()
            if pos[2] < 0.60:
                targets[env.dirs_body[:, 2] < -0.30] = np.maximum(targets[env.dirs_body[:, 2] < -0.30], 0.05)

            # Planar lock
            env.data.qpos[0] = 0.0
            env.data.qvel[0] = 0.0
            env.data.qpos[3:7] = [1, 0, 0, 0]
            env.data.qvel[3:6] = [0, 0, 0]

            env.step(targets)
            pos = env.data.qpos[:3].copy()
            vel = env.data.qvel[:3].copy()

            if step_global % 4 == 0:
                if env.renderer is None:
                    env.render(camera_name="fixed_angle_close_3d")

                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = env.core_body_id
                cam.distance = 1.45
                cam.elevation = -6.0
                cam.azimuth = 180.0
                env.renderer.update_scene(env.data, camera=cam)
                frame = env.renderer.render()

                ann_frame = annotate(
                    frame,
                    "Zero-Slip Stepping Descent",
                    [
                        f"Action: {seg_name}",
                        f"Height Z: {pos[2]:.3f} m",
                        f"Velocity vz: {vel[2]:+.2f} m/s",
                        f"Feet Lifted: {'Horiz' if t_hl < 0.05 else ('Downward' if t_dl < 0.05 else 'None (Dual Stance)')}",
                    ],
                    margin=14,
                )
                writer.append_data(ann_frame)

            if step_global % 30 == 0:
                print(f"Step {step_global:4d} [{seg_name:30s}]: z={pos[2]:.3f}m | vz={vel[2]:+.2f}m/s | HL={t_hl:.2f} DL={t_dl:.2f}")

            step_global += 1

    writer.close()
    env.close()

    print(f"\nZero-Slip Descent Finished! Start Z: {start_z:.2f}m -> Final Z: {pos[2]:.3f}m")
    # Copy to artifacts
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/zero_slip_stepping_descent.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Video saved to artifact: {dst}")

if __name__ == "__main__":
    main()
