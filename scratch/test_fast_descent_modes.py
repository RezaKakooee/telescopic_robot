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

def smooth_blend(t, t_total):
    tau = np.clip(t / max(t_total, 1), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * tau))

def run_fast_stepping_descent():
    """Mode 1: Large-Stride 3-Tier Walking Descent (Snappy, Deep Drop per Step)."""
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = generate_scenario("chimney", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    start_z = 3.50
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = start_z
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print("=== Testing Fast Large-Stride Stepping Descent ===")

    run_dir = make_run_dir(build_run_id("fast_descent", "large_stride_stepping"))
    out_video = Path(run_dir) / "renders" / "fast_large_stride_stepping.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    # Rod groups
    horiz_L, horiz_R = [], []
    down_L, down_R = [], []
    up_L, up_R = [], []

    for i, d in enumerate(env.dirs_body):
        if abs(d[0]) > 0.35: continue
        if d[1] > 0.65:
            if abs(d[2]) <= 0.30: horiz_L.append(i)
            elif d[2] < -0.30: down_L.append(i)
            elif d[2] > +0.30: up_L.append(i)
        elif d[1] < -0.65:
            if abs(d[2]) <= 0.30: horiz_R.append(i)
            elif d[2] < -0.30: down_R.append(i)
            elif d[2] > +0.30: up_R.append(i)

    # Snappy 4-phase large stride cycle:
    # Phase 1: Rapid plant Downward rods deep down (0.24m) [12 steps]
    # Phase 2: Instant release of Horizontal rods into air (0.01m) [6 steps]
    # Phase 3: Fast Yield of Downward rods (0.24m -> 0.03m), dropping core by ~18cm [16 steps]
    # Phase 4: Snap plant Horizontal rods at new low level (0.15m) [10 steps]
    # Phase 5: Retract Downward rods into air (0.01m) [6 steps]
    substeps = [
        {"name": "Plant Downward Feet Deep", "steps": 12, "HL": (0.15, 0.15), "HR": (0.15, 0.15), "DL": (0.01, 0.24), "DR": (0.01, 0.24)},
        {"name": "Lift Horiz Feet Off Wall", "steps": 6,  "HL": (0.15, 0.01), "HR": (0.15, 0.01), "DL": (0.24, 0.24), "DR": (0.24, 0.24)},
        {"name": "Rapid Yield & Drop Core",  "steps": 16, "HL": (0.01, 0.01), "HR": (0.01, 0.01), "DL": (0.24, 0.03), "DR": (0.24, 0.03)},
        {"name": "Snap Plant Horiz at Base", "steps": 10, "HL": (0.01, 0.15), "HR": (0.01, 0.15), "DL": (0.03, 0.03), "DR": (0.03, 0.03)},
        {"name": "Lift Downward Feet in Air","steps": 6,  "HL": (0.15, 0.15), "HR": (0.15, 0.15), "DL": (0.03, 0.01), "DR": (0.03, 0.01)},
    ]

    schedule = [substeps[0]] + (substeps * 20)

    step_global = 0
    landed = False

    for seg in schedule:
        if landed: break
        n_steps = seg["steps"]
        seg_name = seg["name"]
        hl_s, hl_e = seg["HL"]
        hr_s, hr_e = seg["HR"]
        dl_s, dl_e = seg["DL"]
        dr_s, dr_e = seg["DR"]

        for s in range(n_steps):
            w = smooth_blend(s, n_steps)
            t_hl = hl_s + w * (hl_e - hl_s)
            t_hr = hr_s + w * (hr_e - hr_s)
            t_dl = dl_s + w * (dl_e - dl_s)
            t_dr = dr_s + w * (dr_e - dr_s)

            targets = np.full(60, 0.01, dtype=np.float32)
            for i in horiz_L: targets[i] = t_hl
            for i in horiz_R: targets[i] = t_hr
            for i in down_L: targets[i] = t_dl
            for i in down_R: targets[i] = t_dr

            pos = env.data.qpos[:3].copy()
            vel = env.data.qvel[:3].copy()

            # Auto landing flare near floor
            if pos[2] < 0.60:
                targets[env.dirs_body[:, 2] < -0.30] = np.maximum(targets[env.dirs_body[:, 2] < -0.30], 0.05)

            if pos[2] <= 0.23:
                landed = True
                print(f"Touchdown on floor! Step {step_global}: Z={pos[2]:.3f}m | vz={vel[2]:+.2f}m/s")
                for _ in range(30):
                    targets = np.zeros(60, dtype=np.float32)
                    targets[env.dirs_body[:, 2] < -0.30] = 0.045
                    env.step(targets)
                    if step_global % 4 == 0:
                        render_frame(env, writer, "Landed & Standing", pos[2], vel[2], "Ground Feet")
                    step_global += 1
                break

            # Planar lock
            env.data.qpos[0] = 0.0
            env.data.qvel[0] = 0.0
            env.data.qpos[3:7] = [1, 0, 0, 0]
            env.data.qvel[3:6] = [0, 0, 0]

            env.step(targets)

            if step_global % 4 == 0:
                render_frame(env, writer, seg_name, pos[2], vel[2], f"HL={t_hl:.2f} DL={t_dl:.2f}")

            if step_global % 25 == 0:
                print(f"Step {step_global:4d} [{seg_name:26s}]: z={pos[2]:.3f}m | vz={vel[2]:+.2f}m/s")

            step_global += 1

    writer.close()
    env.close()

    print(f"Finished Large-Stride Stepping! Final Z: {pos[2]:.3f}m in {step_global} steps (~{step_global*0.004:.2f}s)")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/fast_large_stride_stepping.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")


def run_snap_catch_micro_hop_descent():
    """Mode 2: Controlled Snap-Release & Catch (Zero-Slip Ballistic Drop & Lock)."""
    cfg = load_config("configs/rl/chimney.yaml")
    scenario = generate_scenario("chimney", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    start_z = 3.50
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = start_z
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    print("\n=== Testing Controlled Snap-Catch Micro-Hop Descent ===")

    run_dir = make_run_dir(build_run_id("fast_descent", "snap_catch_micro_hop"))
    out_video = Path(run_dir) / "renders" / "snap_catch_micro_hop_descent.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

    horiz_rods = []
    for i, d in enumerate(env.dirs_body):
        if abs(d[0]) < 0.35 and abs(d[1]) > 0.70 and abs(d[2]) < 0.40:
            horiz_rods.append(i)

    # Initial clamp hold
    for _ in range(25):
        targets = np.full(60, 0.01, dtype=np.float32)
        targets[horiz_rods] = 0.15
        env.step(targets)

    step = 0
    state = "drop"
    timer = 0
    drop_duration = 18   # steps in free-air drop (~72ms -> drops ~25cm)
    catch_duration = 22  # steps in static clamp hold (~88ms -> locks completely)

    while step < 1200:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        z = pos[2]
        vz = vel[2]

        targets = np.full(60, 0.01, dtype=np.float32)

        # Ground detection
        if z < 0.60:
            targets[env.dirs_body[:, 2] < -0.30] = 0.05

        if z <= 0.23:
            print(f"Touchdown confirmed at step {step}! Final Z: {z:.3f}m")
            for _ in range(30):
                targets = np.zeros(60, dtype=np.float32)
                targets[env.dirs_body[:, 2] < -0.30] = 0.045
                env.step(targets)
                if step % 4 == 0:
                    render_frame(env, writer, "Landed & Standing", z, vz, "Ground Feet")
                step += 1
            break

        timer += 1

        if state == "drop":
            # Retract completely into air: ZERO wall contact, pure clean drop!
            targets[horiz_rods] = 0.01
            phase_name = "Free-Air Drop (0 Slip)"
            if timer >= drop_duration:
                state = "catch"
                timer = 0

        elif state == "catch":
            # Instant clamp to freeze velocity and hold height
            targets[horiz_rods] = 0.16
            phase_name = "Static Wall Catch & Lock"
            if timer >= catch_duration:
                state = "drop"
                timer = 0

        # Planar lock
        env.data.qpos[0] = 0.0
        env.data.qvel[0] = 0.0
        env.data.qpos[3:7] = [1, 0, 0, 0]
        env.data.qvel[3:6] = [0, 0, 0]

        env.step(targets)

        if step % 4 == 0:
            render_frame(env, writer, phase_name, z, vz, f"State={state.upper()}")

        if step % 25 == 0:
            print(f"Step {step:4d} [{phase_name:26s}]: z={z:.3f}m | vz={vz:+.2f}m/s")

        step += 1

    writer.close()
    env.close()

    print(f"Finished Snap-Catch Descent! Final Z: {z:.3f}m in {step} steps (~{step*0.004:.2f}s)")
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/snap_catch_micro_hop_descent.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Saved video to artifact: {dst}")


def render_frame(env, writer, action_name, z, vz, detail_str):
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
        "Zero-Slip Fast Chimney Descent",
        [
            f"Action: {action_name}",
            f"Height Z: {z:.3f} m",
            f"Velocity vz: {vz:+.2f} m/s",
            f"Details: {detail_str}",
        ],
        margin=14,
    )
    writer.append_data(ann_frame)


if __name__ == "__main__":
    run_fast_stepping_descent()
    run_snap_catch_micro_hop_descent()
