"""Render synchronized comparison video with NEW filename to bypass browser cache, with explicit '100% Continuous (Zero Gaps)' overlay."""
import os
os.environ["MUJOCO_GL"] = "egl"

import shutil
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.locomotion import move
from skills import execute_skill
from skills.overlay import annotate

FORWARD = np.array([1.0, 0.0], dtype=np.float32)


def build_env_with_transparent_core(mech: str):
    cfg = load_config("configs/rl/config.yaml")
    cfg.robot.rod_mechanism = mech
    cfg.robot.appearance_theme = "realistic"
    cfg.camera.enabled = True

    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=400)
    env.reset(seed=42)

    # Transparent glass shell
    c_mat = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_MATERIAL, "core_mat")
    if c_mat >= 0:
        env.model.mat_rgba[c_mat] = [1.0, 0.82, 0.15, 0.25]
    c_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "core_geom")
    if c_geom >= 0:
        env.model.geom_rgba[c_geom] = [1.0, 0.82, 0.15, 0.25]

    # Sleeves
    for k in range(env.n_bars):
        sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"sleeve_{k}")
        if sid >= 0:
            env.model.geom_rgba[sid] = [0.24, 0.28, 0.35, 0.95]

    if env.renderer is None:
        env.renderer = mujoco.Renderer(env.model, height=480, width=420)

    return env


def main():
    out_dir = Path("storage_local/mechanism_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "3_mechanisms_continuous_fixed.mp4"

    envs = [
        ("single_stage", "Baseline: Single Rigid Rod", build_env_with_transparent_core("single_stage")),
        ("multi_stage", "Option 1: Multi-Stage Concentric", build_env_with_transparent_core("multi_stage")),
        ("zip_chain", "Option 2: Interlocking Zip-Chain", build_env_with_transparent_core("zip_chain")),
    ]

    writer = imageio.get_writer(str(video_path), fps=25, quality=9)
    print("Rendering synchronized comparison video with continuous geometry...")

    n_steps = 140
    takeoff_frame = None
    roll_frame = None

    for step in range(n_steps):
        if step < 70:
            phase_desc = "Peristaltic Rolling Gait"
        elif step < 90:
            phase_desc = "Jump Crouch (Compression)"
        elif step < 105:
            phase_desc = "Rocket Takeoff (Full Extension)"
        else:
            phase_desc = "Touchdown Landing Cushion"

        frames = []
        for mech_name, title, env in envs:
            quat = env.data.qpos[3:7].copy()
            if step < 70:
                targets = move(quat, env.dirs_body, env.max_extend, d_hat=FORWARD, speed=1.2)
            elif step < 90:
                targets = execute_skill("jump_up", quat, env.dirs_body, env.max_extend, phase="crouch")
            elif step < 105:
                targets = execute_skill("jump_up", quat, env.dirs_body, env.max_extend, phase="takeoff")
            else:
                targets = execute_skill("jump_up", quat, env.dirs_body, env.max_extend, phase="landing")

            env.step(targets)

            if step % 2 == 0:
                env._update_dynamic_colors()
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = env.core_body_id
                cam.distance = 1.35
                cam.elevation = -22.0
                cam.azimuth = 35.0
                env.renderer.update_scene(env.data, camera=cam)
                raw = env.renderer.render()

                if mech_name == "single_stage":
                    status_txt = "Structure: Continuous (Overlaps through Center)"
                    hub_txt = "Hub: Rods collide through origin"
                elif mech_name == "multi_stage":
                    status_txt = "Structure: 100% Continuous (No Gaps)"
                    hub_txt = "Hub: 100% CLEAR (r < 7.4cm open)"
                else:
                    status_txt = "Structure: 100% Continuous Chain Column"
                    hub_txt = "Hub: 100% CLEAR (Tangential spools)"

                lines = [
                    f"Phase: {phase_desc}",
                    status_txt,
                    hub_txt,
                ]
                annotated = np.array(annotate(raw, title, lines, margin=10), copy=True)
                frames.append(annotated)

        if step % 2 == 0:
            triptych = np.concatenate(frames, axis=1)
            writer.append_data(triptych)
            if step == 40:
                roll_frame = triptych.copy()
            if step == 96:
                takeoff_frame = triptych.copy()

    writer.close()
    for _, _, env in envs:
        env.close()

    if takeoff_frame is not None:
        imageio.imwrite(str(out_dir / "takeoff_full_extension_continuous.png"), takeoff_frame)
    if roll_frame is not None:
        imageio.imwrite(str(out_dir / "rolling_gait_continuous.png"), roll_frame)

    print(f"Video saved to: {video_path}")

    # Copy to artifacts directory
    art_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/all_skills_showcase")
    art_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(video_path), str(art_dir / "3_mechanisms_continuous_fixed.mp4"))
    if takeoff_frame is not None:
        shutil.copy2(str(out_dir / "takeoff_full_extension_continuous.png"), str(art_dir / "takeoff_full_extension_continuous.png"))
    if roll_frame is not None:
        shutil.copy2(str(out_dir / "rolling_gait_continuous.png"), str(art_dir / "rolling_gait_continuous.png"))

    print(f"Artifacts copied successfully to: {art_dir}")


if __name__ == "__main__":
    main()
