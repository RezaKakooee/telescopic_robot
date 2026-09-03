import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import cv2
import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill
from skills.overlay import annotate
from scripts.skills.run_curved_training_cones import draw_curved_slalom_minimap
from scripts.skills.run_training_cones import _cone_contact

ARTIFACTS_DIR = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c")

def render_curved_showcase():
    cfg = load_config("configs/rl/curved_training_cones.yaml")
    scenario = generate_scenario("curved_training_cones", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=4000)
    obs, info = env.reset(seed=42)

    cones = np.asarray(scenario.cones, dtype=float)
    n_cones = len(cones)
    goal = np.asarray(scenario.goal, dtype=float)
    path_pts = np.asarray(scenario.path_pts, dtype=float)

    video_path = ARTIFACTS_DIR / "curved_training_cones_slalom_showcase.mp4"
    writer = imageio.get_writer(str(video_path), fps=25, codec="libx264", quality=9)

    traj: list[tuple[float, float]] = []
    cone_cleared = [False] * n_cones
    cone_min_dist = [float("inf")] * n_cones
    total_cone_contacts = 0

    keyframe_shots = {}
    target_cones_to_capture = [0, 2, 4, 6, 8, 9]

    step_dt = float(env.model.opt.timestep) * int(getattr(env, "n_substeps", 10))

    if env.renderer is None:
        env.render(camera_name="fixed_angle_close_3d")

    for step in range(3500):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()
        speed_now = float(np.linalg.norm(vel[:2]))

        traj.append((float(pos[0]), float(pos[1])))

        for ci, c in enumerate(cones):
            dist = float(np.linalg.norm(pos[:2] - c[:2]))
            cone_min_dist[ci] = min(cone_min_dist[ci], dist)
            if pos[0] > c[0] + 0.2:
                cone_cleared[ci] = True

        total_cone_contacts += _cone_contact(env)

        targets = execute_skill(
            "curved_slalom",
            quat,
            env.dirs_body,
            env.max_extend,
            ball_xy=pos[:2],
            lin_vel=vel,
            cones=cones,
            speed=1.1,
            lateral_offset=0.80,
            lead_distance=0.40,
            lateral_gain=5.0,
        )

        obs, reward, terminated, truncated, info = env.step(targets)

        # 3D chase camera
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        eye = np.array([pos[0] - 3.6, pos[1] - 3.4, pos[2] + 3.2])
        look = np.array([pos[0] + 0.9, pos[1] + 0.2, 0.25])
        v = eye - look
        dist = max(float(np.linalg.norm(v)), 1e-6)
        cam.lookat[:] = look
        cam.distance = dist
        cam.elevation = float(np.degrees(np.arcsin(np.clip(-v[2] / dist, -1.0, 1.0))))
        cam.azimuth = float(np.degrees(np.arctan2(-v[1], -v[0])))

        env.renderer.update_scene(env.data, camera=cam)
        frame_3d = env.renderer.render()

        for ci in target_cones_to_capture:
            if ci not in keyframe_shots:
                c = cones[ci]
                if abs(pos[0] - c[0]) < 0.14:
                    keyframe_shots[ci] = frame_3d.copy()

        if step % 3 == 0:
            active_cone = 1
            for ci, clr in enumerate(cone_cleared):
                if not clr:
                    active_cone = ci + 1
                    break
                if ci == n_cones - 1 and clr:
                    active_cone = n_cones

            n_cleared = sum(cone_cleared)
            min_margin = min(cone_min_dist[:n_cleared]) if n_cleared > 0 else min(cone_min_dist)

            frame_annotated = np.array(annotate(
                frame_3d,
                f"Uneven Curved Cones Slalom (10 Cones)",
                [
                    f"progress       {n_cleared} / {n_cones} cones cleared",
                    f"active cone    #{active_cone} (x = {cones[active_cone-1, 0]:.1f}m, y = {cones[active_cone-1, 1]:+.1f}m)",
                    f"weave speed    {speed_now:.2f} m/s",
                    f"lateral pos    y = {pos[1]:+.2f} m  (curved corridor)",
                    f"min clearance  {min_margin:.2f} m",
                    f"cone contacts  {total_cone_contacts}",
                ],
                margin=14,
            ), copy=True)

            minimap = draw_curved_slalom_minimap(traj, cones, path_pts, (pos[0], pos[1]), (goal[0], goal[1]), size_w=400, size_h=150)
            mh, mw, _ = minimap.shape
            fh, fw, _ = frame_annotated.shape
            pad = 16
            frame_annotated[fh - mh - pad : fh - pad, fw - mw - pad : fw - pad] = minimap
            writer.append_data(frame_annotated)

        if np.linalg.norm(pos[:2] - goal[:2]) < 0.60 or pos[0] >= goal[0] + 0.3:
            break

    writer.close()
    env.close()

    print(f"Video written to {video_path}")
    print(f"Captured {len(keyframe_shots)} keyframes at cones: {list(keyframe_shots.keys())}")

    if len(keyframe_shots) >= 6:
        imgs = [keyframe_shots[ci] for ci in target_cones_to_capture]
        target_w, target_h = 480, 270
        resized = [cv2.resize(img, (target_w, target_h)) for img in imgs]
        row1 = np.hstack(resized[:3])
        row2 = np.hstack(resized[3:6])
        grid = np.vstack([row1, row2])

        grid_pil = Image.fromarray(grid)
        draw = ImageDraw.Draw(grid_pil)
        labels = [
            f"Cone 1 Apex (Curved +Y)",
            f"Cone 3 Apex (Curved -Y)",
            f"Cone 5 Apex (Uneven Spacing)",
            f"Cone 7 Apex (Curved Peak)",
            f"Cone 9 Apex (S-Curve Turn)",
            f"Cone 10 Apex & Finish Gate",
        ]
        for idx, lbl in enumerate(labels):
            rx = (idx % 3) * target_w
            ry = (idx // 3) * target_h
            draw.rectangle([rx + 8, ry + 8, rx + 270, ry + 32], fill=(15, 20, 28, 200))
            draw.text((rx + 14, ry + 12), lbl, fill=(255, 200, 50, 255))

        grid_path = ARTIFACTS_DIR / "curved_training_cones_progression_grid.png"
        grid_pil.save(str(grid_path))
        print(f"Progression grid written to {grid_path}")

if __name__ == "__main__":
    render_curved_showcase()
