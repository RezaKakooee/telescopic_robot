"""Run and visually demonstrate the Training Cones (Slalom Weave) skill.

Executes agile alternating weaving between a row of 10 linear training cones,
records the full trajectory, detects any cone contact, and produces video
with live HUD metrics and top-down minimap tracking.
"""
from __future__ import annotations

import argparse
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
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir
from skills import execute_skill
from skills.overlay import annotate


def draw_slalom_minimap(
    traj: list[tuple[float, float]],
    cones: np.ndarray,
    current_pos: tuple[float, float],
    goal_pos: tuple[float, float],
    size_w: int = 400,
    size_h: int = 160,
) -> np.ndarray:
    """Draw a high-contrast overhead minimap of the slalom weave course."""
    img = Image.new("RGBA", (size_w, size_h), (16, 20, 28, 230))
    draw = ImageDraw.Draw(img)

    max_x = max(float(goal_pos[0]) + 1.5, 22.0)
    min_x = -0.5
    span_x = max_x - min_x
    span_y = 4.0  # y from -2.0 to +2.0

    def to_pixel(x, y):
        px = int(20 + ((x - min_x) / span_x) * (size_w - 40))
        py = int(size_h // 2 - (y / (span_y / 2.0)) * (size_h * 0.38))
        return px, py

    # Draw centerline & lane boundaries
    cy = size_h // 2
    draw.line([(10, cy), (size_w - 10, cy)], fill=(50, 65, 85, 160), width=1)
    y_top = int(cy - (1.2 / (span_y / 2.0)) * (size_h * 0.38))
    y_bot = int(cy + (1.2 / (span_y / 2.0)) * (size_h * 0.38))
    draw.line([(10, y_top), (size_w - 10, y_top)], fill=(40, 50, 65, 120), width=1)
    draw.line([(10, y_bot), (size_w - 10, y_bot)], fill=(40, 50, 65, 120), width=1)

    # Draw 10 Cones
    for i, c in enumerate(cones):
        cx_pix, cy_pix = to_pixel(c[0], c[1])
        r_pix = 5
        # Orange cone marker with dark border
        draw.ellipse(
            [cx_pix - r_pix, cy_pix - r_pix, cx_pix + r_pix, cy_pix + r_pix],
            fill=(255, 110, 20, 255),
            outline=(255, 255, 255, 220),
            width=1,
        )
        draw.text((cx_pix - 3, cy_pix + 7), str(i + 1), fill=(200, 210, 225, 200))

    # Draw Historical Trajectory Trail
    if len(traj) > 1:
        pts = [to_pixel(x, y) for x, y in traj]
        for idx in range(len(pts) - 1):
            alpha = int(70 + 185 * (idx / len(pts)))
            color = (30, 225, 245, alpha)
            draw.line([pts[idx], pts[idx + 1]], fill=color, width=2)

    # Draw Goal Marker
    gx, gy = to_pixel(goal_pos[0], goal_pos[1])
    draw.rectangle([gx - 4, gy - 12, gx + 4, gy + 12], fill=(40, 220, 100, 255))

    # Draw Current Robot Position
    rx, ry = to_pixel(current_pos[0], current_pos[1])
    draw.ellipse([rx - 6, ry - 6, rx + 6, ry + 6], fill=(255, 40, 40, 255), outline=(255, 255, 255, 255), width=2)

    # Header label
    draw.text((12, 6), "SLALOM WEAVE OVERHEAD TRACKER", fill=(210, 230, 255, 240))

    return np.asarray(img.convert("RGB"))


def _cone_contact(env) -> int:
    """Return number of contacts with any training cone geoms."""
    contacts = 0
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        n1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        n2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        if n1.startswith("cone_") or n2.startswith("cone_"):
            contacts += 1
    return contacts


def run(
    *,
    speed: float = 1.1,
    lateral_offset: float = 0.80,
    lookahead: float = 0.40,
    lateral_gain: float = 5.0,
    seed: int = 42,
    record_video: bool = True,
    video_name: str = "training_cones_slalom",
    slowmo: int = 1,
    max_steps: int = 3500,
) -> dict:
    cfg = load_config("configs/rl/training_cones.yaml")
    scenario = generate_scenario("training_cones", cfg, seed=seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=max_steps)
    obs, info = env.reset(seed=seed)

    cones = np.asarray(scenario.cones, dtype=float)
    n_cones = len(cones)
    goal = np.asarray(scenario.goal, dtype=float)

    run_id = build_run_id("skills", "training_cones")
    run_dir = make_run_dir(run_id) if record_video else None
    out_video = (run_dir / "renders" / f"{video_name}.mp4") if record_video else None

    writer = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        fps = 25
        writer = imageio.get_writer(str(out_video), fps=fps, codec="libx264", quality=8)

    traj: list[tuple[float, float]] = []
    cone_cleared = [False] * n_cones
    cone_min_dist = [float("inf")] * n_cones
    total_cone_contacts = 0

    print(f"\n=== Training Cones Slalom Weave ===")
    print(f"  arena: {n_cones} cones along center line (spacing {cones[1,0]-cones[0,0]:.2f}m)")
    print(f"  robot: speed {speed:.2f} m/s, lateral weave offset ±{lateral_offset:.2f}m")

    step_dt = float(env.model.opt.timestep) * int(getattr(env, "n_substeps", 10))
    frame_every = 2 if slowmo > 1 else 4

    for step in range(max_steps):
        quat = env.data.qpos[3:7].copy()
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()
        speed_now = float(np.linalg.norm(vel[:2]))

        traj.append((float(pos[0]), float(pos[1])))

        # Check cone clearances
        for ci, c in enumerate(cones):
            dist = float(np.linalg.norm(pos[:2] - c[:2]))
            cone_min_dist[ci] = min(cone_min_dist[ci], dist)
            if pos[0] > c[0] + 0.2:
                cone_cleared[ci] = True

        # Check collisions
        n_contacts = _cone_contact(env)
        total_cone_contacts += n_contacts

        # Compute skill targets
        targets = execute_skill(
            "slalom",
            quat,
            env.dirs_body,
            env.max_extend,
            ball_xy=pos[:2],
            lin_vel=vel,
            cones=cones,
            speed=speed,
            lateral_offset=lateral_offset,
            lead_distance=lookahead,
            lateral_gain=lateral_gain,
        )

        obs, reward, terminated, truncated, info = env.step(targets)

        # Video frame capture
        if writer is not None and step % frame_every == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")
            # Render 3D Perspective View
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            # Smoothly track slightly behind and above the ball
            eye = np.array([pos[0] - 3.2, pos[1] - 3.0, pos[2] + 2.8])
            look = np.array([pos[0] + 0.8, 0.0, 0.25])
            v = eye - look
            dist = max(float(np.linalg.norm(v)), 1e-6)
            cam.lookat[:] = look
            cam.distance = dist
            cam.elevation = float(np.degrees(np.arcsin(np.clip(-v[2] / dist, -1.0, 1.0))))
            cam.azimuth = float(np.degrees(np.arctan2(-v[1], -v[0])))

            env.renderer.update_scene(env.data, camera=cam)
            frame_3d = env.renderer.render()

            # Active cone index
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
                f"Training Cones Slalom Weave" + (f"  ({slowmo}x slow)" if slowmo > 1 else ""),
                [
                    f"progress       {n_cleared} / {n_cones} cones cleared",
                    f"active cone    #{active_cone} (at x = {cones[active_cone-1, 0]:.1f}m)",
                    f"speed          {speed_now:.2f} m/s",
                    f"lateral pos    y = {pos[1]:+.2f} m  (offset ±{lateral_offset:.2f}m)",
                    f"min clearance  {min_margin:.2f} m",
                    f"cone contacts  {total_cone_contacts}",
                ],
                margin=14,
            ), copy=True)

            # Composite Minimap onto bottom right of frame
            minimap = draw_slalom_minimap(traj, cones, (pos[0], pos[1]), (goal[0], goal[1]), size_w=380, size_h=140)
            mh, mw, _ = minimap.shape
            fh, fw, _ = frame_annotated.shape
            # Overlay minimap with subtle border
            pad = 16
            frame_annotated[fh - mh - pad : fh - pad, fw - mw - pad : fw - pad] = minimap

            writer.append_data(frame_annotated)

        # Goal check
        if pos[0] >= goal[0] + 0.3:
            print(f"  Reached goal at x={pos[0]:.2f}m (step {step}, t={step*step_dt:.2f}s)!")
            break

    if writer is not None:
        writer.close()
    env.close()

    n_cleared = sum(cone_cleared)
    min_clearance = min(cone_min_dist)

    print("\n--- Summary ---")
    print(f"  cones cleared:     {n_cleared} of {n_cones}")
    print(f"  cone contacts:     {total_cone_contacts}")
    print(f"  min clearance:     {min_clearance:.3f} m")
    print(f"  final position:    x={pos[0]:.2f}m, y={pos[1]:.2f}m")
    if out_video:
        print(f"  video:             {out_video}")

    return {
        "n_cleared": n_cleared,
        "n_cones": n_cones,
        "contacts": total_cone_contacts,
        "min_clearance": min_clearance,
        "final_pos": (float(pos[0]), float(pos[1])),
        "traj": traj,
        "video": str(out_video) if out_video else None,
    }


def main():
    p = argparse.ArgumentParser(description="Demonstrate Training Cones (Slalom Weave) Skill")
    p.add_argument("--speed", type=float, default=1.1, help="Cruise speed in m/s")
    p.add_argument("--lateral-offset", type=float, default=0.80, help="Lateral weave amplitude in metres")
    p.add_argument("--lookahead", type=float, default=0.40, help="Lookahead distance in metres")
    p.add_argument("--lateral-gain", type=float, default=5.0, help="Lateral steering gain")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slowmo", type=int, default=1)
    p.add_argument("--video", action="store_true", default=True, help="Record composite video")
    args = p.parse_args()

    run(
        speed=args.speed,
        lateral_offset=args.lateral_offset,
        lookahead=args.lookahead,
        lateral_gain=args.lateral_gain,
        seed=args.seed,
        slowmo=args.slowmo,
        record_video=args.video,
    )


if __name__ == "__main__":
    main()
