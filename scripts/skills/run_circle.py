"""Run and visually demonstrate the circular locomotion skill.

Executes continuous circular orbital motion (2+ complete laps), records the
trajectory, marks the live path breadcrumbs and circular trail, and renders
a high-contrast side-by-side composite video (3D Perspective View + Top-Down Overhead View).
"""
from __future__ import annotations

import argparse
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.render import VideoRecorder
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir
from skills import execute_skill
from skills.overlay import annotate



def draw_trajectory_minimap(
    traj: list[tuple[float, float]],
    target_radius: float,
    current_pos: tuple[float, float],
    size: int = 240,
    arena_span: float = 5.0,
) -> np.ndarray:
    """Draw a high-contrast overhead minimap of the circular trajectory."""
    img = Image.new("RGBA", (size, size), (15, 18, 24, 220))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    scale = (size * 0.40) / (arena_span / 2.0)

    def to_pixel(x, y):
        px = int(cx + x * scale)
        py = int(cy - y * scale)  # Invert y for screen coords
        return px, py

    # Draw grid lines & axes
    draw.line([(0, cy), (size, cy)], fill=(45, 55, 70, 180), width=1)
    draw.line([(cx, 0), (cx, size)], fill=(45, 55, 70, 180), width=1)

    # Draw Ideal Target Circle
    r_pix = int(target_radius * scale)
    draw.ellipse(
        [cx - r_pix, cy - r_pix, cx + r_pix, cy + r_pix],
        outline=(255, 200, 40, 200),
        width=2,
    )

    # Draw center hub
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 200, 40, 255))

    # Draw Trail (Historical Path)
    if len(traj) > 1:
        pts = [to_pixel(x, y) for x, y in traj]
        for idx in range(len(pts) - 1):
            alpha = int(80 + 175 * (idx / len(pts)))
            color = (30, 220, 240, alpha)
            draw.line([pts[idx], pts[idx + 1]], fill=color, width=3)

    # Draw Start Line
    sx, sy = to_pixel(target_radius, 0.0)
    draw.rectangle([sx - 3, sy - 8, sx + 3, sy + 8], fill=(255, 100, 30, 255))

    # Draw Current Robot Position
    rx, ry = to_pixel(current_pos[0], current_pos[1])
    draw.ellipse([rx - 6, ry - 6, rx + 6, ry + 6], fill=(255, 40, 40, 255), outline=(255, 255, 255, 255), width=2)

    # Title label
    draw.text((10, 8), "PATH TRAIL (TOP-DOWN)", fill=(200, 225, 255, 240))
    draw.text((10, size - 22), f"R = {target_radius:.2f}m", fill=(255, 200, 40, 240))

    return np.asarray(img.convert("RGB"))


def main():
    p = argparse.ArgumentParser(description="Demonstrate circular locomotion skill")
    p.add_argument("--config", default="configs/rl/circle_track.yaml")
    p.add_argument("--radius", type=float, default=1.8, help="Circle radius in metres")
    p.add_argument("--speed", type=float, default=1.2, help="Cruise speed in m/s")
    p.add_argument("--laps", type=float, default=2.5, help="Number of complete circular laps")
    p.add_argument("--clockwise", action="store_true", help="Drive clockwise instead of CCW")
    p.add_argument("--video", action="store_true", default=True, help="Record composite video")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--frame-every", type=int, default=4)
    args = p.parse_args()

    cfg = load_config(args.config)
    scenario = generate_scenario("circle_track", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    obs, info = env.reset(seed=42)

    run_dir = make_run_dir(build_run_id("run_circle", f"r{args.radius:.1f}_v{args.speed:.1f}"))

    out_video = Path(run_dir) / "renders" / "circle_skill_composite.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)

    import imageio
    writer = imageio.get_writer(str(out_video), fps=args.fps, codec="libx264") if args.video else None

    traj_history = []
    r_history = []
    speed_history = []

    # Calculate steps required for the requested number of laps
    circ = 2.0 * np.pi * args.radius
    est_total_time = (args.laps * circ) / max(args.speed, 0.1)
    total_steps = int(est_total_time / 0.01) + 100

    print(f"=== Running Circle Skill Demo ===")
    print(f"  - Target Radius:  {args.radius:.2f} m (Circumference = {circ:.2f} m)")
    print(f"  - Commanded Speed: {args.speed:.2f} m/s")
    print(f"  - Laps Target:    {args.laps:.1f} laps (Total distance ~ {args.laps * circ:.1f} m)")
    print(f"  - Direction:      {'Clockwise' if args.clockwise else 'Counter-Clockwise'}")
    print(f"  - Step Budget:    {total_steps} steps ({total_steps * 0.01:.1f}s)")

    accum_angle = 0.0
    prev_angle = np.arctan2(env.data.qpos[1], env.data.qpos[0])
    distance_traveled = 0.0
    prev_pos = env.data.qpos[0:2].copy()

    for step in range(1, total_steps + 1):
        pos = env.data.qpos[0:3].copy()
        vel = env.data.qvel[0:3].copy()
        quat = env.data.qpos[3:7].copy()

        # Update position tracking
        dx = np.linalg.norm(pos[0:2] - prev_pos)
        distance_traveled += float(dx)
        prev_pos = pos[0:2].copy()

        # Update angle & lap progress
        cur_angle = np.arctan2(pos[1], pos[0])
        d_th = cur_angle - prev_angle
        # Handle wrap-around
        if d_th > np.pi:
            d_th -= 2 * np.pi
        elif d_th < -np.pi:
            d_th += 2 * np.pi
        accum_angle += abs(d_th)
        prev_angle = cur_angle

        laps_done = accum_angle / (2 * np.pi)
        curr_r = float(np.linalg.norm(pos[0:2]))
        curr_spd = float(np.linalg.norm(vel[0:2]))

        traj_history.append((float(pos[0]), float(pos[1])))
        r_history.append(curr_r)
        speed_history.append(curr_spd)

        # Execute the pure circle skill
        targets = execute_skill(
            "circle", quat, env.dirs_body, env.max_extend,
            ball_xy=pos[0:2],
            center_xy=np.array([0.0, 0.0]),
            radius=args.radius,
            speed=args.speed,
            clockwise=args.clockwise,
            radial_gain=1.6,
        )
        env.step(targets)

        # Record video frames
        if writer is not None and (step % args.frame_every == 0):
            # 1. 3D Close Tracking Perspective
            f_3d = env.render(camera_name="fixed_angle_close_3d")

            # 2. Overhead Top-Down Bird's Eye View
            f_top = env.render(camera_name="bird_fixed")

            # Create side-by-side view
            h, w = f_3d.shape[:2]
            f_top_resized = cv2.resize(f_top, (w, h))

            # Render live HUD on 3D view
            r_err = curr_r - args.radius
            f_3d_annotated = annotate(
                f_3d,
                f"SKILL: circle  (Lap {laps_done:.2f} / {args.laps:.1f})",
                [
                    f"Speed:      {curr_spd:.2f} m/s (target {args.speed:.2f} m/s)",
                    f"Radius:     {curr_r:.3f} m (error {r_err * 100:+.1f} cm)",
                    f"Distance:   {distance_traveled:.1f} m / {args.laps * circ:.1f} m",
                    f"Direction:  {'Clockwise (CW)' if args.clockwise else 'Counter-Clockwise (CCW)'}",
                    f"Time:       {step * 0.01:.1f}s",
                ],
                margin=14,
            )

            # Render Top-Down view with title
            f_top_annotated = annotate(
                f_top_resized,
                "OVERHEAD ARENA VIEW",
                [
                    f"Painted Lane Radius: {args.radius:.2f}m",
                    f"Track Circumference: {circ:.2f}m",
                    f"Center Hub:          [0.0, 0.0]",
                ],
                margin=14,
            )

            # Generate dynamic minimap overlay
            minimap = draw_trajectory_minimap(
                traj_history,
                target_radius=args.radius,
                current_pos=(float(pos[0]), float(pos[1])),
                size=int(h * 0.38),
                arena_span=args.radius * 2.8,
            )
            mh, mw = minimap.shape[:2]

            # Composite side-by-side image
            composite = np.concatenate([f_3d_annotated, f_top_annotated], axis=1)

            # Paste minimap in the top-right corner of the composite
            composite[14:14+mh, composite.shape[1] - mw - 14 : composite.shape[1] - 14] = minimap

            writer.append_data(composite)

        if step % 100 == 0:
            print(f"Step {step:4d}: Lap {laps_done:.2f} | r = {curr_r:.3f}m (err {r_err*100:+.1f}cm) | v = {curr_spd:.2f}m/s | dist = {distance_traveled:.1f}m")

        if laps_done >= args.laps:
            print(f"  🏁 Reached target laps ({laps_done:.2f} >= {args.laps}) at step {step}!")
            break

    if writer is not None:
        writer.close()
    env.close()

    r_arr = np.array(r_history)
    spd_arr = np.array(speed_history)
    mean_r = float(np.mean(r_arr[50:]))
    std_r = float(np.std(r_arr[50:]))
    max_r_err = float(np.max(np.abs(r_arr[50:] - args.radius)))
    mean_spd = float(np.mean(spd_arr[50:]))

    print("\n" + "=" * 70)
    print("  CIRCULAR LOCOMOTION SKILL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  - Target Radius:          {args.radius:.3f} m")
    print(f"  - Achieved Mean Radius:   {mean_r:.3f} ± {std_r:.3f} m")
    print(f"  - Max Radius Deviation:   {max_r_err * 100:.1f} cm (Precision: {(1.0 - max_r_err/args.radius)*100:.1f}%)")
    print(f"  - Mean Speed:             {mean_spd:.2f} m/s (target {args.speed:.2f} m/s)")
    print(f"  - Total Laps Completed:   {laps_done:.2f} laps")
    print(f"  - Total Distance Driven:  {distance_traveled:.2f} m")
    print(f"  - Video Output:           {out_video}")
    print("=" * 70)

    # Save preview image
    if len(traj_history) > 0:
        preview_map = draw_trajectory_minimap(
            traj_history,
            target_radius=args.radius,
            current_pos=(traj_history[-1][0], traj_history[-1][1]),
            size=512,
            arena_span=args.radius * 2.8,
        )
        preview_path = Path("docs/project_journey/assets/circle_skill_trajectory.png")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview_map).save(preview_path)
        print(f"Saved trajectory preview -> {preview_path}")


if __name__ == "__main__":
    main()
