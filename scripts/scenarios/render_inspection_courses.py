"""Render every industrial-inspection course: a bird's-eye view and a start-line view.

Writes one PNG per course plus a montage into a timestamped run dir.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/scenarios/render_inspection_courses.py
"""
from __future__ import annotations

import cv2
import imageio
import mujoco
import numpy as np

from radial_sphere.config import load_config_cli
from radial_sphere.inspection_scenarios import INSPECTION_SCENARIOS
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir


def render(env, lookat, distance, azimuth, elevation, renderer):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    renderer.update_scene(env.data, cam)
    return renderer.render()


def main():
    cfg = load_config_cli(name="playground_parkour_skills")
    out = make_run_dir(build_run_id("render_inspection_courses"))
    tiles = []
    import sys
    wanted = sys.argv[1:] or list(INSPECTION_SCENARIOS)
    for kind in wanted:
        sc = generate_scenario(kind, cfg, seed=0)
        env = MujocoRadialSphereEnv(cfg, max_steps=10, scenario=sc)
        env.reset(seed=0)
        r = mujoco.Renderer(env.model, 480, 640)
        pts = sc.path_pts
        lo, hi = pts.min(0) - 1.5, pts.max(0) + 1.5
        centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0]
        # 45 deg vertical fov: visible height 0.83 d, width 1.1 d (4:3 image); x maps to width
        span_x, span_y = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        top = render(env, centre, 1.05 * max(span_x / 1.1, span_y / 0.83), 90, -89, r)
        views = [top]
        for frac in (0.0, 0.35, 0.7):
            k = min(int(frac * (len(pts) - 1)), len(pts) - 12)
            fwd = pts[k + 10] - pts[k]
            az = float(np.degrees(np.arctan2(fwd[1], fwd[0]))) + 180.0   # camera sits behind, looks along the path
            views.append(render(env, [*pts[k], 0.3], 4.5, az, -22, r))
        for img, tag in zip(views, ("top", "start", "35%", "70%")):
            cv2.putText(img, f"{kind}  [{tag}]", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"{kind}  [{tag}]", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        pair = np.concatenate(views, axis=1)
        imageio.imwrite(str(out / f"{kind}.png"), pair)
        tiles.append(pair)
        env.close()
        print(kind, "path", round(sc.path_length, 1), "m, walls", len(sc.walls))
    montage = np.concatenate(tiles, axis=0)
    imageio.imwrite(str(out / "montage.png"), montage)
    print("saved to", out)


if __name__ == "__main__":
    main()
