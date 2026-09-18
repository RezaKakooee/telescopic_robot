"""Export the policy-camera frames of demo episodes (HDF5) as H.264 videos with a skill HUD.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/data/export_demo_videos.py --demos <run dir> --per-course 2
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np

from radial_sphere.run_id import build_run_id
from radial_sphere.snapshot import make_run_dir
from skills_vla import SKILL_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--per-course", type=int, default=2)
    ap.add_argument("--scale", type=int, default=2, help="upscale factor for the 256 px frames")
    a = ap.parse_args()
    out = make_run_dir(build_run_id("export_demo_videos", tag=Path(a.demos).name[:40]))
    for h5_path in sorted(glob.glob(str(Path(a.demos) / "inspection_*.h5"))):
        with h5py.File(h5_path, "r") as f:
            for k in sorted(f.keys())[: a.per_course]:
                g = f[k]
                frames, skills, hit = g["frames"][:], g["skills"][:], g["hit_flags"][:]
                reasons = [r.decode() if isinstance(r, bytes) else str(r) for r in g["reasons"][:]]
                results = ([r.decode() if isinstance(r, bytes) else str(r) for r in g["results"][:]]
                           if "results" in g else None)
                vid = []
                for t in range(len(frames)):
                    img = cv2.resize(frames[t], (256 * a.scale, 256 * a.scale), interpolation=cv2.INTER_NEAREST)
                    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (12, 16, 24), -1)
                    tag = f" [{results[t]}]" if results and results[t] else ""
                    cv2.putText(img, f"t={t / 10:5.1f}s  {SKILL_NAMES[int(skills[t])]:14s} {reasons[t][:30]}{tag}", (6, 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 235, 245), 1, cv2.LINE_AA)
                    if hit[t]:
                        cv2.putText(img, "HIT", (img.shape[1] - 50, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 80), 2, cv2.LINE_AA)
                    vid.append(img)
                name = f"{Path(h5_path).stem}_{k}_seed{int(g.attrs['seed'])}_hits{int(g.attrs['hits'])}.mp4"
                imageio.mimsave(str(out / name), vid, fps=10)
                print(name, len(vid), "frames")
    print("saved to", out)


if __name__ == "__main__":
    main()
