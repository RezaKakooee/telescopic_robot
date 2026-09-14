"""Convert the inspection demo HDF5 files into a LeRobot dataset (for SmolVLA / OpenVLA style SFT).

Run with the LeRobot environment:

    MUJOCO_GL=egl /home/storage_group/envs/lerobot/bin/python scripts/data/convert_inspection_demos_to_lerobot.py \
        --demos storage_local/<run>/ --repo-id roboball/inspection_tours

Features
--------
observation.image   (3, 256, 256) uint8 video   the policy camera
observation.state   (13,) float32               [pos xy, vel xy, waypoint dir, goal dir, goal dist, quat]
action              (10,) float32               one-hot skill (6) + [heading_ego, speed, power, 0] in [-1, 1]
task                str                          the course instruction
reason              str                          the oracle's reason for the step (extra column)

A VLA predicts the 10-D action; at run time the skill is the argmax of the
first six entries and the params follow. Frames are 10 Hz (one macro step).
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SKILLS = ("roll", "jump_forward", "jump_gap", "traverse_rough", "brake_stop", "crawl_pipe")
STATE_NAMES = ["pos_x", "pos_y", "vel_x", "vel_y", "wp_dir_x", "wp_dir_y", "goal_dir_x", "goal_dir_y", "goal_dist",
               "quat_w", "quat_x", "quat_y", "quat_z"]
ACTION_NAMES = [f"skill_{s}" for s in SKILLS] + ["heading_ego", "speed", "power", "spare"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True, help="run dir with <course>.h5 files")
    ap.add_argument("--repo-id", default="roboball/inspection_tours")
    ap.add_argument("--root", default=None, help="output dir (default: <demos>/lerobot)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-episodes-per-course", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.root) if a.root else Path(a.demos) / "lerobot"
    if root.exists():
        shutil.rmtree(root)
    features = {
        "observation.image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (13,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (10,), "names": ACTION_NAMES},
        "reason": {"dtype": "string", "shape": (1,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=a.fps, features=features, root=root, robot_type="roboball",
                               use_videos=True, image_writer_threads=4)
    n_ep, n_frames = 0, 0
    for h5_path in sorted(glob.glob(str(Path(a.demos) / "inspection_*.h5"))):
        with h5py.File(h5_path, "r") as f:
            keys = sorted(f.keys())
            if a.max_episodes_per_course:
                keys = keys[: a.max_episodes_per_course]
            for k in keys:
                g = f[k]
                task = str(g.attrs["task"])
                frames, states = g["frames"][:], g["states"][:]
                skills, params = g["skills"][:], g["params"][:]
                reasons = [r.decode() if isinstance(r, bytes) else str(r) for r in g["reasons"][:]]
                for t in range(len(frames)):
                    onehot = np.zeros(6, dtype=np.float32)
                    onehot[int(skills[t])] = 1.0
                    ds.add_frame({
                        "observation.image": frames[t],
                        "observation.state": states[t].astype(np.float32),
                        "action": np.concatenate([onehot, params[t]]).astype(np.float32),
                        "reason": reasons[t],
                        "task": task,
                    })
                ds.save_episode()
                n_ep += 1
                n_frames += len(frames)
                print(f"{Path(h5_path).stem} {k}: {len(frames)} frames")
    print(f"wrote {n_ep} episodes, {n_frames} frames to {root}")


if __name__ == "__main__":
    main()
