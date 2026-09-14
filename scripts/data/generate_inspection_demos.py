"""Collect expert demonstrations on the inspection tours for VLA fine-tuning.

For every macro decision (10 Hz) the record holds:

* ``frames``        (T, 256, 256, 3) uint8   the policy camera (no chase video: that is what makes this fast)
* ``states``        (T, 13) float32          [pos xy, vel xy, waypoint dir xy, goal dir xy, goal dist, quat wxyz]
* ``skills``        (T,) int64                index into skills_vla.SKILL_NAMES
* ``params``        (T, 4) float32            [heading_ego, speed, power, 0] in [-1, 1]
* ``reasons``       (T,) str                  the oracle's one-line reason ("jump slab 3 (0.16 m)")
* ``hits``          (T,) int8                 obstacle hit in that step
* attrs             course, seed, task text, success, hits, path length

Jitter per episode: start pose, policy-camera pose, course seed. The simulation
is deterministic, so without jitter every episode of a course is identical.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/data/generate_inspection_demos.py --episodes 30 --workers 10
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np

from radial_sphere.config import load_config_cli
from radial_sphere.inspection_oracle import InspectionOracle
from radial_sphere.inspection_scenarios import INSPECTION_SCENARIOS
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from radial_sphere.snapshot import make_run_dir
from skills_vla import SKILL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generate_inspection_demos")

SKILL_TO_IDX = {n: i for i, n in enumerate(SKILL_NAMES)}

TASK_TEXT = {
    "inspection_warehouse": "Inspect the warehouse aisles. Follow the route, jump the fallen beam, keep clear of the pallets.",
    "inspection_pipe_alley": "Inspect the pipe alley. Follow the lane, hop the pipe saddles, crawl through the conduit.",
    "inspection_tank_farm": "Inspect the tank farm. Weave between the tanks, jump the bund curbs, cross the gravel.",
    "inspection_substation": "Inspect the substation bays. Cross the gravel, jump the cable trenches, avoid the posts.",
    "inspection_loading_dock": "Inspect the loading dock. Go up the ramp, pass the crates, take the stairs down, avoid the trucks.",
    "inspection_boiler_house": "Inspect the boiler house. Jump the floor pipes, climb the stairs to the deck, ramp down.",
    "inspection_utility_tunnel": "Inspect the utility tunnel. Crawl through the conduits, jump the cable cover and the manhole.",
    "inspection_solar_farm": "Inspect the solar farm. Snake through the post rows, cross the ditch, mind the mud.",
    "inspection_quarry": "Inspect the quarry road. Climb the hill, take the hairpin, descend, avoid the potholes.",
    "inspection_rubble_site": "Inspect the rubble site. Jump the beam and the crack, pass the collapsed wall.",
}

# env option name -> skills_vla class; jumps are split by station kind below
ENV_TO_VLA = {"move": "roll", "reverse": "roll", "stop": "brake_stop", "traverse_rough_terrain": "traverse_rough",
              "crawl_pipe": "crawl_pipe", "jump_forward_while_moving": "jump_forward"}
CAM_BASE = dict(distance=2.8, elevation=-32.0, azimuth=135.0)


def render_policy_frame(env, renderer, cam_pose: dict) -> np.ndarray:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.core_body_id
    cam.distance, cam.elevation, cam.azimuth = cam_pose["distance"], cam_pose["elevation"], cam_pose["azimuth"]
    renderer.update_scene(env.data, camera=cam)
    return renderer.render()


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)


def collect_episode(kind: str, cfg, seed: int, rng: np.random.Generator, max_steps: int | None = None) -> dict:
    sc = generate_scenario(kind, cfg, seed=seed, tour=True)
    env = SkillArbitrationEnv(cfg, scenario=sc, seed=seed, max_steps=8000)
    env.reset(seed=seed)
    # jitter: start pose and policy camera
    env.env.data.qpos[0:2] = np.asarray(sc.spawn_xy) + rng.uniform(-0.15, 0.15, size=2)
    env.env.data.qvel[:] = 0
    mujoco.mj_forward(env.env.model, env.env.data)
    cam_pose = {"distance": CAM_BASE["distance"] + rng.uniform(-0.3, 0.3),
                "elevation": CAM_BASE["elevation"] + rng.uniform(-6, 6),
                "azimuth": CAM_BASE["azimuth"] + rng.uniform(-15, 15)}
    cam_yaw = np.radians(cam_pose["azimuth"])           # camera forward direction in the world (for heading_ego)
    renderer = mujoco.Renderer(env.env.model, height=256, width=256)
    oracle = InspectionOracle(sc)
    goal = np.asarray(sc.goal, dtype=np.float32)[:2]
    max_steps = max_steps or int(sc.path_length * 14) + 200

    frames, states, skills, params, reasons, hits_t = [], [], [], [], [], []
    hits, info = 0, {}
    for step in range(max_steps):
        pos = env.env.data.qpos[:3].copy()
        quat = env.env.data.qpos[3:7].copy()
        vel = env.env.data.qvel[:3].copy()
        guidance = env.waypoint_tracker.get_guidance(pos)         # [wx, wy, dw, gx, gy, dg]
        name, _, why = oracle.select(pos)

        frames.append(render_policy_frame(env.env, renderer, cam_pose))
        states.append(np.concatenate([pos[:2], vel[:2], guidance[0:2], guidance[3:5], [guidance[5]], quat]).astype(np.float32))
        vla = ENV_TO_VLA[name]
        if vla == "jump_forward" and "gap" in why:
            vla = "jump_gap"
        skills.append(SKILL_TO_IDX[vla])
        heading_world = np.arctan2(guidance[1], guidance[0]) + (np.pi if name == "reverse" else 0.0)
        heading_ego = (heading_world - cam_yaw + np.pi) % (2 * np.pi) - np.pi
        speed = {"crawl_pipe": 0.6, "traverse_rough_terrain": 0.8, "stop": 0.0}.get(name, 1.1)
        params.append(np.array([heading_ego / np.pi, speed / 1.6 * 2 - 1, 0.9 * 2 - 1, 0.0], dtype=np.float32))
        reasons.append(why)

        act = np.full(len(env.skill_names), -1.0, dtype=np.float32)
        act[env.skill_names.index(name)] = 1.0
        _, _, term, trunc, info = env.step(act)
        h = int(info.get("obstacle_hit", 0))
        hits += h
        hits_t.append(h)
        if term or trunc:
            break
    env.close()
    return {
        "course": kind, "seed": seed, "success": bool(info.get("success", False)), "hits": hits,
        "steps": len(frames), "path_length": float(sc.path_length), "stalled": bool(info.get("stalled", False)),
        "frames": np.asarray(frames, dtype=np.uint8), "states": np.asarray(states, dtype=np.float32),
        "skills": np.asarray(skills, dtype=np.int64), "params": np.asarray(params, dtype=np.float32),
        "reasons": reasons, "hit_flags": np.asarray(hits_t, dtype=np.int8), "camera": cam_pose,
    }


def worker(args):
    kind, config_name, seeds, out_dir = args
    cfg = load_config_cli(name=config_name)
    out = Path(out_dir) / f"{kind}.h5"
    meta = []
    with h5py.File(out, "w") as h5:
        kept = 0
        for seed in seeds:
            rng = np.random.default_rng(seed)
            t0 = time.time()
            ep = collect_episode(kind, cfg, seed, rng)
            ok = ep["success"]
            log.info(f"{kind} seed {seed}: success={ok} hits={ep['hits']} steps={ep['steps']} ({time.time() - t0:.0f}s)")
            meta.append({k: ep[k] for k in ("course", "seed", "success", "hits", "steps", "path_length", "stalled")})
            if not ok:
                continue
            g = h5.create_group(f"episode_{kept:03d}")
            g.create_dataset("frames", data=ep["frames"], compression="gzip", compression_opts=3, chunks=(16, 256, 256, 3))
            for key in ("states", "skills", "params", "hit_flags"):
                g.create_dataset(key, data=ep[key])
            g.create_dataset("reasons", data=np.array(ep["reasons"], dtype=h5py.string_dtype()))
            for k in ("course", "seed", "success", "hits", "steps", "path_length"):
                g.attrs[k] = ep[k]
            g.attrs["task"] = TASK_TEXT[kind]
            g.attrs["camera"] = json.dumps(ep["camera"])
            kept += 1
    return kind, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30, help="episodes per course")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=1000)
    ap.add_argument("--config-name", default="playground_parkour_skills")
    ap.add_argument("--courses", nargs="*", default=list(INSPECTION_SCENARIOS))
    ap.add_argument("--out", default=None, help="run dir (default: new timestamped dir under storage_local/)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else make_run_dir(build_run_id("generate_inspection_demos", tag=f"{a.episodes}ep"))
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, a.config_name, list(range(a.seed_offset, a.seed_offset + a.episodes)), str(out)) for k in a.courses]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(a.workers, len(jobs))) as pool:
        results = pool.map(worker, jobs)
    summary = {kind: meta for kind, meta in results}
    with open(out / "summary.json", "w") as f:
        json.dump({"episodes_per_course": a.episodes, "seconds": time.time() - t0, "skills": list(SKILL_NAMES),
                   "courses": summary}, f, indent=2)
    for kind, meta in results:
        ok = sum(m["success"] for m in meta)
        log.info(f"{kind:28s} kept {ok}/{len(meta)}  mean hits {np.mean([m['hits'] for m in meta if m['success']] or [0]):.1f}")
    log.info(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
