"""Pass rate of the expert on jittered episodes, the way the demo collector runs them.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/vla/check_inspection_expert.py [course ...] [--episodes=12] [--seed-offset=1000] [--workers=12] [--short]

Every episode uses the tour route (``--short`` for the short one), the demo
collector's start jitter, and a fresh seed. Prints one line per course:
kept / episodes, the mean hits of the kept ones, and how the failed ones
ended. No frames, no video: this is the quick regression check for the
expert and the skills it drives.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys

import numpy as np


def run_episode(job):
    kind, seed, tour = job
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from radial_sphere.config import load_config_cli
    from radial_sphere.inspection_oracle import InspectionOracle
    from radial_sphere.scenario import generate_scenario
    from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
    cfg = load_config_cli(name="playground_parkour_skills", overrides=[])
    sc = generate_scenario(kind, cfg, seed=seed, tour=tour)
    env = SkillArbitrationEnv(cfg, scenario=sc, seed=seed, max_steps=8000)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    env.env.data.qpos[0:2] = np.asarray(sc.spawn_xy) + rng.uniform(-0.15, 0.15, size=2)
    oracle = InspectionOracle(sc)
    hits, jumps, results, info = 0, 0, {}, {}
    for step in range(int(sc.path_length * 14)):
        name, _, why = oracle.select(env.env.data.qpos[:3])
        act = np.full(len(env.skill_names), -1.0, dtype=np.float32)
        act[env.skill_names.index(name)] = 1.0
        _, _, term, trunc, info = env.step(act)
        hits += int(info.get("obstacle_hit", 0))
        jumps += name.startswith("jump")
        r = info.get("skill_result")
        if r:
            results[r] = results.get(r, 0) + 1
        if term or trunc:
            break
    env.close()
    return {"course": kind, "seed": seed, "success": bool(info.get("success", False)),
            "stalled": bool(info.get("stalled", False)), "hits": hits, "jumps": jumps,
            "steps": step + 1, "s": round(oracle.progress(env.env.data.qpos[:3]), 1),
            "path": round(float(sc.path_length), 1), "results": results}


def main():
    from radial_sphere.inspection_scenarios import INSPECTION_SCENARIOS
    courses = [a for a in sys.argv[1:] if not a.startswith("--")] or list(INSPECTION_SCENARIOS)
    episodes = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--episodes=")), 12)
    offset = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--seed-offset=")), 1000)
    workers = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--workers=")), 12)
    tour = "--short" not in sys.argv
    jobs = [(k, offset + i, tour) for k in courses for i in range(episodes)]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = pool.map(run_episode, jobs)
    summary = {}
    for k in courses:
        rs = [r for r in rows if r["course"] == k]
        ok = [r for r in rs if r["success"]]
        fails = [f"seed {r['seed']}: s={r['s']}/{r['path']}{' stalled' if r['stalled'] else ''}" for r in rs if not r["success"]]
        res = {}
        for r in rs:
            for key, n in r["results"].items():
                res[key] = res.get(key, 0) + n
        summary[k] = {"kept": len(ok), "episodes": len(rs), "mean_hits": round(float(np.mean([r["hits"] for r in ok])), 2) if ok else None,
                      "mean_jumps": round(float(np.mean([r["jumps"] for r in rs])), 1), "option_results": res, "fails": fails}
        print(f"{k:30s} {len(ok):2d}/{len(rs):<2d} kept  hits {summary[k]['mean_hits']}  jumps {summary[k]['mean_jumps']}  "
              f"results {res}" + (f"  fails: {'; '.join(fails)}" if fails else ""))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
