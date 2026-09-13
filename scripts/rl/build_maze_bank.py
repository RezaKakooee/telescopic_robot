"""Build a reproducible bank of maze tasks.

A task is two integers: the layout seed that draws the walls and the endpoint
seed that picks start and goal. `generate_scenario` is deterministic in both,
so the bank stores no geometry -- it stores the recipe, plus the metadata we
want to sort and filter on.

Layouts are split into train and held-out. The held-out mazes are never seen
during training, so the eval number afterwards measures generalisation rather
than memorisation.
"""
import json, os, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import numpy as np
from radial_sphere.config import load_config
from radial_sphere.scenario import generate_scenario

CONFIG = "configs/rl/maze_complex_blockers_random_endpoints.yaml"
N_LAYOUTS, N_ENDPOINTS, N_HELDOUT = 100, 5, 20
OUT = Path("storage_local/maze_task_bank")
#: Reachable from anywhere; the bank path is what radial_sphere.task_bank reads.
OUT.mkdir(parents=True, exist_ok=True)


def build():
    tasks, failures = [], 0
    for layout in range(N_LAYOUTS):
        for ep in range(N_ENDPOINTS):
            cfg = load_config(CONFIG)
            cfg.camera.enabled = False
            cfg.scenario.maze.layout_seed = layout
            cfg.scenario.maze.random_endpoints = True
            try:
                sc = generate_scenario("maze", cfg, seed=ep)
            except Exception as e:                     # no valid endpoint pair
                failures += 1
                continue
            start = np.asarray(sc.spawn_xy, dtype=float)[:2]
            goal = np.asarray(sc.goal, dtype=float)[:2]
            tasks.append({
                "layout_seed": layout,
                "endpoint_seed": ep,
                "route_m": round(float(sc.path_length), 2),
                "start": [round(float(start[0]), 3), round(float(start[1]), 3)],
                "goal": [round(float(goal[0]), 3), round(float(goal[1]), 3)],
                "n_blockers": int(len(np.asarray(sc.obstacles).reshape(-1, 3))),
                "split": "heldout" if layout >= N_LAYOUTS - N_HELDOUT else "train",
            })
        if (layout + 1) % 20 == 0:
            print(f"  {layout + 1}/{N_LAYOUTS} layouts, {len(tasks)} tasks", flush=True)

    bank = {
        "config": CONFIG,
        "note": "a task is (layout_seed, endpoint_seed); generate_scenario is "
                "deterministic in both, so no geometry is stored",
        "n_layouts": N_LAYOUTS,
        "n_endpoints_per_layout": N_ENDPOINTS,
        "heldout_layouts": list(range(N_LAYOUTS - N_HELDOUT, N_LAYOUTS)),
        "generation_failures": failures,
        "tasks": tasks,
    }
    path = OUT / "tasks.json"
    json.dump(bank, open(path, "w"), indent=1)

    routes = np.array([t["route_m"] for t in tasks])
    tr = [t for t in tasks if t["split"] == "train"]
    ho = [t for t in tasks if t["split"] == "heldout"]
    print(f"\nbank: {path}")
    print(f"  tasks            {len(tasks)}  ({len(tr)} train / {len(ho)} held out)")
    print(f"  layouts          {len(set(t['layout_seed'] for t in tasks))}")
    print(f"  unique start-goal{len(set((tuple(t['start']), tuple(t['goal'])) for t in tasks)):>4}")
    print(f"  route length     min {routes.min():.1f}  median {np.median(routes):.1f}  "
          f"max {routes.max():.1f} m")
    print(f"  route quartiles  {np.percentile(routes, [25, 50, 75]).round(1).tolist()}")
    print(f"  failed to build  {failures}")
    return bank


if __name__ == "__main__":
    build()
