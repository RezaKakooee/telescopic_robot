"""Print the expert's decisions on one course, step by step, with what each option did.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/vla/trace_inspection_expert.py inspection_hurdle_lane [--seed=7] [--tour] [--jitter] [--all]

``--jitter`` applies the demo collector's start jitter for that seed, so a
failure seen in ``check_inspection_expert.py`` can be replayed here.

One line per macro step that is not a plain `move` (every step with --all):
arc position, ball position, velocity, the decision and its reason, the
obstacle hit flag, and for a self-timed jump the plan the option made (the
edge shape, its distance and the firing distance) and how it ended.
"""
from __future__ import annotations

import sys

import numpy as np

from radial_sphere.config import load_config_cli
from radial_sphere.inspection_oracle import InspectionOracle
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    kind = args[0] if args else "inspection_hurdle_lane"
    seed = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--seed=")), 7)
    tour = "--tour" in sys.argv
    show_all = "--all" in sys.argv
    jitter = "--jitter" in sys.argv
    cfg = load_config_cli(name="playground_parkour_skills", overrides=[])
    sc = generate_scenario(kind, cfg, seed=seed, tour=tour)
    env = SkillArbitrationEnv(cfg, scenario=sc, seed=seed, max_steps=8000)
    env.reset(seed=seed)
    if jitter:
        rng = np.random.default_rng(seed)
        env.env.data.qpos[0:2] = np.asarray(sc.spawn_xy) + rng.uniform(-0.15, 0.15, size=2)
    oracle = InspectionOracle(sc)
    print(f"{kind} seed {seed} {'tour' if tour else 'short'}: {len(oracle.stations)} stations")
    for st in oracle.stations:
        print(f"    {st.kind:5s} s={st.s_start:6.1f}..{st.s_end:6.1f}  {st.label}")
    hits = 0
    info = {}
    for step in range(int(sc.path_length * 14)):
        p = env.env.data.qpos[:3].copy()
        v = env.env.data.qvel[:3].copy()
        name, _, why = oracle.select(p)
        s_here = oracle.progress(p)
        seen = ""
        if name.startswith("jump") and "--probe" in sys.argv:
            g = env.waypoint_tracker.get_guidance(p)[:2]
            es = env.terrain_probe.edges(g)
            seen = " | probe " + ", ".join(f"{e.kind[0]}{e.change:+.2f}@{e.dist:.2f}/{e.length if e.length < 9 else 9:.1f}{'*' if e.shape else ''}" for e in es[:4])
            seen += f" heading ({g[0]:+.2f},{g[1]:+.2f}) floor {p[2] - env.env.sphere_radius - 0.03:.2f}"
        act = np.full(len(env.skill_names), -1.0, dtype=np.float32)
        act[env.skill_names.index(name)] = 1.0
        _, _, term, trunc, info = env.step(act)
        hits += int(info.get("obstacle_hit", 0))
        q = env.env.data.qpos[:3]
        if show_all or name != "move" or info.get("obstacle_hit"):
            plan = info.get("skill_plan")
            plan_s = (f" plan={plan['shape']} edge {plan['edge_dist']:.2f} m fire at {plan['trigger']:.2f}"
                      if plan else "")
            res = info.get("skill_result")
            print(f"{step:4d} s={s_here:6.1f} ({p[0]:5.2f},{p[1]:5.2f},{p[2]:4.2f}) v={np.linalg.norm(v[:2]):.2f} "
                  f"-> ({q[0]:5.2f},{q[1]:5.2f},{q[2]:4.2f}) {name:26s} {why[:34]:34s} "
                  f"hit={int(info.get('obstacle_hit', 0))} {sorted(info.get('obstacle_hit_geoms', []))[:2]}"
                  f"{' res=' + res if res else ''}{plan_s}{seen}")
        if term or trunc:
            break
    print(f"END success={info.get('success')} stalled={info.get('stalled')} hits={hits} steps={step + 1}")


if __name__ == "__main__":
    main()
