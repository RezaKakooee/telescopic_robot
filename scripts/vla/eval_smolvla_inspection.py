"""Closed-loop evaluation of a fine-tuned SmolVLA policy on the inspection courses.

Run with the LeRobot environment:

    MUJOCO_GL=egl PYTHONPATH=. /home/storage_group/envs/lerobot/bin/python scripts/vla/eval_smolvla_inspection.py \
        --checkpoint storage_local/<run>/train/checkpoints/020000/pretrained_model [--tour] [--video] [course ...]

At every macro step (10 Hz) the policy sees the 256x256 policy camera, the
13-D state and the course instruction, and predicts a 10-D action: one-hot
skill (6) + params. The argmax skill is executed through SkillArbitrationEnv
(same options as the expert). The policy plans a chunk; ``--replan`` sets
how many actions of a chunk are executed before it looks again.

Reports goal reached, obstacle hits, clean success, and the skill agreement
with the expert oracle on the same states.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
import torch

from radial_sphere.config import load_config_cli
from radial_sphere.inspection_oracle import InspectionOracle
from radial_sphere.inspection_scenarios import INSPECTION_SCENARIOS
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from radial_sphere.snapshot import make_run_dir
from scripts.data.generate_inspection_demos import CAM_BASE, ENV_TO_VLA, TASK_TEXT, render_policy_frame
from scripts.vla.run_inspection_oracle import FPS, MiniMap, chase_frame
from skills_vla import ENV_SKILL_MAP, SKILL_NAMES

CAM_KEY = "observation.images.camera1"


def load_policy(checkpoint: str, device: str, replan: int):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.config.n_action_steps = replan
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=checkpoint,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


@torch.no_grad()
def predict(policy, pre, post, frame: np.ndarray, state: np.ndarray, task: str, device: str) -> np.ndarray:
    obs = {
        CAM_KEY: torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0),
        "observation.state": torch.from_numpy(state).float().unsqueeze(0),
        "task": [task],
    }
    obs = pre(obs)
    act = policy.select_action(obs)
    act = post(act)
    return act[0].detach().cpu().numpy()


def run_course(kind, cfg, policy, pre, post, device, out_dir, tour, video, seed=0):
    sc = generate_scenario(kind, cfg, seed=seed, tour=tour)
    env = SkillArbitrationEnv(cfg, scenario=sc, seed=seed, max_steps=8000)
    env.reset(seed=seed)
    oracle = InspectionOracle(sc)
    policy.reset()
    renderer = mujoco.Renderer(env.env.model, height=256, width=256)
    task = TASK_TEXT[kind]
    max_steps = int(sc.path_length * 14) + 200
    hits, agree, n, info, frames = 0, 0, 0, {}, []
    counts = {}
    if video:
        # Same recording as the expert videos: every control step at 25 fps,
        # chase camera, minimap, plus the policy camera as an inset.
        chase = mujoco.Renderer(env.env.model, height=480, width=640)
        minimap = MiniMap(sc)
        hud_state = {"vla": "roll", "exp": "roll", "hits": 0, "step": 0, "frame": None}
        next_t = [0.0]

        def on_control_step(e):
            now = float(e.env.data.time)
            while now + 1e-9 >= next_t[0]:
                hud = (f"{kind} | step {hud_state['step']:3d} | t {now:5.1f}s | vla: {hud_state['vla']:14s} "
                       f"expert: {hud_state['exp']:14s} | hits {hud_state['hits']}")
                img = chase_frame(e, chase, hud, minimap)
                if hud_state["frame"] is not None:
                    img[38:38 + 128, 8:8 + 128] = cv2.resize(hud_state["frame"], (128, 128))
                    cv2.rectangle(img, (8, 38), (8 + 128, 38 + 128), (80, 180, 240), 1)
                frames.append(img)
                next_t[0] += 1.0 / FPS

        env.on_control_step = on_control_step
    for step in range(max_steps):
        pos = env.env.data.qpos[:3].copy()
        quat = env.env.data.qpos[3:7].copy()
        vel = env.env.data.qvel[:3].copy()
        guidance = env.waypoint_tracker.get_guidance(pos)
        frame = render_policy_frame(env.env, renderer, CAM_BASE)
        state = np.concatenate([pos[:2], vel[:2], guidance[0:2], guidance[3:5], [guidance[5]], quat]).astype(np.float32)
        action = predict(policy, pre, post, frame, state, task, device)
        vla_skill = SKILL_NAMES[int(np.argmax(action[:6]))]
        env_skill = ENV_SKILL_MAP[vla_skill]
        # what the expert would do here, for the agreement score (it does not drive)
        exp_name, _, why = oracle.select(pos)
        exp_vla = ENV_TO_VLA[exp_name]
        if exp_vla == "jump_forward" and "gap" in why:
            exp_vla = "jump_gap"
        agree += int(exp_vla == vla_skill)
        n += 1
        counts[vla_skill] = counts.get(vla_skill, 0) + 1

        if video:
            hud_state.update(vla=vla_skill, exp=exp_vla, step=step, frame=frame)
        act = np.full(len(env.skill_names), -1.0, dtype=np.float32)
        act[env.skill_names.index(env_skill)] = 1.0
        _, _, term, trunc, info = env.step(act)
        hits += int(info.get("obstacle_hit", 0))
        if video:
            hud_state["hits"] = hits
        if term or trunc:
            break
    env.close()
    success = bool(info.get("success", False))
    if video and frames:
        imageio.mimsave(str(out_dir / f"{kind}_{'success' if success else 'fail'}_hits{hits}.mp4"), frames, fps=FPS)
    res = {"course": kind, "success": success, "hits": hits, "clean": success and hits == 0, "steps": step + 1,
           "final_dist": round(float(np.linalg.norm(env.env.data.qpos[:2] - np.asarray(sc.goal)[:2])), 2),
           "agreement": round(agree / max(n, 1), 3), "skills": counts, "stalled": bool(info.get("stalled", False))}
    print(f"{kind:28s} success={success!s:5s} hits={hits:3d} steps={step + 1:4d} dist={res['final_dist']:5.2f} "
          f"agree={res['agreement']:.2f} skills={counts}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tour", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--replan", type=int, default=5, help="actions executed per predicted chunk")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("courses", nargs="*", default=list(INSPECTION_SCENARIOS))
    a = ap.parse_args()
    cfg = load_config_cli(name="playground_parkour_skills")
    out = make_run_dir(build_run_id("eval_smolvla_inspection", tag="tour" if a.tour else "short"))
    policy, pre, post = load_policy(a.checkpoint, a.device, a.replan)
    t0 = time.time()
    results = [run_course(k, cfg, policy, pre, post, a.device, out, a.tour, a.video, a.seed) for k in a.courses]
    with open(out / "summary.json", "w") as f:
        json.dump({"checkpoint": a.checkpoint, "replan": a.replan, "seed": a.seed, "tour": a.tour, "results": results}, f, indent=2)
    n_ok = sum(r["success"] for r in results)
    n_clean = sum(r["clean"] for r in results)
    print(f"\n{n_ok}/{len(results)} reached the goal, {n_clean} clean, mean agreement "
          f"{np.mean([r['agreement'] for r in results]):.2f}, {time.time() - t0:.0f}s. Saved to {out}")


if __name__ == "__main__":
    main()
