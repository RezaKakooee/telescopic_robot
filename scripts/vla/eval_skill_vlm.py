"""Run camera/language skill selection or record PPO-labeled demonstrations.

Examples (PYTHONPATH=. MUJOCO_GL=egl):
  python scripts/vla/eval_skill_vlm.py source=teacher teacher_run=storage_local/<run>
  python scripts/vla/eval_skill_vlm.py source=vlm allow_download=true
"""
from __future__ import annotations

import json
import pickle
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from radial_sphere.config import script_config, load_config_cli
from radial_sphere.render import VideoRecorder
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from radial_sphere.vision_skill_selector import (
    LocalTransformersPredictor, VisionSkillSelector, execution_feedback,
    make_prompt, skill_action,
)


def main():
    args = script_config("eval_skill_vlm", passthrough=True)
    if args.source not in {"teacher", "vlm"}:
        raise ValueError("source must be teacher or vlm")
    if int(args.max_decisions) < 1 or int(args.max_invalid_commands) < 1:
        raise ValueError("Decision/error limits must be positive")
    if not str(args.instruction).strip():
        raise ValueError("Provide a non-empty instruction")

    teacher_run = Path(args.teacher_run) if args.teacher_run else None
    if args.source == "teacher" and teacher_run is None:
        raise ValueError("source=teacher requires teacher_run=<PPO run directory>")
    config_path = args.config
    config_name = args.config_name
    if not config_path and not config_name:
        if args.source == "teacher":
            config_path = teacher_run / "code" / "config.yaml"
        else:
            config_name = "playground_parkour_skills"
    cfg = load_config_cli(path=config_path, name=config_name, overrides=args.scenario_overrides)
    if getattr(cfg.rl, "skill_backend", "skills_rl") != "skills" or cfg.rl.action_mode != "macro":
        raise ValueError("This selector requires rl.skill_backend=skills and rl.action_mode=macro")

    predictor = teacher = norm = None
    if args.source == "vlm":
        predictor = VisionSkillSelector(LocalTransformersPredictor(
            str(args.model_id), device=str(args.device), cache_dir=str(args.cache_dir),
            allow_download=bool(args.allow_download), max_new_tokens=int(args.max_new_tokens)))
    else:
        from stable_baselines3 import PPO
        teacher = PPO.load(teacher_run / "checkpoints" / "final.zip", device="cpu")
        with (teacher_run / "vecnormalize.pkl").open("rb") as handle:
            norm = pickle.load(handle)
        norm.training = False

    output = Path(args.output_dir) if args.output_dir else Path("storage_local") / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "__skill_vlm_" + str(args.source))
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    OmegaConf.save(cfg, output / "config.yaml")
    OmegaConf.save(args, output / "selector.yaml")
    print(f"Output: {output.resolve()}", flush=True)

    env = SkillArbitrationEnv(cfg, scenario=generate_scenario(str(cfg.scenario.kind), cfg, seed=int(args.seed)),
                              max_steps=int(cfg.rl.max_steps), training=False)
    recorder = None
    started = time.monotonic()
    counts = Counter()
    invalid_commands = 0
    total_reward = 0.
    stop_reason = "decision_limit"
    try:
        if teacher is not None:
            from stable_baselines3.common.utils import check_for_correct_spaces
            check_for_correct_spaces(env, teacher.observation_space, teacher.action_space)
        obs, info = env.reset(seed=int(args.seed))
        initial_xyz = env.env.data.qpos[:3].copy().tolist()
        previous_image = previous_path = None
        if args.video:
            fps = int(cfg.video.fps)
            recorder = VideoRecorder(output / "rollout.mp4", fps=fps)
            recorder.add(env.render(camera_name=str(args.camera)))
            next_frame = float(env.env.data.time) + 1. / fps

            def record_frame(current):
                nonlocal next_frame
                now = float(current.env.data.time)
                while now + 1e-9 >= next_frame:
                    recorder.add(current.render(camera_name=str(args.camera)))
                    next_frame += 1. / fps
            env.on_control_step = record_frame

        with (output / "decisions.jsonl").open("w") as stream:
            for decision in range(int(args.max_decisions)):
                frame = env.render(camera_name=str(args.camera))
                frame_path = f"frames/decision_{decision:04d}.png"
                Image.fromarray(frame).save(output / frame_path)
                images, paths = [frame], [frame_path]
                if args.history and previous_image is not None:
                    images.insert(0, previous_image)
                    paths.insert(0, previous_path)
                feedback = execution_feedback(info)
                prompt = make_prompt(str(args.instruction), feedback, len(images))
                query_start = time.monotonic()
                error = None
                if predictor is not None:
                    selection = predictor.select(images, str(args.instruction), feedback)
                    name, response, error = selection.skill, selection.raw_response, selection.error
                else:
                    action, _ = teacher.predict(norm.normalize_obs(obs), deterministic=True)
                    name = env.skill_names[int(np.argmax(action))]
                    response = json.dumps({"skill": name})
                query_seconds = time.monotonic() - query_start
                invalid_commands += int(error is not None)
                counts[name] += 1
                before_time = float(env.env.data.time)
                obs, reward, terminated, truncated, info = env.step(skill_action(name))
                total_reward += float(reward)
                record = {
                    "decision": decision, "source": str(args.source),
                    "instruction": str(args.instruction), "images": paths,
                    "feedback": feedback, "prompt": prompt,
                    "response": response, "selected_skill": name, "parse_error": error,
                    "query_seconds": query_seconds, "simulation_time_before": before_time,
                    "simulation_time_after": float(env.env.data.time), "reward": float(reward),
                    "terminated": terminated, "truncated": truncated, "outcome": info,
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(f"{decision:03d} {name}: reward={reward:+.2f}, "
                      f"success={info['success']}, invalid={error is not None}", flush=True)
                previous_image, previous_path = frame, frame_path
                if terminated or truncated:
                    stop_reason = "terminated" if terminated else "truncated"
                    break
                if invalid_commands >= int(args.max_invalid_commands):
                    stop_reason = "invalid_command_limit"
                    break
        Image.fromarray(env.render(camera_name=str(args.camera))).save(output / "frames" / "final.png")
        summary = {
            "source": str(args.source), "model_id": str(args.model_id) if predictor else None,
            "teacher_run": str(teacher_run) if teacher else None,
            "seed": int(args.seed), "initial_xyz": initial_xyz,
            "final_xyz": env.env.data.qpos[:3].copy().tolist(),
            "instruction": str(args.instruction), "success": bool(info.get("success", False)),
            "stop_reason": stop_reason, "decisions": sum(counts.values()),
            "skill_counts": dict(counts), "invalid_commands": invalid_commands,
            "return": total_reward, "wall_seconds": time.monotonic() - started,
            "final_info": info,
            "route_guidance_in_executor": True,
            "simulation_paused_during_inference": True,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.on_control_step = None
        if recorder:
            recorder.close()
        env.close()


if __name__ == "__main__":
    main()
