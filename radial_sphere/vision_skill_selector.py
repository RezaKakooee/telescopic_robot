"""Camera + instruction -> a named skill, with an optional local VLM backend.

The selector never receives the PPO observation/map or a teacher action. The
existing skill executor still uses route guidance and robot feedback internally.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .handcrafted_skill_backend import SKILL_NAMES


SKILL_DESCRIPTIONS = {
    "move": "Move toward the next waypoint for a short control interval.",
    "stop": "Brake and hold position for a short control interval.",
    "reverse": "Move away from the next waypoint for a short control interval.",
    "follow_path": "Follow the predefined route, including steering around its corners.",
    "straddle_gap": "Drive ALONG a narrow longitudinal gap supported on both sides; not across a valley.",
    "traverse_rough_terrain": "Drive with terrain/contact suspension over uneven ground.",
    "jump_up": "Execute a full vertical jump: crouch, launch, flight, landing.",
    "jump_forward_while_stopped": "Execute a full forward jump from rest toward the next waypoint.",
    "jump_forward_while_moving": "Execute a full running forward jump, including run-up; requires clear space.",
    "jump_to": "Execute a forward jump with velocity feedback, toward the next waypoint.",
    "crawl_pipe": "Roll through a round pipe or conduit along its axis, centred, without touching its walls.",
    "flip": "Turn the travel direction around (brake for this interval); move then goes the other way until the next flip.",
}


def skill_action(name):
    if name not in SKILL_NAMES:
        raise ValueError(f"Unknown skill: {name!r}")
    action = np.full(len(SKILL_NAMES), -1., dtype=np.float32)
    action[SKILL_NAMES.index(name)] = 1.
    return action


def parse_skill(response):
    """Accept one JSON command, optionally inside one Markdown code fence."""
    text = response.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    command = json.loads(text)
    if not isinstance(command, dict) or set(command) != {"skill"}:
        raise ValueError('Expected exactly {"skill": "<allowed skill>"}')
    name = command["skill"]
    if not isinstance(name, str) or name not in SKILL_NAMES:
        raise ValueError(f"Unsupported skill: {name!r}")
    return name


def execution_feedback(info):
    # No map, goal distance, privileged path progress, or next teacher action.
    keys = ("skill_name", "skill_timed_out", "wall_contact", "stalled", "success")
    return {key: info[key] for key in keys if key in info}


def make_prompt(instruction, feedback, image_count):
    if not instruction.strip():
        raise ValueError("Provide a non-empty task instruction")
    if image_count not in (1, 2):
        raise ValueError("Use the current image and optionally one previous image")
    catalog = "\n".join(f"- {name}: {SKILL_DESCRIPTIONS[name]}" for name in SKILL_NAMES)
    views = ("One image: the current camera view." if image_count == 1 else
             "Two images in chronological order: previous decision, then current camera view.")
    return (
        "You select one existing skill for a telescopic sphere robot in simulation.\n"
        f"Task instruction: {instruction}\n{views}\n"
        "The executor supplies the predefined route heading; you choose which maneuver to use. "
        "Ordinary movement skills run briefly; jumps run their complete phase sequence. "
        "Use the images and previous execution feedback to decide the next skill.\n"
        f"Previous execution feedback: {json.dumps(feedback, sort_keys=True)}\n"
        f"Available skills:\n{catalog}\n"
        'Return only one JSON object: {"skill": "<one name from the list>"}. '
        "Do not return a sequence, rod positions, or additional text."
    )


@dataclass
class Selection:
    skill: str
    raw_response: str
    error: str | None = None


class VisionSkillSelector:
    def __init__(self, predictor):
        self.predictor = predictor

    def select(self, images, instruction, feedback):
        prompt = make_prompt(instruction, feedback, len(images))
        response = self.predictor(images, prompt)
        try:
            return Selection(parse_skill(response), response)
        except (ValueError, TypeError) as exc:
            # Log the error and brake; never silently substitute a PPO action.
            return Selection("stop", response, str(exc))


class LocalTransformersPredictor:
    """SmolVLM image/text inference; optional dependencies are imported lazily."""

    def __init__(self, model_id, device="cuda", cache_dir=None,
                 allow_download=False, max_new_tokens=64):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.torch, self.device = torch, device
        self.max_new_tokens = int(max_new_tokens)
        options = dict(cache_dir=cache_dir, local_files_only=not allow_download,
                       trust_remote_code=False)
        self.processor = AutoProcessor.from_pretrained(model_id, **options)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa", **options).to(device).eval()

    def __call__(self, images, prompt):
        from PIL import Image

        pil_images = [Image.fromarray(np.asarray(image, dtype=np.uint8)) for image in images]
        messages = [{"role": "user", "content": [
            *[{"type": "image"} for _ in images], {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=pil_images, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        suffix = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(suffix, skip_special_tokens=True)[0]
