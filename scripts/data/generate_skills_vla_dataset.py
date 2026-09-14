"""Generate Expert Demonstration Dataset for Parkour VLA with Modular skills_vla.

Rolls out a deterministic, context-aware scripted oracle on the 3D Playground Parkour task.
Records synchronized:
  - Visual RGB observation frame (256x256x3) from onboard tracking camera
  - Proprioceptive robot state vector (11-D: pos, vel, rel_goal, dist_goal, quat)
  - Ground-truth skill selection (5 classes: roll, jump_forward, jump_gap, traverse_rough, brake_stop)
  - Normalized continuous skill parameters in [-1, 1]
  - Natural language instruction string
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import h5py
import numpy as np

from radial_sphere import playground_course as PC
from radial_sphere.config import load_config_cli
from radial_sphere.run_id import build_run_id
from radial_sphere.snapshot import make_run_dir
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from scripts.data.generate_obstacle_vla_dataset import render_vla_frame
from skills_vla import ENV_SKILL_MAP, SKILL_NAMES, get_skill

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_skills_vla_dataset")

# Mapping from canonical skills_vla name to discrete skill index
SKILL_TO_IDX = {name: idx for idx, name in enumerate(SKILL_NAMES)}
# Canonical list: ('roll', 'jump_forward', 'jump_gap', 'traverse_rough', 'brake_stop')


# Jump triggers, measured with scratch/measure_jump_trigger.py on the macro
# options (running jump = jump_forward_while_moving with the sprint skipped;
# jump_to at the macro take-off velocity). Each entry: (axis, sign, edge, near,
# far). The jump fires when the ball is between `near` and `far` metres before
# `edge` along `axis` in direction `sign`. Windows are at least 0.12 m wide so
# one macro decision (about 0.11 m at 1.1 m/s) always lands inside.
JUMP_TRIGGERS = {
    "hurdle":  (0, +1, PC.HURDLE_X - PC.HURDLE_BAR_R, 0.66, 0.85),
    "box1":    (1, +1, PC.BOX_Y0[0], 0.58, 0.75),
    "valley1": (1, +1, PC.BOX_Y1[0], 0.18, 0.36),
    "valley2": (1, +1, PC.BOX_Y1[1], 0.18, 0.36),
    "wall":    (1, +1, PC.WALL_Y - PC.WALL_HALF_T, 0.56, 0.70),
}
# One trigger per stair riser. The 1.8 m treads leave room to land and run up again.
for _i, _rx in enumerate(PC.STAIR_RISER_X):
    JUMP_TRIGGERS[f"stairs{_i}"] = (0, -1, _rx, 0.38, 0.55)
STAIR_KEYS = tuple(f"stairs{i}" for i in range(PC.STAIR_N))


def _in_window(pos: np.ndarray, key: str) -> bool:
    axis, sign, edge, near, far = JUMP_TRIGGERS[key]
    d = sign * (edge - float(pos[axis]))
    return near <= d <= far


def select_scripted_skill(pos: np.ndarray, guidance: np.ndarray, goal_dist: float) -> tuple[str, list[float], str]:
    """Deterministic spatial oracle for the 3D Playground Parkour.

    Jumps fire from a measured distance before each obstacle edge so the ball
    clears it without touching (see JUMP_TRIGGERS). Everything else rolls.
    """
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

    if goal_dist < 0.45 or y >= 16.2:
        return "brake_stop", [0.0, 0.0], "Halt and stabilize position on the green goal pad."

    # Jump windows first: they are narrow and must not be shadowed by a region rule.
    if y < 1.0 and x < 8.0 and _in_window(pos, "hurdle"):
        return "jump_forward", [0.0, 0.95], "Leap over the 0.18 m hurdle without touching the bar."
    if x > 7.5 and y < 3.0 and _in_window(pos, "box1"):
        return "jump_gap", [0.0, 0.85, 0.80], "Hop onto the first platform box without touching its face."
    if x > 7.5 and _in_window(pos, "valley1"):
        return "jump_gap", [0.0, 0.85, 0.80], "Leap across Valley 1 onto the next platform."
    if x > 7.5 and _in_window(pos, "valley2"):
        return "jump_gap", [0.0, 0.85, 0.80], "Leap across Valley 2 onto the landing deck."
    if y > 7.5 and any(_in_window(pos, k) for k in STAIR_KEYS):
        return "jump_forward", [0.0, 0.95], "Vault up the next stair step without touching the riser."
    if x <= 1.5 and y > 10.5 and _in_window(pos, "wall"):
        return "jump_forward", [0.0, 0.90], "Leap over the 0.20 m wall without touching it."

    # 1. Leg 1: (0, 0) -> (9.0, 0.0) heading East (+X)
    if y < 1.0 and x < 8.0:
        return "roll", [0.0, 0.50], "Roll centered along the street corridor away from side walls."

    # Turn 1: Corner near (9.0, 0.0) turning North (+Y)
    if x >= 8.0 and y < 1.5:
        return "roll", [0.25, 0.35], "Round Turn 1 apex smoothly along the corridor centerline."

    # 2. Leg 2: x ~ 9.0, moving North (+Y) across Box 1, Box 2, and Chasms
    if x > 7.5 and 1.5 <= y < 8.5:
        return "roll", [0.0, 0.40], "Roll forward centered along the elevated parkour platform."

    # Turn 2: Corner near (9.0, 9.0) turning West (-X)
    if x > 7.5 and y >= 8.0:
        return "roll", [0.25, 0.35], "Round Turn 2 smoothly to align heading west toward the stairs."

    # 3. Leg 3: y ~ 9.0, moving West (-X) over the 3-step stairs and the terrace deck
    if y > 7.5 and 1.5 < x <= 7.5:
        return "roll", [0.0, 0.40], "Roll centered across the elevated terrace deck and down the ramp."

    # Turn 3: Corner near (0.0, 9.0) turning North (+Y)
    if x <= 1.5 and y < 10.5:
        return "roll", [0.25, 0.35], "Round Turn 3 north into the final approach corridor."

    # 4. Leg 4: moving North (+Y) across the cobblestone bed, then over the jump wall
    if 10.5 <= y < 12.4:
        return "traverse_rough", [0.0, 0.30, 0.60], "Traverse irregular cobblestones with compliant suspension."

    # Station 7: Final approach to green goal pad
    return "roll", [0.0, 0.40], "Roll along final corridor toward the green goal pad."


def collect_skills_vla_demonstrations(
    config_name: str = "playground_parkour_skills",
    output_h5: Path | None = None,
    n_episodes: int = 30,
    seed_offset: int = 100,
    max_macro_steps: int = 300,
):
    """Collect pure scripted expert demonstrations using modular skills_vla."""
    cfg = load_config_cli(name=config_name)
    if output_h5 is None:
        # Every run gets its own timestamped folder under storage_local/.
        output_h5 = make_run_dir(build_run_id("generate_skills_vla_dataset", tag=config_name)) / "parkour_skills_vla_demos.h5"
    scenario = generate_scenario("playground", cfg, seed=seed_offset)
    env = SkillArbitrationEnv(cfg, scenario=scenario, seed=seed_offset, max_steps=1500)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(output_h5, "w")

    successful_episodes = 0
    total_transitions = 0
    episodes_meta = []

    env_skill_map = ENV_SKILL_MAP

    logger.info(f"Collecting {n_episodes} demonstration episodes on 3D Playground Parkour...")

    for ep_idx in range(n_episodes):
        seed = seed_offset + ep_idx
        obs, info = env.reset(seed=seed)

        frames = []
        states = []
        skills = []
        params_list = []
        instructions = []

        terminated = truncated = False
        step = 0
        hits = 0

        goal = np.asarray(env.scenario.goal, dtype=np.float32)[:2]

        while not (terminated or truncated) and step < max_macro_steps:
            # 1. Capture onboard RGB visual perception frame
            frame = render_vla_frame(env.env, width=256, height=256)

            # 2. Capture proprioceptive state vector
            pos = env.env.data.qpos[:3].copy()
            quat = env.env.data.qpos[3:7].copy()
            vel = env.env.data.qvel[:3].copy()
            ball_xy = pos[:2]
            rel_goal = goal - ball_xy
            dist_goal = float(np.linalg.norm(rel_goal))

            guidance = env.waypoint_tracker.get_guidance(pos)

            state_vec = np.concatenate([
                ball_xy.astype(np.float32),
                vel[:2].astype(np.float32),
                rel_goal.astype(np.float32),
                np.array([dist_goal], dtype=np.float32),
                quat.astype(np.float32),
            ])

            # 3. Scripted Oracle selects skill & parameters
            skill_name, skill_params, instruction = select_scripted_skill(pos, guidance, dist_goal)
            skill_idx = SKILL_TO_IDX[skill_name]

            # Store transition
            frames.append(frame)
            states.append(state_vec)
            skills.append(skill_idx)
            # Pad parameters to 4-D for uniform storage
            padded_params = np.zeros(4, dtype=np.float32)
            padded_params[:len(skill_params)] = skill_params
            params_list.append(padded_params)
            instructions.append(instruction)

            # 4. Execute selected skill in physics
            env_skill = env_skill_map[skill_name]
            act = np.full(len(env.skill_names), -1.0, dtype=np.float32)
            act[env.skill_names.index(env_skill)] = 1.0
            obs, reward, terminated, truncated, info = env.step(act)
            step += 1
            hits += int(info.get("obstacle_hit", 0))

        final_pos = env.env.data.qpos[:2].copy()
        final_dist = float(np.linalg.norm(final_pos - goal))
        success = bool(info.get("success", False) or final_dist < 0.60)

        logger.info(
            f"Episode {ep_idx + 1:02d}/{n_episodes:02d} (seed={seed}) | "
            f"Steps: {step} | Pos: ({final_pos[0]:.2f}, {final_pos[1]:.2f}) | "
            f"Final Dist: {final_dist:.2f}m | Success: {success} | Hits: {hits}"
        )

        if success and hits == 0:  # keep only clean demonstrations
            ep_grp = h5_file.create_group(f"episode_{successful_episodes:03d}")
            ep_grp.create_dataset("frames", data=np.array(frames, dtype=np.uint8), compression="gzip", compression_opts=4)
            ep_grp.create_dataset("states", data=np.array(states, dtype=np.float32))
            ep_grp.create_dataset("skills", data=np.array(skills, dtype=np.int64))
            ep_grp.create_dataset("params", data=np.array(params_list, dtype=np.float32))
            ep_grp.create_dataset("instructions", data=np.array(instructions, dtype=h5py.string_dtype()))

            ep_grp.attrs["steps"] = len(frames)
            ep_grp.attrs["success"] = success
            ep_grp.attrs["final_dist"] = final_dist
            ep_grp.attrs["obstacle_hits"] = hits
            ep_grp.attrs["seed"] = seed

            successful_episodes += 1
            total_transitions += len(frames)
            episodes_meta.append({
                "episode": successful_episodes,
                "seed": seed,
                "steps": len(frames),
                "success": success,
                "final_dist": final_dist,
                "obstacle_hits": hits,
            })

    env.close()
    h5_file.close()

    summary = {
        "dataset_path": str(output_h5),
        "total_episodes": successful_episodes,
        "total_transitions": total_transitions,
        "skills": list(SKILL_NAMES),
        "episodes": episodes_meta,
    }

    summary_file = output_h5.parent / "skills_vla_demos_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 65)
    logger.info(f"SCRIPTED DEMONSTRATION GENERATION COMPLETE: {output_h5}")
    logger.info(f"Total Successful Episodes: {successful_episodes}/{n_episodes}")
    logger.info(f"Total Macro Transitions:  {total_transitions}")
    logger.info(f"Summary JSON:              {summary_file}")
    logger.info("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Collect scripted parkour demonstrations with skills_vla.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="playground_parkour_skills",
        help="Config preset name",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output HDF5 path (default: a new timestamped run dir under storage_local/)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=25,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=100,
        help="Starting random seed",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Max macro steps per episode",
    )
    args = parser.parse_args()

    collect_skills_vla_demonstrations(
        config_name=args.config_name,
        output_h5=Path(args.out) if args.out else None,
        n_episodes=args.episodes,
        seed_offset=args.seed_offset,
        max_macro_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
