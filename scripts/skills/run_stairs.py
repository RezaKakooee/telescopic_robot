"""Traverse configured stairs by composing calibrated jump, stop, and fall skills."""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import imageio
import mujoco
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario, stairs_course_geometry
from radial_sphere.snapshot import make_run_dir
from skills import execute_skill
from skills.hop_planner import (
    PROBE_VX_STEP, PROBE_VZ_STEP, ROLL_RADIUS, STAND_EDGE,
    plan_standing_hop,
)
from skills.overlay import annotate

FORWARD = np.array([1.0, 0.0], dtype=np.float32)
CROUCH_STEPS = 22
MAX_BURN_STEPS = 45


def run(
    *,
    seed: int = 42,
    record_video: bool = True,
    video_name: str = "stairs_verified_composite",
    slowmo: int = 1,
    max_steps: int = 5000,
    preflight_shifts: tuple[float, float] = (0.08, 0.0),
) -> dict:
    """Climb and descend every configured tread, accepting only real contacts."""
    cfg = load_config("configs/rl/stairs_course.yaml")
    geo = stairs_course_geometry(cfg)
    scenario = generate_scenario("stairs", cfg, seed=seed)
    env = MujocoRadialSphereEnv(
        cfg, scenario=scenario, randomize=False, max_steps=max_steps,
    )
    env.reset(seed=seed)
    dt = float(env.model.opt.timestep * env.action_repeat)

    run_dir = make_run_dir(build_run_id("skills", "stairs")) if record_video else None
    out_video = run_dir / "renders" / f"{video_name}.mp4" if run_dir else None
    writer = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(out_video), fps=max(1, int(25 * slowmo)), codec="libx264", quality=9,
        )

    geom_ids = {
        name: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ["floor"]
        + [s["geom"] for s in geo["ascent"]]
        + [s["geom"] for s in geo["descent"] if s["geom"] != "floor"]
    }
    trajectory: list[dict] = []
    climb_results: list[dict] = []
    descent_results: list[dict] = []
    peak_z = float(env.data.qpos[2])
    core_impact_steps = 0
    tick = 0

    def quat() -> np.ndarray:
        return env.data.qpos[3:7].copy()

    def contact_with(name: str) -> bool:
        target_id = geom_ids[name]
        for i in range(env.data.ncon):
            con = env.data.contact[i]
            if con.geom1 == target_id and con.geom2 in env.robot_geom_ids:
                return True
            if con.geom2 == target_id and con.geom1 in env.robot_geom_ids:
                return True
        return False

    def core_contact() -> bool:
        return any(
            env.core_geom_id in (env.data.contact[i].geom1, env.data.contact[i].geom2)
            for i in range(env.data.ncon)
        )

    def direction_toward(sign: float) -> np.ndarray:
        d = np.array([sign, -1.5 * float(env.data.qpos[1])], dtype=np.float32)
        return d / max(float(np.linalg.norm(d)), 1e-6)

    def advance(targets: np.ndarray, phase: str, target_name: str = "") -> None:
        nonlocal peak_z, core_impact_steps, tick
        env.step(targets)
        tick += 1
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        peak_z = max(peak_z, float(pos[2]))
        hit = core_contact()
        core_impact_steps += int(hit)
        trajectory.append({
            "step": tick, "t": tick * dt, "phase": phase, "target": target_name,
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "vx": float(vel[0]), "vy": float(vel[1]), "vz": float(vel[2]),
            "core_contact": hit,
        })
        if writer is not None and tick % 3 == 0:
            frame = env.render(camera_name="stairs_close_30")
            writer.append_data(np.array(annotate(frame, "Verified Composed Stair Skill", [
                f"phase       {phase}",
                f"target      {target_name or '-'}",
                f"position    x {pos[0]:5.2f}  y {pos[1]:+5.2f}  z {pos[2]:5.2f} m",
                f"velocity    vx {vel[0]:+5.2f}  vz {vel[2]:+5.2f} m/s",
                f"verified    up {len(climb_results)}/{geo['n_steps']}  "
                f"down {len(descent_results)}/{geo['n_steps']}",
                f"core contact steps {core_impact_steps}",
            ], margin=14), copy=True))

    def stairs_targets(phase: str, **kwargs) -> np.ndarray:
        return execute_skill(
            "stairs", quat(), env.dirs_body, env.max_extend,
            d_hat=kwargs.pop("d_hat", FORWARD), phase=phase,
            lin_vel=env.data.qvel[:3].copy(), **kwargs,
        )

    def hold(steps: int, label: str) -> None:
        for _ in range(steps):
            advance(stairs_targets("poise", stance_height=0.045), label)

    def position_at(x_target: float, surface_height: float, label: str,
                    tol: float = 0.035, max_position_steps: int = 1600) -> bool:
        stable = 0
        expected_z = surface_height + ROLL_RADIUS
        for _ in range(max_position_steps):
            x = float(env.data.qpos[0])
            err = x_target - x
            speed = float(np.linalg.norm(env.data.qvel[:2]))
            if abs(err) < tol and speed < 0.08:
                stable += 1
            else:
                stable = 0
            if stable >= 8:
                return True
            if abs(float(env.data.qpos[2]) - expected_z) > 0.22:
                return False
            if abs(err) > tol:
                phase = "cruise"
                targets = stairs_targets(
                    phase, d_hat=direction_toward(1.0 if err > 0 else -1.0), speed=0.45,
                )
            else:
                targets = stairs_targets("poise", stance_height=0.045)
            advance(targets, label)
        return False

    def shift_y(dy: float, surface_height: float, label: str) -> bool:
        """Change the rods under the ball without leaving the current tread."""
        y0 = float(env.data.qpos[1])
        y_target = float(np.clip(y0 + dy, -geo["width"] / 2.0 + 0.45,
                                 geo["width"] / 2.0 - 0.45))
        for _ in range(450):
            err = y_target - float(env.data.qpos[1])
            if abs(err) < 0.035 and float(np.linalg.norm(env.data.qvel[:2])) < 0.08:
                hold(35, label)
                return abs(float(env.data.qpos[2]) - (surface_height + ROLL_RADIUS)) < 0.22
            if abs(err) > 0.035:
                d = np.array([0.0, 1.0 if err > 0.0 else -1.0], dtype=np.float32)
                targets = stairs_targets("cruise", d_hat=d, speed=0.42)
            else:
                targets = stairs_targets("poise", stance_height=0.045)
            advance(targets, label)
        return False

    def on_surface(height: float, near: float, far: float) -> bool:
        x, z = float(env.data.qpos[0]), float(env.data.qpos[2])
        return near < x < far and abs(z - (height + ROLL_RADIUS)) < 0.22

    def verified_on(target: dict, contact_seen: bool) -> bool:
        x, z = float(env.data.qpos[0]), float(env.data.qpos[2])
        return (
            contact_seen
            and
            target["near"] < x < target["far"]
            and target["height"] + 0.10 < z < target["height"] + 0.48
            and abs(float(env.data.qvel[2])) < 0.35
        )

    def execute_hop(plan, target: dict) -> dict:
        phase, phase_step = "hop_crouch", 0
        best_vz = -np.inf
        aborted = None
        target_contact_seen = False
        local_peak = float(env.data.qpos[2])
        for _ in range(700):
            z = float(env.data.qpos[2])
            vx, vz = float(env.data.qvel[0]), float(env.data.qvel[2])
            local_peak = max(local_peak, z)
            if phase == "hop_crouch" and phase_step >= CROUCH_STEPS:
                phase, phase_step = "hop_takeoff", 0
            elif phase == "hop_takeoff":
                best_vz = max(best_vz, vz)
                burn_over = vz >= plan.vz_cmd or vz < best_vz - 0.15 or phase_step >= MAX_BURN_STEPS
                # A weak horizontal probe must not end the burn here. On a
                # connected stair the robot already has substantial vertical
                # momentum at step 5; tucking then produces a conspicuous hop
                # that dies against the riser. Keep pushing horizontally and
                # let the velocity servo recover. A launch that is too strong
                # is still aborted because it can overshoot the whole tread.
                if phase_step == PROBE_VX_STEP and vx > plan.vx_gate_hi + 0.08:
                    aborted = f"vx {vx:.2f} above {plan.vx_gate_hi:.2f}"
                elif (phase_step == PROBE_VZ_STEP or burn_over) and best_vz < plan.vz_gate:
                    aborted = f"vz {best_vz:.2f} below {plan.vz_gate:.2f}"
                if aborted or burn_over:
                    phase, phase_step = "hop_airborne", 0
            elif (
                phase == "hop_airborne" and phase_step > 6 and vz < 0.0
                and z < target["height"] + ROLL_RADIUS + 0.16
            ):
                phase, phase_step = "hop_landing", 0

            drop = max(local_peak - (target["height"] + ROLL_RADIUS), 0.10)
            advance(stairs_targets(
                phase, vx_target=plan.vx_cmd, vz_target=plan.vz_cmd,
                wall_lock=True, drop_height=drop,
            ), phase, target["geom"])
            phase_step += 1
            target_contact_seen |= contact_with(target["geom"])
            if phase == "hop_landing" and phase_step > 12 and abs(float(env.data.qvel[2])) < 0.35:
                break

        hold(70, f"step-{target['index']}-brake")
        # The probe is predictive, while a settled target contact is direct
        # evidence. A launch may trip a conservative gate and still land.
        ok = verified_on(target, target_contact_seen)
        return {
            "index": target["index"], "verified": ok, "geom": target["geom"],
            "x": float(env.data.qpos[0]), "z": float(env.data.qpos[2]),
            "peak_z": local_peak, "abort": aborted,
        }

    def execute_drop(current: dict, target: dict) -> dict:
        deck_core = target["height"] + ROLL_RADIUS
        drop_height = current["height"] - target["height"]
        phase, phase_step = "edge", 0
        target_contact_seen = False
        for _ in range(1200):
            z, vz = float(env.data.qpos[2]), float(env.data.qvel[2])
            if phase == "edge" and z < current["height"] + ROLL_RADIUS - 0.06:
                phase, phase_step = "freefall", 0
            elif phase == "freefall" and z < deck_core + 0.10:
                phase, phase_step = "absorb", 0
            elif phase == "absorb" and (contact_with(target["geom"]) or z < deck_core + 0.04):
                phase, phase_step = "brake", 0

            if phase == "brake":
                targets = stairs_targets("poise", stance_height=0.045)
            else:
                targets = stairs_targets(
                    phase, drop_height=drop_height, stance_height=0.045,
                )
            advance(targets, f"drop-{phase}", target["geom"])
            phase_step += 1
            target_contact_seen |= contact_with(target["geom"])
            if phase == "brake" and phase_step > 70:
                break

        hold(50, f"drop-{target['index']}-settle")
        ok = verified_on(target, target_contact_seen)
        return {
            "index": target["index"], "verified": ok, "geom": target["geom"],
            "x": float(env.data.qpos[0]), "z": float(env.data.qpos[2]),
            "drop_height": drop_height,
        }

    print("\n=== Verified stair composition: jump_to + stop + fall_down ===")
    print(f"  {geo['n_steps']} steps, rise {geo['rise']:.2f} m, run {geo['run']:.2f} m")

    plans = []
    from_height, from_range = 0.0, (-100.0, geo["start_x"])
    for target in geo["ascent"]:
        plan = plan_standing_hop(from_height, from_range, target)
        if plan is None:
            raise RuntimeError(f"no calibrated hop plan for stair {target['index']}")
        plans.append(plan)
        from_height = target["height"]
        from_range = (target["near"] + STAND_EDGE, target["far"] - STAND_EDGE)

    success = True
    current_height = 0.0
    current_range = (-100.0, geo["start_x"])
    for plan, target in zip(plans, geo["ascent"]):
        # A landing changes which rods sit under the sphere. Re-deal that
        # footing before the next launch with a small, deliberate side roll;
        # this is visually a setup action, not a failed jump against a riser.
        if target["index"] > 1:
            prep_dy = float(preflight_shifts[target["index"] - 2])
            if abs(prep_dy) > 1e-6:
                shift_y(prep_dy, current_height, f"prepare-step-{target['index']}")
        result = None
        for attempt in range(1, 5):
            if attempt > 1:
                dy = (0.08 + 0.03 * (attempt - 2)) * (1.0 if attempt % 2 else -1.0)
                if not shift_y(dy, current_height,
                               f"re-deal-step-{target['index']}-{attempt}"):
                    break
            positioned = position_at(
                plan.x0, current_height, f"stage-step-{target['index']}-{attempt}",
            )
            if not positioned:
                result = {
                    "index": target["index"], "verified": False, "geom": target["geom"],
                    "x": float(env.data.qpos[0]), "z": float(env.data.qpos[2]),
                    "peak_z": peak_z, "abort": "could not reach staging point",
                }
                break
            result = execute_hop(plan, target)
            result["attempts"] = attempt
            if result["verified"]:
                break
            if not on_surface(current_height, current_range[0], current_range[1]):
                break
        assert result is not None
        climb_results.append(result)
        print(f"  climb {target['index']}: {'VERIFIED' if result['verified'] else 'FAILED'} "
              f"on {result['geom']} at x={result['x']:.2f}, z={result['z']:.2f} "
              f"({result.get('attempts', 1)} attempt{'s' if result.get('attempts', 1) != 1 else ''})")
        if not result["verified"]:
            success = False
            break
        current_height = target["height"]
        current_range = (target["near"], target["far"])

    if success:
        first_edge = geo["descent_start"] + geo["run"]
        success = position_at(first_edge - 0.32, geo["top"], "cross-plateau")
        current = {"height": geo["top"], "far": first_edge}
        for target in geo["descent"]:
            if not success:
                break
            result = execute_drop(current, target)
            descent_results.append(result)
            print(f"  descent {target['index']}: {'VERIFIED' if result['verified'] else 'FAILED'} "
                  f"on {result['geom']} at x={result['x']:.2f}, z={result['z']:.2f}")
            success = result["verified"]
            if success and target["index"] < geo["n_steps"]:
                success = position_at(target["far"] - 0.32, target["height"],
                                      f"stage-drop-{target['index'] + 1}")
            current = target

    if success:
        position_at(geo["finish_x"] - 0.22, 0.0, "finish")
        hold(80, "finish-settle")

    end_pos = env.data.qpos[:3].copy()
    end_vel = env.data.qvel[:6].copy()
    if writer is not None:
        writer.close()
    env.close()

    all_climbs = len(climb_results) == geo["n_steps"] and all(x["verified"] for x in climb_results)
    all_descents = len(descent_results) == geo["n_steps"] and all(x["verified"] for x in descent_results)
    success = bool(success and all_climbs and all_descents)
    print("\n--- Summary ---")
    print(f"  verified climbs:  {sum(x['verified'] for x in climb_results)} / {geo['n_steps']}")
    print(f"  verified descents:{sum(x['verified'] for x in descent_results)} / {geo['n_steps']}")
    print(f"  elapsed:          {len(trajectory) * dt:.2f} s")
    print(f"  core impacts:     {core_impact_steps}")
    print(f"  final:            x={end_pos[0]:.2f}, y={end_pos[1]:+.2f}, z={end_pos[2]:.2f}")
    if out_video is not None:
        print(f"  video:            {out_video}")

    return {
        "success": success,
        "steps_climbed": sum(x["verified"] for x in climb_results),
        "steps_descended": sum(x["verified"] for x in descent_results),
        "climbs": climb_results, "descents": descent_results,
        "peak_z": peak_z, "final_pos": tuple(float(x) for x in end_pos),
        "end_speed": float(np.linalg.norm(end_vel[:3])),
        "end_spin": float(np.linalg.norm(end_vel[3:6])),
        "core_impacts": core_impact_steps,
        "elapsed_s": len(trajectory) * dt,
        "trajectory": trajectory,
        "video": str(out_video) if out_video is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verified composed stair traversal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slowmo", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run(seed=args.seed, slowmo=args.slowmo, record_video=not args.no_video)


if __name__ == "__main__":
    main()
