"""Render the rough-terrain suspension demo (active suspension over rough ground).

Two runs roam the same rough arena side by side for two minutes. There is no
goal and no finish line. Both runs use the same roaming skill, the same
command speed and the same seed. Only the terrain feedback differs:

  * left  - fixed stance. Height, contact and terrain gains are all zero.
  * right - active suspension. Height PD, contact yield and terrain reach.

Heading comes from ``stay_in_boundary``, which keeps the robot inside a
wall-less circle and picks its own roaming actions. Both panels therefore
receive the same action sequence and diverge only through physics. The rod
support corrections under test are the ones in ``skills/low_level/suspension.py``,
shared with ``traverse_rough_terrain``.

The camera tracks each robot from the side, so the rods under the body stay
large in frame. Each foot is recoloured every frame by what its own terrain
ray measured, so the sensing is visible:

  * green - the ray found ground below the nominal floor, a pit
  * amber - the ray found ground above the nominal floor, a rock
  * dark  - flat ground, or nothing inside the rod's reach

Outputs:
  - docs/blog/assets/rough-terrain-suspension.mp4
  - docs/blog/assets/rough-terrain-suspension-preview.png
  - docs/blog/assets/rough-terrain-suspension.csv
  - docs/blog/assets/rough-terrain-suspension-results.json
"""
import os
import sys
from collections import deque
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

os.environ.setdefault("MUJOCO_GL", "egl")
import csv
import json
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _assets import assets_dir  # noqa: E402

from radial_sphere.config import load_config, script_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.mid_level.navigation import stay_in_boundary
from skills.low_level.suspension import SuspensionGains, SuspensionState

BOUNDARY_RADIUS = 3.4
N_STONES = 160
MAX_STONE_SIZE = 0.080

# Recessed pits scattered across the arena. Four centimetres is deep enough
# to read as a hole from the tracking camera and shallow enough that a rod
# can reach the bottom and push out again.
PIT_DEPTH = 0.040
PITS = [
    # (cx, cy, half_x, half_y, depth)
    [1.37, 0.72, 0.20, 0.20, PIT_DEPTH],
    [-1.44, 0.98, 0.20, 0.20, PIT_DEPTH],
    [1.11, -1.50, 0.20, 0.20, PIT_DEPTH],
    [-1.24, -1.05, 0.20, 0.20, PIT_DEPTH],
    [0.07, 2.03, 0.20, 0.20, PIT_DEPTH],
    [-2.22, -0.07, 0.20, 0.20, PIT_DEPTH],
    [2.35, -0.60, 0.20, 0.20, PIT_DEPTH],
    [-0.55, -2.35, 0.20, 0.20, PIT_DEPTH],
]

# Low timber ledges laid across the arena. The plain rolling gait is stopped
# by a step of about 4 cm, so these are the clearest place to see the robot
# climb rather than steer around.
LEDGE_HEIGHT = 0.065
LEDGES = [
    # (cx, cy, half_x, half_y, height)
    [0.52, -2.42, 0.80, 0.09, LEDGE_HEIGHT],
    [-2.03, 1.50, 0.80, 0.09, LEDGE_HEIGHT],
    [2.42, 1.24, 0.09, 0.80, LEDGE_HEIGHT],
    [-0.39, -0.72, 0.09, 0.70, LEDGE_HEIGHT],
    [1.57, -2.62, 0.70, 0.09, LEDGE_HEIGHT],
    [-1.80, -1.90, 0.09, 0.70, LEDGE_HEIGHT],
    [0.95, 1.60, 0.70, 0.09, LEDGE_HEIGHT],
    [-0.10, 0.55, 0.70, 0.09, LEDGE_HEIGHT],
]

# Terrain reading below which a rod is called "on a rock", and above which it
# is called "over a pit". Chosen just outside the ray noise on flat ground.
ROCK_LEVEL = -0.012
PIT_LEVEL = 0.012

# Drive-wave amplitude for rough travel, taken straight from the original
# rocky-terrain controller config. The speed calibration would ask for about
# 1.45 here, and speed feedback then throttles it hard whenever the robot
# runs fast, which starves the push right before the next rock. Commanding
# amplitude directly keeps it. 2.0 lifts the climbable step from 6 cm to 8 cm.
ROUGH_DRIVE_GAIN = 2.0

# The rolling gait holds the core near 23 cm on flat ground with this build.
# A target above the rod stroke would pin the height term at its clip and
# turn the PD correction into a constant offset.
RIDE_HEIGHT = 0.23

# One tuning, and its control arm. Deriving the fixed-stance run from the
# active one is the point of the comparison: the gait, the ride-height target
# and the slew limit stay identical, only the corrections stop.
ACTIVE_SUSPENSION = SuspensionGains(
    target_ride_height=RIDE_HEIGHT,
    kp=0.80,
    kd=0.20,
    force_compliance=0.0022,
    nominal_support_force=10.0,
    terrain_adaptation=0.90,
    hole_reach=0.055,
)
FIXED_STANCE = ACTIVE_SUSPENSION.without_feedback()

RUNS = [
    ("FIXED STANCE", "no terrain feedback", FIXED_STANCE, (250, 204, 21)),
    ("ACTIVE SUSPENSION", "height + contact + terrain", ACTIVE_SUSPENSION, (74, 222, 128)),
]

CANVAS_W, CANVAS_H = 1280, 700
PANEL_W, PANEL_H = 620, 420
RENDER_H = 580          # rendered tall, then cropped to drop empty sky
CROP_TOP = 130
PANEL_X = (16, 644)
PANEL_Y = 130
MAP_Y, MAP_SIZE = 556, 108
TEXT_X_OFFSET = MAP_SIZE + 14

BG = (15, 23, 42)
INK = (226, 232, 240)
DIM = (148, 163, 184)
RULE = (51, 65, 85)

TRAIL_STRIDE = 10       # sim steps between stored trail points
TRAIL_POINTS = 420      # about 42 s of history


def build_env(cfg):
    """One env holding the shared rough arena."""
    scenario = generate_scenario(
        "boundary", cfg, radius=BOUNDARY_RADIUS, n_segments=64, seed=42,
        n_stones=N_STONES, max_stone_size=MAX_STONE_SIZE,
    )
    scenario.gaps = PITS
    scenario.steps = LEDGES
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=200000)
    env.reset(seed=42)
    return env


def lighten_pit_floors(env):
    """Make the pit interior read as sunken ground, not as a black void.

    The shared gap geometry paints a deep chasm, which suits the jump demos.
    Here the pits are four-centimetre depressions, so the floor is given a
    plain ground colour instead.
    """
    for gid in range(env.model.ngeom):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("gap_pit_floor_"):
            env.model.geom_rgba[gid] = [0.16, 0.17, 0.20, 1.0]


def paint_feet(env, clearances):
    """Colour every foot by what its own terrain ray measured this step.

    The env only refreshes rod colours inside its own ``render``, which this
    script does not use. Repaint the default first, or a foot keeps the
    colour it was given several frames ago.
    """
    env._update_dynamic_colors()
    for k in range(env.n_bars):
        fid, _ = env._bar_geom_ids[k]
        if fid < 0:
            continue
        value = clearances[k]
        if np.isnan(value):
            continue
        if value > PIT_LEVEL:
            env.model.geom_rgba[fid] = [0.29, 0.87, 0.50, 1.0]
        elif value < ROCK_LEVEL:
            env.model.geom_rgba[fid] = [0.98, 0.80, 0.08, 1.0]


def make_camera(env):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.core_body_id
    camera.azimuth = 90.0
    camera.elevation = -24.0     # look down enough to see into a 4 cm pit
    camera.distance = 1.25
    return camera


def draw_map(draw, x0, y0, size, trail, ball_xy, accent):
    """Plan view of the arena: boundary, pits, recent path and the robot."""
    span = BOUNDARY_RADIUS + 0.45
    half = size / 2.0
    scale = half / span

    def to_px(xy):
        return (x0 + half + xy[0] * scale, y0 + half - xy[1] * scale)

    draw.rectangle([(x0, y0), (x0 + size, y0 + size)], fill=(23, 32, 50), outline=RULE)
    ring = BOUNDARY_RADIUS * scale
    draw.ellipse([(x0 + half - ring, y0 + half - ring), (x0 + half + ring, y0 + half + ring)],
                 fill=(38, 50, 70), outline=(190, 160, 40), width=1)
    for cx, cy, hx, hy, _ in LEDGES:
        draw.rectangle([to_px((cx - hx, cy + hy)), to_px((cx + hx, cy - hy))],
                       fill=(150, 100, 58))
    for cx, cy, hx, hy, _ in PITS:
        draw.rectangle([to_px((cx - hx, cy + hy)), to_px((cx + hx, cy - hy))],
                       fill=(8, 12, 22))
    if len(trail) > 1:
        draw.line([to_px(p) for p in trail], fill=(120, 138, 165), width=1)
    px, py = to_px(ball_xy)
    draw.ellipse([(px - 4, py - 4), (px + 4, py + 4)], fill=accent,
                 outline=(15, 23, 42), width=1)


def main():
    args = script_config("render_rough_terrain")
    if args.seconds <= 0:
        raise SystemExit("seconds must be positive")
    # `output_dir` wins when it is set to something other than the default,
    # otherwise fall back to the shared BLOG_ASSETS_DIR override.
    assets = Path(args.output_dir)
    if str(args.output_dir) == "docs/blog/assets":
        assets = assets_dir()
    elif not assets.is_absolute():
        assets = repo_root / assets
    assets.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/rl/standing_jump_showcase.yaml")
    cfg.camera.enabled = False

    envs, cameras, renderers, states, trails = [], [], [], [], []
    for _ in RUNS:
        env = build_env(cfg)
        lighten_pit_floors(env)
        env.model.vis.global_.offwidth = PANEL_W
        env.model.vis.global_.offheight = RENDER_H
        # Let the robot settle onto its rods before the recording starts.
        for _ in range(25):
            env.step(np.full(env.n_bars, 0.025, dtype=np.float32))
        envs.append(env)
        cameras.append(make_camera(env))
        renderers.append(mujoco.Renderer(env.model, height=RENDER_H, width=PANEL_W))
        states.append(SuspensionState(targets=env.data.ctrl.copy()))
        trails.append(deque(maxlen=TRAIL_POINTS))

    dt = float(envs[0].model.opt.timestep * envs[0].action_repeat)
    total_steps = round(args.seconds / dt)
    video_stride = 4
    fps = round(1.0 / (dt * video_stride))

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    title_font = ImageFont.truetype(font_bold_path, 23)
    label_font = ImageFont.truetype(font_bold_path, 15)
    body_font = ImageFont.truetype(font_path, 15)
    small_font = ImageFont.truetype(font_path, 13)
    clock_font = ImageFont.truetype(font_bold_path, 26)

    video_path = assets / "rough-terrain-suspension.mp4"
    preview_path = assets / "rough-terrain-suspension-preview.png"
    csv_path = assets / "rough-terrain-suspension.csv"
    json_path = assets / "rough-terrain-suspension-results.json"

    writer = imageio.get_writer(
        str(video_path), fps=fps, codec="libx264", pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart", "-crf", str(args.crf), "-preset", "medium"],
    )

    logs = [[] for _ in RUNS]
    path_len = [0.0 for _ in RUNS]
    obstruction = [0 for _ in RUNS]
    climb_steps = [0 for _ in RUNS]
    escape_steps = [0 for _ in RUNS]
    last_xy = [envs[i].data.qpos[:2].copy() for i in range(len(RUNS))]
    csv_rows = []
    saved_preview = False
    t_sim = 0.0
    start_wall = time.time()
    print(f"Recording {args.seconds:.0f} s side-by-side rough-terrain roam "
          f"({total_steps} steps, {total_steps // video_stride} frames) to {video_path}...")

    for step in range(total_steps):
        frame_state = []
        for run_i, (_, _, gains, _) in enumerate(RUNS):
            env = envs[run_i]
            pos = env.data.qpos[:3].copy()
            vel = env.data.qvel[:3].copy()
            clearances = env.get_terrain_clearances()
            forces = env.get_rod_contact_forces()

            targets, meta = stay_in_boundary(
                env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                ball_xy=pos[:2], lin_vel=vel[:2],
                boundary_radius=BOUNDARY_RADIUS, speed=args.speed,
                safety_margin=0.70, step_count=step,
                core_z=float(pos[2]), core_vz=float(vel[2]),
                contact_forces=forces, terrain_clearances=clearances,
                enable_suspension=True, suspension=gains,
                suspension_state=states[run_i],
                control_dt=dt, rough_terrain_gait=True,
                rough_drive_gain=ROUGH_DRIVE_GAIN,
                obstruction_steps=obstruction[run_i],
                return_metadata=True,
            )
            obstruction[run_i] = obstruction[run_i] + 1 if meta["is_obstructed"] else 0
            climb_steps[run_i] += meta["action_name"] == "roam_climb_over"
            escape_steps[run_i] += meta["action_name"] == "roam_obstacle_escape"
            env.step(targets)
            paint_feet(env, clearances)

            path_len[run_i] += float(np.linalg.norm(pos[:2] - last_xy[run_i]))
            last_xy[run_i] = pos[:2].copy()
            if step % TRAIL_STRIDE == 0:
                trails[run_i].append((float(pos[0]), float(pos[1])))

            finite = clearances[np.isfinite(clearances)]
            frame_state.append({
                "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                "speed": float(np.linalg.norm(vel[:2])),
                "n_rock": int(np.sum(finite < ROCK_LEVEL)),
                "n_pit": int(np.sum(finite > PIT_LEVEL)),
                "stroke_max": float(np.max(targets)),
                "max_force": float(np.max(forces)) if len(forces) else 0.0,
                "action": meta["action_name"],
                "path": path_len[run_i],
            })
            logs[run_i].append((float(pos[2]), float(vel[2]), float(np.linalg.norm(vel[:2]))))

        t_sim += dt

        if (step + 1) % 10 == 0:
            row = {"step": step + 1, "time_s": round(t_sim, 3)}
            for run_i in range(len(RUNS)):
                tag = "fixed" if run_i == 0 else "active"
                s = frame_state[run_i]
                row.update({
                    f"{tag}_x_m": round(s["x"], 4),
                    f"{tag}_y_m": round(s["y"], 4),
                    f"{tag}_core_z_m": round(s["z"], 4),
                    f"{tag}_speed_mps": round(s["speed"], 4),
                    f"{tag}_path_m": round(s["path"], 3),
                    f"{tag}_rods_on_rock": s["n_rock"],
                    f"{tag}_rods_over_pit": s["n_pit"],
                    f"{tag}_action": s["action"],
                })
            csv_rows.append(row)

        if (step + 1) % video_stride:
            continue

        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        draw = ImageDraw.Draw(canvas)

        draw.text((20, 12), "Rough terrain: fixed stance against active suspension",
                  font=title_font, fill=(255, 255, 255))
        draw.text((20, 44),
                  "Two minutes of free roaming in the same rocky arena. No goal. "
                  "Same skill, same command speed, same seed.", font=body_font, fill=DIM)
        draw.text((20, 66),
                  "Foot colour is the rod's own terrain ray:  green = ground below "
                  "the floor (pit),  amber = ground above it (rock).",
                  font=small_font, fill=DIM)
        draw.text((1132, 12), f"{t_sim:5.1f} s", font=clock_font, fill=(255, 255, 255))
        draw.text((1132, 44), f"command {args.speed * 100:.0f} cm/s",
                  font=small_font, fill=DIM)
        draw.line([(0, 98), (CANVAS_W, 98)], fill=RULE, width=2)

        for run_i, (name, subtitle, _, accent) in enumerate(RUNS):
            px = PANEL_X[run_i]
            state = frame_state[run_i]

            renderers[run_i].update_scene(envs[run_i].data, cameras[run_i])
            view = renderers[run_i].render()[CROP_TOP:CROP_TOP + PANEL_H]
            canvas.paste(Image.fromarray(view), (px, PANEL_Y))
            draw.rectangle([(px, PANEL_Y), (px + PANEL_W - 1, PANEL_Y + PANEL_H - 1)],
                           outline=accent, width=2)

            draw.rectangle([(px, 104), (px + PANEL_W, 124)], fill=accent)
            draw.text((px + 10, 105), f"{name}  -  {subtitle}", font=label_font, fill=BG)

            draw_map(draw, px, MAP_Y, MAP_SIZE, trails[run_i],
                     (state["x"], state["y"]), accent)

            tx = px + TEXT_X_OFFSET
            draw.text((tx, MAP_Y + 2),
                      f"Distance rolled {state['path']:6.1f} m", font=body_font, fill=INK)
            draw.text((tx, MAP_Y + 26),
                      f"Speed {state['speed'] * 100:3.0f} cm/s     "
                      f"Core {state['z'] * 100:4.1f} cm", font=body_font, fill=INK)
            used = "used" if run_i == 1 else "measured, not used"
            draw.text((tx, MAP_Y + 50),
                      f"Rods on rock {state['n_rock']:2d}   Rods over pit "
                      f"{state['n_pit']:2d}   ({used})", font=body_font, fill=accent)
            draw.text((tx, MAP_Y + 74),
                      f"Action {state['action']}     Longest rod "
                      f"{state['stroke_max'] * 100:4.1f} cm     "
                      f"Peak foot force {state['max_force']:4.0f} N",
                      font=small_font, fill=DIM)

        draw.line([(0, 672), (CANVAS_W, 672)], fill=RULE, width=1)
        draw.text((20, 678),
                  f"Arena: {BOUNDARY_RADIUS:.1f} m wall-less circle, {N_STONES} boulders "
                  f"up to {MAX_STONE_SIZE * 100:.1f} cm, {len(PITS)} pits "
                  f"{PIT_DEPTH * 100:.0f} cm deep, {len(LEDGES)} timber ledges "
                  f"{LEDGE_HEIGHT * 100:.1f} cm high.    Map: yellow ring = boundary, "
                  "black = pits, brown = ledges, grey line = path.",
                  font=small_font, fill=DIM)

        writer.append_data(np.array(canvas))

        if not saved_preview and frame_state[1]["n_pit"] >= 5 and t_sim > 5.0:
            canvas.save(preview_path)
            saved_preview = True
            print(f"Saved preview at t={t_sim:.1f}s to {preview_path}")

        if (step + 1) % 2000 == 0:
            print(f"  step {step + 1}/{total_steps}  t={t_sim:.0f}s  "
                  f"fixed {path_len[0]:.1f} m  active {path_len[1]:.1f} m  "
                  f"climbs {climb_steps[0]}/{climb_steps[1]}  "
                  f"[{time.time() - start_wall:.0f}s]")

    if not saved_preview:
        canvas.save(preview_path)
        print(f"Saved fallback preview to {preview_path}")

    writer.close()
    for env in envs:
        env.close()

    with open(csv_path, "w", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        dict_writer.writeheader()
        dict_writer.writerows(csv_rows)

    # The first second is start-up, not roaming.
    warmup = round(1.0 / dt)
    summary = {}
    for run_i, _ in enumerate(RUNS):
        z, vz, speed = np.asarray(logs[run_i])[warmup:].T
        stalled = speed < 0.06
        longest, current = 0, 0
        for flag in stalled:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        tag = "fixed_stance" if run_i == 0 else "active_suspension"
        summary[tag] = {
            "distance_rolled_m": round(path_len[run_i], 2),
            "mean_speed_mps": round(float(speed.mean()), 3),
            "stalled_time_pct": round(float(stalled.mean() * 100), 1),
            "longest_stall_s": round(longest * dt, 2),
            "climb_over_steps": int(climb_steps[run_i]),
            "pivot_away_steps": int(escape_steps[run_i]),
            "core_height_mean_cm": round(float(z.mean() * 100), 2),
            "core_height_std_cm": round(float(z.std() * 100), 2),
            "vertical_speed_rms_cm_s": round(float(np.sqrt((vz ** 2).mean()) * 100), 2),
        }

    results = {
        "scenario": "rough_arena_free_roam_boulders_and_pits",
        "comparison": "same roaming skill and command speed, terrain feedback gains on or off",
        "heading_source": "stay_in_boundary",
        "seed": 42,
        "command_speed_mps": args.speed,
        "rough_drive_gain": ROUGH_DRIVE_GAIN,
        "target_ride_height_m": RIDE_HEIGHT,
        "boundary_radius_m": BOUNDARY_RADIUS,
        "n_stones": N_STONES,
        "max_stone_size_m": MAX_STONE_SIZE,
        "n_pits": len(PITS),
        "pit_depth_m": PIT_DEPTH,
        "n_ledges": len(LEDGES),
        "ledge_height_m": LEDGE_HEIGHT,
        "camera_distance_m": 1.25,
        "metrics_warmup_s": round(warmup * dt, 2),
        "sim_duration_s": round(t_sim, 2),
        "video_duration_s": round(total_steps / (video_stride * fps), 2),
        "runs": summary,
    }
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nRender complete in {time.time() - start_wall:.0f}s")
    print(f"Video: {video_path} ({results['video_duration_s']}s, "
          f"{video_path.stat().st_size / 1e6:.1f} MB)")
    for tag, block in summary.items():
        print(f"  {tag:18s} rolled {block['distance_rolled_m']:6.1f} m, "
              f"mean speed {block['mean_speed_mps']:.2f} m/s, "
              f"stalled {block['stalled_time_pct']}%")
    print(f"Results JSON: {json_path}")


if __name__ == "__main__":
    main()
