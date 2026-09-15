"""Drive the scripted expert through the inspection courses and record videos.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/vla/run_inspection_oracle.py [course ...]

Writes one chase-camera video per course plus a summary JSON into a
timestamped run dir under storage_local/.
"""
from __future__ import annotations

import json
import sys

import cv2
import imageio
import mujoco
import numpy as np

from radial_sphere.config import load_config_cli
from radial_sphere.inspection_oracle import InspectionOracle
from radial_sphere.inspection_scenarios import INSPECTION_SCENARIOS
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from radial_sphere.snapshot import make_run_dir

FPS = 25
MAP_PX = 230       # minimap size in pixels


class MiniMap:
    """Top-down sketch of the course: walls, obstacles, the route, progress and the ball."""

    def __init__(self, sc, px: int = MAP_PX):
        from radial_sphere.terrain import Gap, Pipe, Staircase, Step, StoneField, rows
        pts = np.asarray(sc.path_pts, dtype=float)
        walls = np.asarray(sc.walls, dtype=float).reshape(-1, 4) if sc.walls is not None else np.zeros((0, 4))
        allx = np.concatenate([pts[:, 0], walls[:, [0, 2]].ravel()])
        ally = np.concatenate([pts[:, 1], walls[:, [1, 3]].ravel()])
        self.lo = np.array([allx.min(), ally.min()]) - 0.6
        self.hi = np.array([allx.max(), ally.max()]) + 0.6
        self.scale = (px - 8) / float(max(self.hi - self.lo))
        self.px = px
        self.path = pts
        self.bg = np.array((28, 30, 36), dtype=np.uint8)
        base = np.full((px, px, 3), self.bg, dtype=np.uint8)
        for step in rows(Step, getattr(sc, "steps", None)):
            self._rect(base, step.x, step.y, step.half_x, step.half_y, (235, 150, 60))       # slabs, beams: orange
        for gap in rows(Gap, getattr(sc, "gaps", None)):
            self._rect(base, gap.x, gap.y, gap.half_x, gap.half_y, (40, 40, 40))
        for field in rows(StoneField, getattr(sc, "stones", None)):
            self._rect(base, field.x, field.y, field.half_x, field.half_y, (90, 90, 90))
        for pipe in rows(Pipe, getattr(sc, "pipes", None)):
            yaw = np.radians(float(pipe.yaw_deg or 0.0))
            a = np.array([pipe.x, pipe.y]); b = a + pipe.length * np.array([np.cos(yaw), np.sin(yaw)])
            cv2.line(base, self._p(a), self._p(b), (80, 210, 230), max(2, int(2 * pipe.inner_radius * self.scale)))   # pipes: cyan
        for flight in rows(Staircase, getattr(sc, "staircases", None)):
            yaw = np.radians(float(flight.yaw_deg)); d = np.array([np.cos(yaw), np.sin(yaw)]); n = np.array([-d[1], d[0]])
            for i in range(int(flight.n_steps)):
                r = np.array([flight.x, flight.y]) + d * i * float(flight.run)
                cv2.line(base, self._p(r - n * flight.width / 2), self._p(r + n * flight.width / 2), (60, 160, 235), 2)   # stair risers: blue
        if sc.obstacles is not None:
            for x, y, r in np.asarray(sc.obstacles).reshape(-1, 3):
                cv2.circle(base, self._p((x, y)), max(2, int(r * self.scale)), (120, 120, 140), -1)
        for x1, y1, x2, y2 in walls:
            cv2.line(base, self._p((x1, y1)), self._p((x2, y2)), (230, 230, 230), 2)
        # the full route, thin
        cv2.polylines(base, [np.array([self._p(q) for q in pts], dtype=np.int32)], False, (150, 170, 255), 1, cv2.LINE_AA)   # route ahead: light blue
        cv2.circle(base, self._p(sc.goal), 5, (60, 220, 60), -1)
        cv2.circle(base, self._p(sc.spawn_xy), 4, (255, 255, 255), 1)
        self.base = base

    def _p(self, xy):
        x = 4 + (float(xy[0]) - self.lo[0]) * self.scale
        y = self.px - 4 - (float(xy[1]) - self.lo[1]) * self.scale      # y up
        return int(round(x)), int(round(y))

    def _rect(self, img, cx, cy, hx, hy, color):
        cv2.rectangle(img, self._p((cx - hx, cy - hy)), self._p((cx + hx, cy + hy)), color, -1)

    def draw(self, pos, path_idx: int) -> np.ndarray:
        img = self.base.copy()
        if path_idx > 1:      # the part of the route already covered, thick
            done = np.array([self._p(q) for q in self.path[: path_idx + 1]], dtype=np.int32)
            cv2.polylines(img, [done], False, (255, 210, 80), 2, cv2.LINE_AA)
        cv2.circle(img, self._p(pos), 5, (255, 70, 50), -1)      # ball: red
        cv2.circle(img, self._p(pos), 5, (255, 255, 255), 1)
        return img


def chase_frame(env, renderer, hud: str, minimap: MiniMap | None = None) -> np.ndarray:
    core = env.env
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = core.core_body_id
    cam.distance, cam.elevation = 3.2, -48.0     # steep enough to look over 1 m walls
    # Look the way the ball is going (a tour crosses itself, so the nearest
    # route point is not a safe guide); hold the last heading when it is slow.
    v = core.data.qvel[:2]
    smooth = getattr(core, "_cam_az", None)
    if np.linalg.norm(v) > 0.3:
        yaw = float(np.degrees(np.arctan2(v[1], v[0])))
    else:
        pts = env.scenario.path_pts
        yaw = smooth if smooth is not None else float(np.degrees(np.arctan2(*(pts[10] - pts[0])[::-1])))
    if smooth is None:
        smooth = yaw
    delta = (yaw - smooth + 180.0) % 360.0 - 180.0
    core._cam_az = smooth + 0.12 * delta
    cam.azimuth = core._cam_az
    renderer.update_scene(core.data, camera=cam)
    img = renderer.render().copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (12, 16, 24), -1)
    cv2.putText(img, hud, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 235, 245), 1, cv2.LINE_AA)
    if minimap is not None:
        tracker = env.waypoint_tracker
        idx = getattr(tracker, "_last_idx", None)          # read only: never touch the tracker state here
        if idx is None:
            pts = env.scenario.path_pts
            idx = int(np.argmin(np.linalg.norm(pts - core.data.qpos[:2], axis=1)))
        m = minimap.draw(core.data.qpos[:2], idx)
        h, w = m.shape[:2]
        y0, x0 = img.shape[0] - h - 8, img.shape[1] - w - 8
        roi = img[y0: y0 + h, x0: x0 + w]
        # Transparent background: only the drawn features are blended in.
        drawn = np.any(m != minimap.bg, axis=2)
        roi[drawn] = (0.15 * roi[drawn] + 0.85 * m[drawn]).astype(np.uint8)
        roi[~drawn] = (0.7 * roi[~drawn]).astype(np.uint8)          # slight darkening so lines stay readable
    return img


def run_course(kind: str, cfg, out_dir, max_macro_steps: int = 900, seed: int = 0, tour: bool = False, video: bool = True) -> dict:
    sc = generate_scenario(kind, cfg, seed=seed, tour=tour)
    max_macro_steps = max(max_macro_steps, int(sc.path_length * 14))      # ~1 m/s at 10 macro steps per second
    env = SkillArbitrationEnv(cfg, scenario=sc, seed=seed, max_steps=4000)
    env.reset(seed=seed)
    oracle = InspectionOracle(sc)
    renderer = mujoco.Renderer(env.env.model, height=480, width=640)
    minimap = MiniMap(sc)
    frames, state = [], {"skill": "move", "why": "", "hits": 0, "step": 0}
    next_t = [0.0]

    def on_control_step(e):
        now = float(e.env.data.time)
        while now + 1e-9 >= next_t[0]:
            hud = (f"{kind} | step {state['step']:3d} | t {now:5.1f}s | {state['skill']:26s} | "
                   f"{state['why'][:28]:28s} | hits {state['hits']}")
            frames.append(chase_frame(e, renderer, hud, minimap))
            next_t[0] += 1.0 / FPS

    if video:
        env.on_control_step = on_control_step
    hits = 0
    info = {}
    skills_used = {}
    for step in range(max_macro_steps):
        name, _, why = oracle.select(env.env.data.qpos[:3])
        state.update(skill=name, why=why, step=step)
        act = np.full(len(env.skill_names), -1.0, dtype=np.float32)
        act[env.skill_names.index(name)] = 1.0
        _, _, term, trunc, info = env.step(act)
        hits += int(info.get("obstacle_hit", 0))
        state["hits"] = hits
        skills_used[name] = skills_used.get(name, 0) + 1
        if term or trunc:
            break
    success = bool(info.get("success", False))
    final_dist = float(np.linalg.norm(env.env.data.qpos[:2] - np.asarray(sc.goal)[:2]))
    video = out_dir / f"{kind}_{'success' if success else 'fail'}_hits{hits}.mp4"
    if frames:
        imageio.mimsave(str(video), frames, fps=FPS)
    env.close()
    result = {"course": kind, "success": success, "clean": success and hits == 0, "hits": hits,
              "steps": step + 1, "final_dist": round(final_dist, 2), "path_length": round(sc.path_length, 1),
              "stalled": bool(info.get("stalled", False)), "skills": skills_used,
              "stations": [(s.kind, round(s.s_start, 1), s.label) for s in oracle.stations], "video": video.name}
    print(f"{kind:28s} success={success!s:5s} hits={hits:3d} steps={step + 1:4d} dist={final_dist:5.2f} "
          f"stalled={result['stalled']} skills={skills_used}")
    return result


def main():
    cfg = load_config_cli(name="playground_parkour_skills")
    out = make_run_dir(build_run_id("run_inspection_oracle", tag="tour" if "--tour" in sys.argv else "short"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tour = "--tour" in sys.argv
    video = "--no-video" not in sys.argv
    seed = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--seed=")), 0)
    wanted = args or list(INSPECTION_SCENARIOS)
    results = [run_course(k, cfg, out, tour=tour, seed=seed, video=video) for k in wanted]
    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    n_ok = sum(r["success"] for r in results)
    n_clean = sum(r["clean"] for r in results)
    print(f"\n{n_ok}/{len(results)} reached the goal, {n_clean} of them without a hit. Saved to {out}")


if __name__ == "__main__":
    main()
