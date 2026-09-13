"""Plan a route past pillars and drive it, choosing a skill each step.

Nothing here decides how to move. `skills.high_level.go_to_goal` answers with
a skill name and its arguments, this loop runs that skill, and the next step
asks again. The video is two panes: the robot from behind, and a plan view
that draws what the planner is actually thinking -- the pillars it inflated,
the polyline it chose, and how far the robot has drifted off it.

    python scripts/run_demo.py demo=goal_seeking
    python scripts/run_demo.py demo=goal_seeking video=true
    python demos/goal_seeking/runner.py knobs.seed=3
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from radial_sphere.config import demo_config, load_config
from radial_sphere.demo import Recorder
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.overlay import annotate
from radial_sphere.scenario import generate_scenario
from skills import execute_skill
from skills.high_level import go_to_goal

PANE = (640, 480)

#: One colour per skill the planner can name, so the plan view and the badge
#: agree without a legend.
SKILL_COLOUR = {
    "follow_path": (56, 189, 248),
    "climb_stairs": (192, 132, 252),
    "stop": (74, 222, 128),
}


class PlanView:
    """Top-down drawing of the map, the planned route and the path taken.

    The world-to-pixel transform is fixed once from the arena's extent, so the
    route does not swim about between frames.
    """

    def __init__(self, obstacles, spawn, goal, clearance, size=PANE, pad=0.8):
        self.w, self.h = size
        self.obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 3)
        self.clearance = float(clearance)
        self.goal = np.asarray(goal, dtype=float)[:2]

        pts = [np.asarray(spawn, dtype=float)[:2], self.goal]
        for cx, cy, r in self.obstacles:
            pts += [np.array([cx - r, cy - r]), np.array([cx + r, cy + r])]
        pts = np.asarray(pts)
        lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
        # One scale for both axes, so circles stay circles.
        span = float(max(hi - lo))
        self.scale = min(self.w, self.h) / max(span, 1e-6)
        self.origin = 0.5 * (lo + hi)

    def _px(self, p):
        """World xy to pixel. World +y is up, so the y axis flips."""
        d = (np.asarray(p, dtype=float)[:2] - self.origin) * self.scale
        return (self.w / 2 + d[0], self.h / 2 - d[1])

    def _dot(self, draw, p, r_px, fill, outline=None, width=2):
        x, y = self._px(p)
        draw.ellipse([x - r_px, y - r_px, x + r_px, y + r_px],
                     fill=fill, outline=outline, width=width)

    def render(self, ball_xy, route, trail, skill):
        img = Image.new("RGB", (self.w, self.h), (16, 20, 28))
        draw = ImageDraw.Draw(img)
        colour = SKILL_COLOUR.get(skill, (148, 163, 184))

        for cx, cy, r in self.obstacles:
            # The clearance ring is what the planner routes against; the solid
            # disc is the pillar the robot would actually hit.
            self._dot(draw, (cx, cy), (r + self.clearance) * self.scale,
                      fill=None, outline=(71, 85, 105), width=1)
            self._dot(draw, (cx, cy), r * self.scale, fill=(100, 116, 139))

        if len(trail) > 1:
            draw.line([self._px(p) for p in trail], fill=(71, 85, 105), width=2)

        route = np.asarray(route, dtype=float).reshape(-1, 2)
        if len(route) > 1:
            draw.line([self._px(p) for p in route], fill=colour, width=3)
            for p in route[1:-1]:
                self._dot(draw, p, 4, fill=(15, 23, 42), outline=colour, width=2)

        gx, gy = self._px(self.goal)
        draw.line([(gx - 9, gy - 9), (gx + 9, gy + 9)], fill=(250, 204, 21), width=3)
        draw.line([(gx - 9, gy + 9), (gx + 9, gy - 9)], fill=(250, 204, 21), width=3)
        self._dot(draw, ball_xy, 7, fill=(248, 250, 252))
        return np.asarray(img)


def pillar_gap(obstacles, ball_xy):
    """Distance from the ball centre to the nearest pillar surface."""
    if not len(obstacles):
        return float("inf")
    obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 3)
    return float((np.linalg.norm(obstacles[:, :2] - np.asarray(ball_xy)[:2], axis=1)
                  - obstacles[:, 2]).min())


def main():
    args = demo_config("goal_seeking").knobs

    cfg = load_config(args.config)
    scenario = generate_scenario("obstacle", cfg, seed=args.seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False,
                                max_steps=args.max_steps + 200)
    env.reset(seed=args.seed)
    for _ in range(25):
        env.step(np.full(len(env.dirs_body), 0.025, dtype=np.float32))

    goal = np.asarray(scenario.goal, dtype=float)[:2]
    obstacles = np.asarray(scenario.obstacles, dtype=float).reshape(-1, 3)
    start = env.data.qpos[:2].copy()

    recorder = Recorder(env, "run_goal_seeking", tag=f"seed{args.seed}",
                        enabled=args.video, fps=args.fps, every=1,
                        out_name="goal_seeking")
    plan_view = PlanView(obstacles, start, goal, args.clearance)

    def chase_frame():
        if env.renderer is None:
            env.render(camera_name="fixed_angle_close_3d")
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = env.core_body_id
        cam.distance, cam.elevation, cam.azimuth = 2.4, -28.0, 135.0
        env.renderer.update_scene(env.data, camera=cam)
        return env.renderer.render()

    trail = [start.copy()]
    min_gap = float("inf")
    max_detours = 0
    reasons, skills = [], {}
    arrived = False
    arrived_step = 0
    step = 0
    # Keep recording briefly after arriving. Cutting on the same frame the
    # brake finishes leaves no time to see that it stopped on the goal.
    hold_left = int(args.hold_steps)

    print(f"goal seeking: {len(obstacles)} pillars, goal at "
          f"({goal[0]:.2f}, {goal[1]:.2f}), seed {args.seed}")

    for step in range(args.max_steps):
        pos = env.data.qpos[:3].copy()
        quat = env.data.qpos[3:7].copy()
        vel = env.data.qvel[:3].copy()

        cmd = go_to_goal(
            ball_xy=pos[:2], goal_xy=goal, lin_vel=vel[:2],
            obstacles=obstacles, steps=scenario.steps,
            speed=args.speed, clearance=args.clearance,
            goal_tolerance=args.goal_tolerance,
        )
        skills[cmd.skill] = skills.get(cmd.skill, 0) + 1
        max_detours = max(max_detours, cmd.meta.get("n_detours", 0))
        if not reasons or reasons[-1] != cmd.reason:
            reasons.append(cmd.reason)
            print(f"  t={step * 0.01:5.2f}s  {cmd.skill:<12} {cmd.reason}")

        env.step(execute_skill(cmd.skill, quat, env.dirs_body,
                               env.max_extend, **cmd.kwargs))

        now = env.data.qpos[:2].copy()
        min_gap = min(min_gap, pillar_gap(obstacles, now))
        if not len(trail) or float(np.linalg.norm(now - trail[-1])) > 0.02:
            trail.append(now)

        if recorder.enabled and step % args.frame_every == 0:
            dist = float(np.linalg.norm(goal - now))
            chase = annotate(
                chase_frame(), f"go_to_goal [{cmd.skill}]",
                [cmd.reason,
                 f"goal in {dist:5.2f} m   speed {float(np.linalg.norm(env.data.qvel[:2])):4.2f} m/s",
                 f"nearest pillar {pillar_gap(obstacles, now):5.2f} m",
                 f"t {step * 0.01:5.2f}s"])
            plan = plan_view.render(now, cmd.meta.get("route", [now, goal]),
                                    trail, cmd.skill)
            recorder.add(np.concatenate([chase, plan], axis=1))

        if cmd.skill == "stop" and float(np.linalg.norm(env.data.qvel[:2])) < 0.12:
            if not arrived:
                arrived = True
                arrived_step = step
            hold_left -= 1
            if hold_left <= 0:
                break

    end = env.data.qpos[:2].copy()
    miss = float(np.linalg.norm(end - goal))
    # The route clears every pillar by `clearance`; the robot does not, because
    # `follow_path` cuts corners. The gap below is the robot's, not the plan's.
    ok = arrived and miss <= args.goal_tolerance and min_gap > 0.0
    print(f"  route  : at most {max_detours} detour waypoint(s), "
          f"skills {skills}")
    print(f"  arrival: {arrived} at {(arrived_step if arrived else step) * 0.01:.1f}s, "
          f"{miss:.2f} m from the goal (tolerance {args.goal_tolerance:.2f})")
    print(f"  safety : closest pillar surface {min_gap:.2f} m "
          f"(planned clearance {args.clearance:.2f})")
    print(f"  final  : x {end[0]:+.2f} y {end[1]:+.2f} -> "
          f"{'SUCCESS' if ok else 'FAILED'}")
    recorder.close()
    if recorder.enabled:
        print(f"video: {recorder.path}  ({recorder.n_frames / args.fps:.1f}s)")
    env.close()


if __name__ == "__main__":
    main()
