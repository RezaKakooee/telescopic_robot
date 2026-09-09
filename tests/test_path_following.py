"""Unit and physics tests for closed-loop ground path following (follow_path).

Verifies:
1. Sub-skill delegation: dynamically selects stop, move, curve, or turn based on path geometry.
2. Curvature-adaptive speed scaling into turns.
3. Closed-loop path tracking in simulation with bounded cross-track error.
"""
import unittest
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario
from skills import execute_skill


class TestPathFollowingSkill(unittest.TestCase):
    def setUp(self):
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        self.dirs_body = np.random.randn(60, 3)
        self.dirs_body /= np.linalg.norm(self.dirs_body, axis=1, keepdims=True)
        self.max_extend = 0.16

    def test_subskill_delegation_logic(self):
        """Verify dynamic delegation to stop, move, curve, and turn."""
        # 1. Goal arrival -> should dispatch `stop`
        path_short = np.array([[0.0, 0.0], [0.20, 0.0]])
        ball_at_goal = np.array([0.15, 0.0])
        _, meta = execute_skill(
            "follow_path", self.quat, self.dirs_body, self.max_extend,
            ball_xy=ball_at_goal, path_pts=path_short, goal_tolerance=0.30,
            return_metadata=True,
        )
        self.assertEqual(meta["sub_skill"], "stop", "Expected delegation to `stop` when at goal")

        # 2. Straightaway -> should dispatch `move`
        path_straight = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
        ball_straight = np.array([0.5, 0.0])
        _, meta = execute_skill(
            "follow_path", self.quat, self.dirs_body, self.max_extend,
            ball_xy=ball_straight, path_pts=path_straight, lin_vel=np.array([1.0, 0.0]),
            return_metadata=True,
        )
        self.assertEqual(meta["sub_skill"], "move", "Expected delegation to `move` on straight path")
        self.assertAlmostEqual(meta["curvature"], 0.0, places=2)

        # 3. Smooth arc / bend -> should dispatch `curve`
        # Circular arc of radius R = 2.0 m
        angles = np.linspace(0, np.pi / 2, 20)
        path_curved = np.stack([2.0 * np.sin(angles), 2.0 * (1.0 - np.cos(angles))], axis=1)
        ball_arc = np.array([path_curved[2, 0], path_curved[2, 1]])
        _, meta = execute_skill(
            "follow_path", self.quat, self.dirs_body, self.max_extend,
            ball_xy=ball_arc, path_pts=path_curved, lin_vel=np.array([1.0, 0.2]),
            lookahead=0.80, curve_threshold=0.15, return_metadata=True,
        )
        self.assertEqual(meta["sub_skill"], "curve", "Expected delegation to `curve` on curved path")
        self.assertGreater(abs(meta["curvature"]), 0.15, "Curvature should exceed curve threshold")

        # 4. Sharp 90-degree corner -> should dispatch `turn`
        path_corner = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
        ball_before_corner = np.array([0.80, 0.0])
        _, meta = execute_skill(
            "follow_path", self.quat, self.dirs_body, self.max_extend,
            ball_xy=ball_before_corner, path_pts=path_corner,
            lin_vel=np.array([1.0, 0.0]), lookahead=0.70,
            turn_angle_threshold_deg=25.0, return_metadata=True,
        )
        self.assertEqual(meta["sub_skill"], "turn", "Expected delegation to `turn` on sharp corner")
        print("✅ Sub-skill delegation logic verified (stop, move, curve, turn)")

    def test_closed_loop_path_tracking_sim(self):
        """Simulate RoboBall actively tracking a winding ground path."""
        cfg = load_config("configs/rl/config.yaml")
        cfg.camera.enabled = False

        # Winding S-curve path
        s_vals = np.linspace(0, 6.0, 61)
        # y = 0.5 * sin(0.8 * x)
        pts = np.stack([s_vals, 0.45 * np.sin(0.8 * s_vals)], axis=1).astype(np.float32)

        scenario = Scenario(
            kind="goal",
            name="path_tracking_test",
            spawn_xy=np.array([0.0, 0.0], dtype=np.float32),
            goal=pts[-1],
            path_pts=pts,
            markers=np.empty((0, 2), dtype=np.float32),
            path_length=6.0,
        )

        env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=1000)
        env.reset(seed=42)

        # Settle
        for _ in range(40):
            env.step(execute_skill("jump_up", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, phase="stand"))

        cross_track_errors = []
        sub_skills_called = set()

        for step in range(240):
            pos = env.data.qpos[:2].copy()
            vel = env.data.qvel[:3].copy()
            quat = env.data.qpos[3:7].copy()

            targets, meta = execute_skill(
                "follow_path", quat, env.dirs_body, env.max_extend,
                ball_xy=pos, path_pts=pts, lin_vel=vel,
                lookahead=0.75, speed=1.1, return_metadata=True,
            )
            env.step(targets)

            cross_track_errors.append(abs(meta["cross_track_error"]))
            sub_skills_called.add(meta["sub_skill"])

        end_pos = env.data.qpos[:2].copy()
        mean_cte = float(np.mean(cross_track_errors))
        max_cte = float(np.max(cross_track_errors))
        print(f"Tracking run: end_x={end_pos[0]:.2f}m, mean CTE={mean_cte*100:.1f}cm, max CTE={max_cte*100:.1f}cm, sub-skills={sub_skills_called}")

        # Check forward progress and tracking precision
        self.assertGreater(end_pos[0], 1.80, f"Insufficient progress along path: x={end_pos[0]:.2f}")
        self.assertLess(mean_cte, 0.12, f"Mean cross-track error too high: {mean_cte*100:.1f} cm")
        self.assertLess(max_cte, 0.25, f"Peak cross-track error too high: {max_cte*100:.1f} cm")
        print(f"✅ Closed-loop path tracking verified: end_x={end_pos[0]:.2f}m, mean CTE={mean_cte*100:.1f}cm, max CTE={max_cte*100:.1f}cm")


    def test_degenerate_paths_do_not_crash(self):
        """A one-point path used to raise TypeError from the stop branch."""
        import numpy as np
        from radial_sphere.geometry import fibonacci_sphere
        from skills.navigation import follow_path

        dirs = fibonacci_sphere(60).astype(np.float32)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        for pts in (np.array([[0.0, 0.0]]), np.zeros((0, 2))):
            out = follow_path(quat, dirs, 0.16, path_pts=pts,
                              ball_xy=np.array([0.0, 0.0]), min_offset=0.03)
            self.assertEqual(out.shape, (60,))

if __name__ == "__main__":
    unittest.main()
