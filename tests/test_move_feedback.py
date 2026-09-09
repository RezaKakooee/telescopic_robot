"""Physics regressions for calibrated forward movement and runner routing."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import unittest
from radial_sphere.config import load_config
from radial_sphere.scenario import generate_scenario
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from skills.runner import run_skill, skill_targets
from skills.locomotion import move_forward


def make_env():
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range = [0., 0.]
    cfg.scenario.goal.y_range = [-400., -400.]
    robot = MujocoRadialSphereEnv(
        cfg, scenario=generate_scenario("goal", cfg, seed=42),
        randomize=False, max_steps=10000,
    )
    robot.reset(seed=42)
    return robot


def check_forward_tracking(env, speed, angle):
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    velocity, drift = [], []
    origin = env.data.qpos[:2].copy()

    def record(robot, step):
        velocity.append(float(robot.data.qvel[:2] @ direction))
        drift.append(float((robot.data.qpos[:2] - origin) @ normal))

    stats = run_skill(env, "move", steps=600, d_hat=direction,
                      speed=speed, on_frame=record)
    mean_speed = float(np.mean(velocity[-200:]))
    print(f"speed={speed:.2f}, angle={angle:.2f}: achieved={mean_speed:.3f}, "
          f"drift={drift[-1]:.3f}, max_drift={max(abs(np.array(drift))):.3f}")
    assert abs(mean_speed - speed) < 0.08
    assert abs(drift[-1]) < 0.10
    assert max(abs(np.array(drift))) < 0.15
    assert stats["displacement"][:2] @ direction > 3 * speed


def check_runner_routes_state_and_explicit_gain_bypasses_feedback(env):
    direction = np.array([1., 0.])
    routed = skill_targets(env, "move_forward", d_hat=direction)
    direct = move_forward(env.data.qpos[3:7], env.dirs_body, env.max_extend,
                          direction, lin_vel=env.data.qvel[:2],
                          rod_mechanism="multi_stage")
    np.testing.assert_allclose(routed, direct)
    legacy = move_forward(env.data.qpos[3:7], env.dirs_body, env.max_extend,
                          direction, back_gain=1.6)
    explicit = move_forward(env.data.qpos[3:7], env.dirs_body, env.max_extend,
                            direction, back_gain=1.6, lin_vel=[4., 1.],
                            cross_track_error=2., rod_mechanism="multi_stage")
    np.testing.assert_array_equal(legacy, explicit)


class TestMoveFeedback(unittest.TestCase):
    def test_signed_speed_and_zero_stop(self):
        env = make_env()
        try:
            for speed in [1.2, -1.2, 0., 1.2]:
                with self.subTest(speed=speed):
                    samples = []
                    run_skill(env, "move", steps=600, d_hat=[1., 0.], speed=speed,
                              on_frame=lambda e, step: samples.append(e.data.qvel[:2].copy()))
                    avg = np.mean(samples[-100:], axis=0)
                    print(f"signed move request={speed:+.2f}, measured vx={avg[0]:+.3f}")
                    self.assertLess(abs(avg[0] - speed), .10)
                    if speed == 0:
                        self.assertLess(np.linalg.norm(avg), .05)
                    else:
                        self.assertEqual(np.sign(avg[0]), np.sign(speed))
        finally:
            env.close()

    def test_negative_speed_rotates_the_travel_reference(self):
        from skills.locomotion import move, stop
        env = make_env()
        try:
            q = env.data.qpos[3:7]
            actual = move(q, env.dirs_body, env.max_extend, [0., 1.], speed=-1.2,
                          lin_vel=[.1, -.8], cross_track_error=.2)
            equivalent = move(q, env.dirs_body, env.max_extend, [0., -1.], speed=1.2,
                              lin_vel=[.1, -.8], cross_track_error=.2)
            np.testing.assert_array_equal(actual, equivalent)
            np.testing.assert_array_equal(
                move(q, env.dirs_body, env.max_extend, [1., 0.], speed=0., lin_vel=[1., .1]),
                stop(q, env.dirs_body, env.max_extend, lin_vel=[1., .1]))
            for speed in [float("nan"), float("inf"), -float("inf")]:
                with self.assertRaises(ValueError):
                    move(q, env.dirs_body, env.max_extend, [1., 0.], speed=speed)
        finally:
            env.close()

    def test_one_skill_changes_speed_without_reset(self):
        env = make_env()
        try:
            for speed in [1.2, 2.0, 1.2, 0.6, 1.2]:
                with self.subTest(speed=speed):
                    velocities = []
                    run_skill(env, "move", steps=500, d_hat=[1., 0.], speed=speed,
                              on_frame=lambda e, step: velocities.append(e.data.qvel[0]))
                    measured = float(np.mean(velocities[-200:]))
                    print(f"move speed request={speed:.2f}, measured={measured:.3f}")
                    self.assertLess(abs(measured - speed), .10)
        finally:
            env.close()

    def test_signed_degree_turns(self):
        for angle_deg in [30., -30.]:
            with self.subTest(angle_deg=angle_deg):
                env = make_env()
                try:
                    run_skill(env, "move_forward", steps=200, d_hat=[1., 0.])
                    samples = []
                    run_skill(env, "turn", steps=400, d_hat=[1., 0.], angle_deg=angle_deg,
                              on_frame=lambda e, step: samples.append(e.data.qvel[:2].copy()))
                    velocity = np.mean(samples[-100:], axis=0)
                    achieved = -np.degrees(np.arctan2(velocity[1], velocity[0]))
                    print(f"turn request={angle_deg:+.0f} degrees, measured={achieved:+.2f}")
                    self.assertLess(abs(achieved - angle_deg), 8)
                    self.assertEqual(np.sign(achieved), np.sign(angle_deg))
                finally:
                    env.close()

    def test_forward_tracking(self):
        for speed, angle in [(0.6, 0.), (1.2, 0.), (2.0, 0.),
                             (1.2, np.pi / 2), (1.2, -np.pi / 4)]:
            with self.subTest(speed=speed, angle=angle):
                env = make_env()
                try:
                    check_forward_tracking(env, speed, angle)
                finally:
                    env.close()

    def test_routing_and_legacy_gain(self):
        env = make_env()
        try:
            check_runner_routes_state_and_explicit_gain_bypasses_feedback(env)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
