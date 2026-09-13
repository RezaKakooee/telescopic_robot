"""Tests for the RL primitive set.

Two kinds of check. The contract tests are cheap and cover every primitive:
valid targets, bounded parameters, a stable action layout. The physics tests
cost a few seconds each and assert that a primitive does the thing its name
claims, because a primitive that returns legal targets and moves the robot the
wrong way would pass every contract test.
"""

import unittest

import numpy as np

import skills_rl as S
from skills_rl.base import Param, RobotState


def fake_state(max_extend=0.16, **kw):
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(60, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return RobotState(quat=np.array([1.0, 0.0, 0.0, 0.0]), dirs_body=dirs,
                      max_extend=max_extend, **kw)


class ContractTests(unittest.TestCase):
    def test_every_primitive_returns_legal_targets(self):
        state = fake_state(lin_vel=np.array([0.9, 0.0]), core_z=0.20, core_vz=0.0)
        for name, mod in S.PRIMITIVES.items():
            with self.subTest(skill=name):
                t = mod.act(state, **mod.SPEC.scale(np.zeros(mod.SPEC.n_params)))
                self.assertEqual(t.shape, (60,))
                self.assertTrue(np.all(np.isfinite(t)))
                self.assertTrue(np.all(t >= -1e-6))
                self.assertTrue(np.all(t <= state.max_extend + 1e-6))

    def test_extreme_parameters_stay_legal(self):
        """A policy will emit the corners of the box on its first update."""
        state = fake_state(lin_vel=np.array([2.0, -1.0]), core_z=0.20, core_vz=-0.4)
        for name, mod in S.PRIMITIVES.items():
            n = mod.SPEC.n_params
            for corner in (-np.ones(n), np.ones(n)):
                with self.subTest(skill=name, corner=corner[0]):
                    t = mod.act(state, **mod.SPEC.scale(corner))
                    self.assertTrue(np.all(np.isfinite(t)))
                    self.assertTrue(np.all(t >= -1e-6))
                    self.assertTrue(np.all(t <= state.max_extend + 1e-6))

    def test_missing_state_does_not_crash(self):
        """Only `brake` and `conform` read optional state; both must cope."""
        bare = fake_state()
        for name in ("brake", "conform"):
            with self.subTest(skill=name):
                mod = S.PRIMITIVES[name]
                t = mod.act(bare, **mod.SPEC.scale(np.zeros(mod.SPEC.n_params)))
                self.assertEqual(t.shape, (60,))

    def test_param_scaling_round_trips(self):
        p = Param("speed", 0.33, 2.8)
        for v in (0.33, 1.0, 2.8):
            self.assertAlmostEqual(p.scale(p.unit(v)), v, places=9)
        self.assertEqual(p.scale(-5.0), 0.33)      # clipped, not extrapolated
        self.assertEqual(p.scale(+5.0), 2.8)

    def test_action_layout_is_stable(self):
        """Reordering PRIMITIVES would invalidate every trained policy."""
        self.assertEqual(S.SKILL_NAMES,
                         ("drive", "brake", "stance", "thrust", "tuck",
                          "brace", "conform"))
        self.assertEqual(S.action_space().shape,
                         (len(S.PRIMITIVES) + S.MAX_PARAMS,))

    def test_decode_picks_the_argmax_skill(self):
        for i, name in enumerate(S.SKILL_NAMES):
            a = np.full(len(S.SKILL_NAMES) + S.MAX_PARAMS, -1.0)
            a[i] = 1.0
            with self.subTest(skill=name):
                self.assertEqual(S.decode(a)[0], name)

    def test_decode_returns_arguments_the_primitive_accepts(self):
        a = np.zeros(len(S.SKILL_NAMES) + S.MAX_PARAMS)
        state = fake_state(lin_vel=np.array([0.5, 0.0]), core_z=0.2, core_vz=0.0)
        for i, name in enumerate(S.SKILL_NAMES):
            a[:len(S.SKILL_NAMES)] = -1.0
            a[i] = 1.0
            with self.subTest(skill=name):
                got, kwargs = S.decode(a)
                self.assertEqual(got, name)
                S.act(got, state, **kwargs)          # must not raise

    def test_max_params_covers_every_primitive(self):
        for name, mod in S.PRIMITIVES.items():
            with self.subTest(skill=name):
                self.assertLessEqual(mod.SPEC.n_params, S.MAX_PARAMS)


class PhysicsTests(unittest.TestCase):
    """What the primitives do to a real ball on real ground."""

    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault("MUJOCO_GL", "egl")
        from radial_sphere.config import load_config
        from radial_sphere.mujoco_env import MujocoRadialSphereEnv
        from radial_sphere.scenario import generate_scenario
        cls.cfg = load_config("configs/rl/config.yaml")
        cls.cfg.camera.enabled = False
        cls.scenario = generate_scenario("path", cls.cfg, seed=0)
        cls.EnvCls = MujocoRadialSphereEnv

    def run_skill(self, name, steps=120, settle=25, **params):
        env = self.EnvCls(self.cfg, scenario=self.scenario, randomize=False,
                          max_steps=steps + settle + 10)
        env.reset(seed=0)
        for _ in range(settle):
            env.step(np.full(60, 0.025, dtype=np.float32))
        p0 = env.data.qpos[:3].copy()
        peak = float(p0[2])
        for _ in range(steps):
            st = RobotState(quat=env.data.qpos[3:7].copy(),
                            dirs_body=env.dirs_body, max_extend=env.max_extend,
                            lin_vel=env.data.qvel[:3].copy(),
                            core_z=float(env.data.qpos[2]),
                            core_vz=float(env.data.qvel[2]))
            env.step(S.act(name, st, **params))
            peak = max(peak, float(env.data.qpos[2]))
        out = dict(dxy=(env.data.qpos[:2] - p0[:2]).copy(),
                   speed=float(np.linalg.norm(env.data.qvel[:2])),
                   peak_rise=peak - float(p0[2]))
        env.close()
        return out

    def test_drive_travels_along_the_commanded_heading(self):
        for az in (0.0, np.pi / 2):
            with self.subTest(azimuth=az):
                r = self.run_skill("drive", azimuth=az, speed=1.2, flank=0.0)
                want = np.array([np.cos(az), np.sin(az)])
                moved = float(np.linalg.norm(r["dxy"]))
                self.assertGreater(moved, 0.30, "drive did not travel")
                # Mostly along the command, not sideways.
                self.assertGreater(float(r["dxy"] @ want) / moved, 0.7)

    def test_drive_speed_parameter_orders_the_result(self):
        slow = self.run_skill("drive", azimuth=0.0, speed=0.5, flank=0.0)
        fast = self.run_skill("drive", azimuth=0.0, speed=2.4, flank=0.0)
        self.assertGreater(float(np.linalg.norm(fast["dxy"])),
                           float(np.linalg.norm(slow["dxy"])))

    def test_negative_speed_reverses(self):
        fwd = self.run_skill("drive", azimuth=0.0, speed=1.2, flank=0.0)
        rev = self.run_skill("drive", azimuth=0.0, speed=-1.2, flank=0.0)
        self.assertGreater(fwd["dxy"][0], 0.2)
        self.assertLess(rev["dxy"][0], -0.2)

    def test_thrust_leaves_the_ground(self):
        r = self.run_skill("thrust", steps=40, azimuth=0.0,
                           elevation=-np.pi / 2, power=1.0, spread=0.55)
        self.assertGreater(r["peak_rise"], 0.05, "thrust produced no lift")

    def test_stance_holds_still(self):
        r = self.run_skill("stance", steps=120, height=0.045)
        self.assertLess(float(np.linalg.norm(r["dxy"])), 0.10)
        self.assertLess(r["speed"], 0.15)


if __name__ == "__main__":
    unittest.main()
