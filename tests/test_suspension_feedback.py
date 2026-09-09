"""Regression checks for terrain sensing and continuous suspension commands."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import unittest
import mujoco
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.runner import skill_targets
from skills.suspension import SuspensionState, apply_suspension


class SuspensionFeedbackTests(unittest.TestCase):
    def test_command_rate_is_bounded_across_contact_switches(self):
        for dt in (.005, .01, .02):
            state = SuspensionState(targets=np.full(3, .025))
            previous = state.targets.copy()
            for step in range(80):
                target = np.full(3, .16 if step % 2 else .025)
                result, _ = apply_suspension(
                    target, np.array([-1., -.5, .5]), .16,
                    core_z=.20, core_vz=(-1)**step,
                    contact_forces=np.full(3, 80 if step % 2 else 0),
                    terrain_clearances=np.full(3, .08 if step % 2 else -.05),
                    state=state, dt=dt, max_target_speed=.45,
                )
                self.assertLessEqual(np.max(abs(result-previous)), .45*dt+2e-8)
                self.assertTrue(np.all((result >= .025-1e-8) & (result <= .16)))
                previous = result.copy()

    def test_unloaded_foot_does_not_invent_a_hole(self):
        kwargs = dict(core_z=.28, core_vz=0, contact_forces=np.zeros(2))
        flat, meta = apply_suspension(np.full(2,.06), -np.ones(2), .16,
                                     terrain_clearances=np.zeros(2), **kwargs)
        missing, _ = apply_suspension(np.full(2,.06), -np.ones(2), .16,
                                     terrain_clearances=np.full(2,np.nan), **kwargs)
        hole, _ = apply_suspension(np.full(2,.06), -np.ones(2), .16,
                                  terrain_clearances=np.full(2,.06), **kwargs)
        self.assertEqual(meta['delta_hole_mean'], 0)
        np.testing.assert_allclose(flat, missing)
        self.assertTrue(np.all(hole > flat))

    def test_contact_yield_is_limited(self):
        kwargs=dict(core_z=.28, core_vz=0, terrain_clearances=np.zeros(2))
        free,_=apply_suspension(np.full(2,.10), -np.ones(2), .16, **kwargs)
        loaded,_=apply_suspension(np.full(2,.10), -np.ones(2), .16,
                                  contact_forces=np.full(2,10000), **kwargs)
        self.assertTrue(np.all(free-loaded <= .012+1e-8))

    def make_env(self):
        cfg=load_config('configs/rl/standing_jump_showcase.yaml')
        cfg.camera.enabled=False
        scenario=generate_scenario('goal',cfg,seed=42)
        env=MujocoRadialSphereEnv(cfg,scenario=scenario,randomize=False,max_steps=100)
        env.reset(seed=42)
        self.addCleanup(env.close)
        return env

    def test_rays_ignore_all_robot_stages(self):
        env=self.make_env()
        env.data.qpos[2]=.22
        env.data.qpos[env.slide_qpos_adr]=.12
        mujoco.mj_forward(env.model,env.data)
        down=(env.dirs_body @ quat_to_rotmat(env.data.qpos[3:7]).T)[:,2]
        i=int(np.argmin(down))
        stage=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,'stage1_geom_0')
        self.assertIn(stage,env.robot_geom_ids)
        self.assertIn(stage,env.rod_geom_map)
        # The reading is a vertical terrain height, so it does not depend on
        # the rod angle. Raised floor reads negative, sunken floor positive.
        for floor_z in (0., .04, -.08):
            env.model.geom_pos[env.floor_geom_id,2]=floor_z
            mujoco.mj_forward(env.model,env.data)
            # Static world geoms cache their pose; set the ray-query transform
            # explicitly for this isolated sensor test (no dynamics are stepped).
            env.data.geom_xpos[env.floor_geom_id,2]=floor_z
            actual=env.get_terrain_clearances()[i]
            self.assertAlmostEqual(actual, -floor_z, places=5)

    def test_rays_ignore_terrain_the_rod_cannot_reach(self):
        env=self.make_env()
        env.data.qpos[2]=.22
        mujoco.mj_forward(env.model,env.data)
        down=(env.dirs_body @ quat_to_rotmat(env.data.qpos[3:7]).T)[:,2]
        reach=env.sphere_radius+env.max_extend+env.TERRAIN_RAY_MARGIN
        clear=env.get_terrain_clearances()
        for i,uz in enumerate(down):
            if uz>=-.15:
                self.assertEqual(clear[i],0.)
                continue
            # Flat floor: the hit sits at z=0, so any in-range rod reads zero
            # and every out-of-range rod reads NaN.
            if -.22/uz<=reach:
                self.assertAlmostEqual(float(clear[i]),0.,places=5)
            else:
                self.assertTrue(np.isnan(clear[i]))

    def test_runner_keeps_state_and_clears_it_after_reset(self):
        env=self.make_env()
        target=skill_targets(env,'traverse_rough_terrain',d_hat=[1.,0.])
        first=env._suspension_state
        env.step(target)
        skill_targets(env,'traverse_rough_terrain',step=1,d_hat=[1.,0.])
        self.assertIs(first,env._suspension_state)
        env.reset(seed=42)
        skill_targets(env,'traverse_rough_terrain',d_hat=[1.,0.])
        self.assertIsNot(first,env._suspension_state)

    def test_gains_object_matches_the_separate_keywords(self):
        """`suspension=` must be a pure repackaging of the nine keywords."""
        from skills.navigation import stay_in_boundary
        from skills.terrain_following import traverse_rough_terrain
        from skills.suspension import SuspensionGains
        from radial_sphere.geometry import fibonacci_sphere

        dirs = fibonacci_sphere(60).astype(np.float32)
        quat = np.array([0.94, 0.05, -0.12, 0.31])
        quat = quat / np.linalg.norm(quat)
        forces = np.linspace(0, 40, 60)
        clear = np.linspace(-0.08, 0.08, 60)
        spread = dict(target_ride_height=.23, suspension_kp=.8, suspension_kd=.2,
                      suspension_force_compliance=.0022, nominal_support_force=12.,
                      terrain_adaptation_gain=.9, hole_reach_gain=.055,
                      max_target_speed=.4, suspension_filter_time=.05)
        packed = SuspensionGains(target_ride_height=.23, kp=.8, kd=.2,
                                 force_compliance=.0022, nominal_support_force=12.,
                                 terrain_adaptation=.9, hole_reach=.055,
                                 max_target_speed=.4, filter_time=.05)
        shared = dict(core_z=.21, core_vz=-.05, contact_forces=forces,
                      terrain_clearances=clear, control_dt=.01)

        a = traverse_rough_terrain(quat, dirs, .16, d_hat=[1., 0.], speed=.72,
                                   **shared, **spread)
        b = traverse_rough_terrain(quat, dirs, .16, d_hat=[1., 0.], speed=.72,
                                   suspension=packed, **shared)
        np.testing.assert_allclose(a, b, atol=1e-12)

        boundary = dict(ball_xy=np.array([1.2, .4]), lin_vel=np.array([.3, .1]),
                        boundary_radius=3.4, speed=1.3, safety_margin=.7,
                        step_count=40, rough_terrain_gait=True, rough_drive_gain=2.0,
                        **shared)
        c = stay_in_boundary(quat, dirs, .16, **boundary, **spread)
        d = stay_in_boundary(quat, dirs, .16, **boundary, suspension=packed)
        np.testing.assert_allclose(c, d, atol=1e-12)

    def test_without_feedback_keeps_the_gait_and_drops_the_corrections(self):
        from skills.suspension import SuspensionGains
        gains = SuspensionGains(target_ride_height=.23, kp=.8, max_target_speed=.4)
        off = gains.without_feedback()
        self.assertEqual(off.target_ride_height, .23)
        self.assertEqual(off.max_target_speed, .4)
        for field in ("kp", "kd", "force_compliance", "terrain_adaptation", "hole_reach"):
            self.assertEqual(getattr(off, field), 0.0, field)

    def test_invalid_timestep_rejected(self):
        with self.assertRaises(ValueError):
            apply_suspension(np.zeros(2),-np.ones(2),.16,core_z=.2,core_vz=0,dt=0)


if __name__ == '__main__':
    unittest.main()
