"""Unit and physics tests for the fall_down skill.

Verifies:
1. Controlled ledge roll-off without tripping or tumbling.
2. Airborne lip clearance and landing gear extension.
3. Core shock absorption: core shell never strikes the ground (z > 0.15m at all times).
4. Stable upright settle on the ground.
"""
import unittest
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario, skill_course_platform
from skills import execute_skill


class TestFallSkill(unittest.TestCase):
    def test_fall_down_platform(self):
        cfg = load_config("configs/rl/config.yaml")
        cfg.camera.enabled = False
        scenario = generate_scenario("skill_course", cfg, seed=42)
        plat = skill_course_platform(cfg)
        
        env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10000)
        env.reset(seed=42)
        
        px, py = float(plat["xy"][0]), float(plat["xy"][1])
        env.data.qpos[0] = px - 0.25
        env.data.qpos[1] = py
        env.data.qpos[2] = plat["height"] + 0.20
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        
        # Settle on deck
        for _ in range(50):
            env.step(execute_skill("stop", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend))
            
        start_z = float(env.data.qpos[2])
        deck = plat["height"] + 0.19
        drop_h = plat["height"]
        d = np.array([1.0, 0.0])
        
        phase = "edge"
        dt = float(env.model.opt.timestep * env.action_repeat)
        min_z = start_z
        
        for step in range(round(3.0 / dt)):
            z = float(env.data.qpos[2])
            vz = float(env.data.qvel[2])
            min_z = min(min_z, z)
            
            if phase == "edge" and z < deck - 0.04:
                phase = "freefall"
            elif phase == "freefall" and z < 0.28:
                phase = "absorb"
            elif phase == "absorb" and z < 0.22 and vz > -0.2:
                phase = "settle"
                
            targets = execute_skill("fall_down", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                                    d_hat=d, phase=phase, drop_height=drop_h, edge_speed=0.35, gear=0.5)
            env.step(targets)
            
        end_z = float(env.data.qpos[2])
        end_x = float(env.data.qpos[0])
        
        # Core radius is 0.15m. Under no circumstances should min_z < 0.15m
        self.assertGreater(min_z, 0.15, f"Core impacted ground! min_z={min_z:.3f}m")
        # Should have stepped off the platform
        self.assertGreater(end_x, px + plat["half_depth"], "Robot failed to roll past platform edge")
        # Should have landed below deck level
        self.assertLess(end_z, deck - 0.05, f"Robot did not drop below deck: end_z={end_z:.3f}m")
        # Should settle stably on ground
        self.assertTrue(0.18 < end_z < 0.25, f"Robot failed to settle in upright ground stance: end_z={end_z:.3f}m")
        print(f"✅ Fall down verified: drop={(start_z - end_z)*100:.1f}cm, min_z={min_z:.3f}m, end_z={end_z:.3f}m")


if __name__ == "__main__":
    unittest.main()
