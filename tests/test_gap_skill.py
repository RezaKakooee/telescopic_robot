"""Unit and physics tests for the trench and gap crossing skill (straddle_gap).

Verifies:
1. Dual-flank outrigger locomotion spans an open 22 cm trench (25 cm depth) between Box 1 and Box 2.
2. Core elevation is maintained on top of the platforms throughout the 5 m course (z > 0.35 m at all times).
3. Active centerline feedback keeps the robot centered over the trench.
4. Contrast verification: standard move_forward fails to traverse the gap due to void traction loss.
"""
import unittest
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill


class TestGapSkill(unittest.TestCase):
    def test_straddle_gap_traversal(self):
        """Verify straddle_gap crosses the full 5m course on dual platforms without dropping into the trench."""
        cfg = load_config("configs/rl/gap_bridge.yaml")
        cfg.camera.enabled = False
        scenario = generate_scenario("gap_bridge", cfg, seed=42)
        scenario.walls = np.zeros((0, 4), dtype=np.float32)
        
        env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10000)
        env.reset(seed=42)
        
        deck_h = 0.25
        env.data.qpos[0] = 0.0
        env.data.qpos[1] = 0.0
        env.data.qpos[2] = deck_h + 0.19
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        
        # Settle on platforms
        for _ in range(25):
            env.step(execute_skill("straddle_gap", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend))
            
        d_fwd = np.array([1.0, 0.0])
        min_z = float(env.data.qpos[2])
        max_y = 0.0
        
        for step in range(500):
            pos = env.data.qpos[:3].copy()
            quat = env.data.qpos[3:7].copy()
            
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            min_z = min(min_z, z)
            max_y = max(max_y, abs(y))
            
            if x >= 4.80:
                break
                
            targets = execute_skill(
                "straddle_gap", quat, env.dirs_body, env.max_extend,
                d_hat=d_fwd, lateral_offset=y, centering_gain=2.2,
            )
            env.step(targets)
            
        end_x = float(env.data.qpos[0])
        end_y = float(env.data.qpos[1])
        end_z = float(env.data.qpos[2])
        env.close()
        
        # Must traverse across the platforms
        self.assertGreater(end_x, 4.2, f"Failed forward progress: end_x={end_x:.2f}m")
        # Must stay on top of the platforms (deck height 0.25m, core radius 0.15m)
        self.assertGreater(min_z, deck_h + 0.10, f"Core dropped into trench! min_z={min_z:.3f}m")
        # Centering deviation must stay bounded
        self.assertLess(max_y, 0.15, f"Centerline deviation too large: max_y={max_y:.3f}m")
        print(f"✅ Straddle gap traverse verified: end_x={end_x:.2f}m, end_y={end_y*100:+.1f}cm, min_z={min_z:.3f}m")

    def test_uncontrolled_move_forward_stall(self):
        """Verify that standard underbelly drive stalls on the gap because pushers fire into thin air."""
        cfg = load_config("configs/rl/gap_bridge.yaml")
        cfg.camera.enabled = False
        scenario = generate_scenario("gap_bridge", cfg, seed=42)
        scenario.walls = np.zeros((0, 4), dtype=np.float32)
        
        env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=1000)
        env.reset(seed=42)
        
        deck_h = 0.25
        env.data.qpos[0] = 0.0
        env.data.qpos[1] = 0.0
        env.data.qpos[2] = deck_h + 0.19
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        
        d_fwd = np.array([1.0, 0.0])
        for step in range(250):
            targets = execute_skill("move_forward", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, d_hat=d_fwd)
            env.step(targets)
            
        end_x = float(env.data.qpos[0])
        env.close()
        
        # Standard move_forward pushes into empty trench air and cannot progress
        self.assertLess(end_x, 0.30, f"Standard move_forward unexpectedly moved across gap: end_x={end_x:.2f}m")
        print(f"✅ Baseline failure verified: move_forward stalled at x={end_x:.2f}m (< 0.30m)")


if __name__ == "__main__":
    unittest.main()
