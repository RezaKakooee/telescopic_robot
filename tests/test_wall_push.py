"""Unit and physics tests for the wall contact & obstacle pushing skill (push_against_wall).

Verifies:
1. Selective rod projection: only rods facing the wall surface extend; opposite rods remain tucked.
2. Normal force & lateral shove: active pushing against a wall generates repulsive thrust, displacing the robot away into free space.
"""
import unittest
import numpy as np
import mujoco

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario
from skills import execute_skill


class TestWallPushSkill(unittest.TestCase):
    def test_selective_rod_extension(self):
        """Kinematic check: rods facing the wall extend, opposite rods remain tucked."""
        cfg = load_config("configs/rl/config.yaml")
        dirs_body = np.random.randn(60, 3)
        dirs_body /= np.linalg.norm(dirs_body, axis=1, keepdims=True)
        quat = np.array([1.0, 0.0, 0.0, 0.0]) # identity rotation
        
        # Wall is along -y, so normal pointing toward robot is +y [0, 1]
        wall_normal = np.array([0.0, 1.0])
        max_extend = 0.16
        
        targets = execute_skill("push_against_wall", quat, dirs_body, max_extend, wall_normal=wall_normal, push_strength=0.90)
        
        # For identity quaternion, dirs_world == dirs_body
        # Wall projection is dirs_body[:, 1]
        facing_wall = dirs_body[:, 1] < -0.30
        away_from_wall = dirs_body[:, 1] > 0.10
        
        # Wall-facing rods should extend significantly
        self.assertTrue(np.any(targets[facing_wall] > 0.08), "Expected wall-facing rods to extend")
        # Opposite rods should stay tucked (at or below baseline stance)
        self.assertTrue(np.all(targets[away_from_wall] <= 0.045), "Expected opposite rods to stay tucked")
        print("✅ Selective rod extension verified")

    def test_dynamic_wall_shove(self):
        """Physics check: robot pushes against a real barrier and drives itself away."""
        cfg = load_config("configs/rl/config.yaml")
        cfg.camera.enabled = False
        
        wall_y = -0.45
        scenario = Scenario(
            kind="goal",
            name="test_wall",
            spawn_xy=np.array([0.0, -0.26], dtype=np.float32),
            goal=np.array([5.0, 0.0], dtype=np.float32),
            path_pts=np.array([[0.0, -0.26], [5.0, -0.26]], dtype=np.float32),
            markers=np.empty((0, 2), dtype=np.float32),
            path_length=5.0,
            walls=np.array([[-1.0, wall_y, 6.0, wall_y]], dtype=np.float32),
        )
        env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=300)
        env.reset(seed=42)
        
        start_y = float(env.data.qpos[1])
        wall_normal = np.array([0.0, 1.0])
        
        # Apply push_against_wall for 60 steps
        max_vy = 0.0
        for step in range(70):
            quat = env.data.qpos[3:7].copy()
            targets = execute_skill("push_against_wall", quat, env.dirs_body, env.max_extend,
                                    wall_normal=wall_normal, push_strength=0.95)
            env.step(targets)
            max_vy = max(max_vy, float(env.data.qvel[1]))
            
        end_y = float(env.data.qpos[1])
        displacement = end_y - start_y
        env.close()
        
        self.assertGreater(displacement, 0.12, f"Failed to push away from wall: displacement={displacement:.3f}m")
        self.assertGreater(max_vy, 0.25, f"Peak repulsive velocity too low: max_vy={max_vy:.2f}m/s")
        print(f"✅ Dynamic wall shove verified: pushed {displacement*100:+.1f}cm away, max vy={max_vy:.2f}m/s")


if __name__ == "__main__":
    unittest.main()
