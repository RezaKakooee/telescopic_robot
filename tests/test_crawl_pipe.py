"""crawl_pipe rolls through a conduit after a diagonal approach without touching the walls."""
import unittest

import mujoco
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.runner import skill_targets


class CrawlPipeTests(unittest.TestCase):
    def test_diagonal_entry_no_wall_hits(self):
        cfg = load_config()
        sc = generate_scenario("inspection_utility_tunnel", cfg, seed=0)   # first pipe x in [1.0, 3.4], r = 0.44
        env = MujocoRadialSphereEnv(cfg, max_steps=100000, scenario=sc)
        env.reset(seed=0)
        env.data.qpos[:3] = [-0.8, 0.0, 0.2]
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        hits = 0
        for t in range(1200):
            yaw = np.radians(35.0) if t < 60 else 0.0          # aim 35 deg off the axis at first
            targets = skill_targets(env, "crawl_pipe", t, d_hat=[np.cos(yaw), np.sin(yaw)], speed=0.7)
            _, _, _, _, info = env.step(targets)
            hits += int(info["obstacle_hit"])
            if env.data.qpos[0] > 3.8:
                break
        env.close()
        self.assertGreater(env.data.qpos[0], 3.8, "did not get through the pipe")
        self.assertEqual(hits, 0)


if __name__ == "__main__":
    unittest.main()
