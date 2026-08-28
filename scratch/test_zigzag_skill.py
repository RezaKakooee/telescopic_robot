import numpy as np
import mujoco
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
obs, info = env.reset(seed=42)

env.data.qpos[1] = -0.05
env.data.qpos[2] = 0.5
mujoco.mj_forward(env.model, env.data)

class ZigZagClimber:
    def __init__(self, direction="up"):
        self.state = "jump_right"
        self.timer = 0
        self.direction = direction # "up" or "down"
        
    def step(self, env):
        pos = env.data.qpos[:3]
        vel = env.data.qvel[:3]
        targets = np.zeros(60)
        
        # Determine push force based on direction
        # If climbing UP, push HARD (0.16) to gain height
        # If climbing DOWN, push SOFT (0.06) so gravity pulls it down while it bounces
        force = 0.16 if self.direction == "up" else 0.08
        
        if self.state == "jump_right":
            # At left wall, push down-left rods
            for i, d in enumerate(env.dirs_body):
                if d[1] < -0.2 and d[2] < -0.2:
                    targets[i] = force
            self.timer += 1
            if self.timer > 4 and vel[1] > 0.4:
                self.state = "fly_right"
                self.timer = 0
                
        elif self.state == "fly_right":
            if pos[1] > 0.03:
                self.state = "jump_left"
                self.timer = 0
                
        elif self.state == "jump_left":
            # At right wall, push down-right rods
            for i, d in enumerate(env.dirs_body):
                if d[1] > 0.2 and d[2] < -0.2:
                    targets[i] = force
            self.timer += 1
            if self.timer > 4 and vel[1] < -0.4:
                self.state = "fly_left"
                self.timer = 0
                
        elif self.state == "fly_left":
            if pos[1] < -0.03:
                self.state = "jump_right"
                self.timer = 0
                
        return targets

print("Testing Ascend...")
climber = ZigZagClimber(direction="up")
for step in range(500):
    targets = climber.step(env)
    env.step(targets)

print(f"After Ascend: z={env.data.qpos[2]:.3f}m")

print("Testing Descend...")
climber.direction = "down"
for step in range(500):
    targets = climber.step(env)
    env.step(targets)
    
print(f"After Descend: z={env.data.qpos[2]:.3f}m")

