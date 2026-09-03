import mujoco
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from omegaconf import OmegaConf

cfg = load_config("configs/rl/circle_track.yaml")
OmegaConf.set_struct(cfg, False)
if not hasattr(cfg, "scenario") or cfg.scenario is None:
    cfg.scenario = {}
cfg.scenario.floor_radius = 1.6
cfg.scenario.wall_radius = 2.4
cfg.scenario.apron_height = 0.8
cfg.scenario.cylinder_height = 4.5

scenario = generate_scenario("motordrome", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
env.reset(seed=42)

# Check geoms around r=1.6
for i in range(env.model.ngeom):
    name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name and "apron" in name and i < 20:
        pos = env.model.geom_pos[i]
        size = env.model.geom_size[i]
        body_id = env.model.geom_bodyid[i]
        body_pos = env.model.body_pos[body_id]
        print(f"Geom {name}: body_pos={body_pos}, pos={pos}, size={size}")
