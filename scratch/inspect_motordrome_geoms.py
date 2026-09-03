import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario

def main():
    cfg = load_config("configs/rl/circle_track.yaml")
    scenario = generate_scenario("motordrome", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
    env.reset(seed=42)

    print("=== Motordrome Geom Inspection ===")
    for i in range(env.model.ngeom):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
        pos = env.model.geom_pos[i]
        size = env.model.geom_size[i]
        gtype = env.model.geom_type[i]
        contype = env.model.geom_contype[i]
        conaffinity = env.model.geom_conaffinity[i]
        if "md_" in (name or ""):
            print(f"Geom {i:3d}: {name:20s} | type={gtype} | pos={pos} | size={size} | contype={contype} | conaff={conaffinity}")

if __name__ == "__main__":
    main()
