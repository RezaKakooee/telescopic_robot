import numpy as np
import mujoco
from radial_sphere.config import load_config_cli
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv

cfg = load_config_cli(name="playground_parkour_skills")
sc = generate_scenario("playground", cfg, seed=100)
env = SkillArbitrationEnv(cfg, scenario=sc, seed=100)

print("Starting jump test...", flush=True)

for target_x in [2.0, 2.2, 2.5, 2.7, 2.9, 3.1]:
    env.reset(seed=100)
    jumped = False
    jump_start_x = 0.0
    contacts = [0]
    
    def check_c(cur_env):
        d = cur_env.env.data
        m = cur_env.env.model
        for i in range(d.ncon):
            g1 = m.geom(d.contact[i].geom1).name
            g2 = m.geom(d.contact[i].geom2).name
            if "hurdle" in g1 or "hurdle" in g2:
                contacts[0] += 1
    env.on_control_step = check_c

    for s in range(50):
        pos = env.env.data.qpos[:3].copy()
        if pos[0] >= target_x and not jumped:
            sk = "jump_forward_while_moving"
            jumped = True
            jump_start_x = float(pos[0])
        elif jumped:
            sk = "move"
        else:
            sk = "move"
            
        act = np.full(len(env.skill_names), -1.0, dtype=np.float32)
        act[env.skill_names.index(sk)] = 1.0
        env.step(act)
        if jumped:
            landed_pos = env.env.data.qpos[:3]
            print(f"Trigger >= {target_x:4.1f}m (start x={jump_start_x:4.2f}m) -> landed at x={landed_pos[0]:4.2f}m, z={landed_pos[2]:.2f}m | hurdle contacts: {contacts[0]}", flush=True)
            break
