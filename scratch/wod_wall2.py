"""Hold test, corrected. The ball rolls on an envelope of ~0.21 m, so its
centre must orbit at wall_radius - 0.21, not deeper."""
import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import wall_of_death, surface_normal

FOOT = 0.173   # foot outer surface at zero stroke

def trial(v0, press, ext_frac, z0=1.60, climb=0.0, steps=600, seed=1, speed=None):
    cfg = load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg, False)
    sc = generate_scenario("motordrome", cfg, seed=seed); md=sc.motordromes[0]
    wall_r, mu = md[3], md[6]
    env = MujocoRadialSphereEnv(cfg, scenario=sc, randomize=False, max_steps=10**6); env.reset(seed=seed)
    E = env.max_extend
    R_roll = FOOT + ext_frac*E
    r0 = wall_r - R_roll
    env.data.qpos[0]=r0; env.data.qpos[1]=0; env.data.qpos[2]=z0
    env.data.qpos[3:7]=[1,0,0,0]; env.data.qvel[:]=0.0
    env.data.qvel[1]=v0
    # Rolling on the wall means the contact point stands still. Touching at
    # +x while moving in +y, that needs spin about z, not x.
    env.data.qvel[5]=-v0/R_roll
    mujoco.mj_forward(env.model, env.data)
    kw = {} if speed is None else {"speed": speed}
    zs=[];vs=[];nc=[]
    for i in range(steps):
        pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
        n=surface_normal(env.model, env.data, pos)
        tg,info=wall_of_death(env.data.qpos[3:7].copy(), env.dirs_body, E, pos, vel,
                              normal=n, wall_radius=wall_r, wall_friction=mu,
                              ccw=True, climb_rate=climb, press=press, **kw)
        env.step(tg); zs.append(info["z"]); vs.append(info["speed"]); nc.append(env.data.ncon)
        if info["z"]<0.45: break
    env.close()
    return dict(n=len(zs), z_end=zs[-1], drop=z0-zs[-1], v_end=vs[-1],
                v_min=float(np.min(vs)), ncon=float(np.mean(nc)))

print("Hold test v0=4.2, 6 s, correct roll spin. drop < 0.2 m = holding")
print(f"{'ext':>5} {'press':>6} {'gain(v)':>8} {'steps':>6} {'z_end':>6} {'drop':>6} {'v_end':>6} {'v_min':>6} {'ncon':>5}")
for ef in (0.35,):
    for pr in (0.0, 0.15, 0.30):
        for sp in (0.6, 1.2, 2.0, 2.8):
            r = trial(4.2, pr, ef, speed=sp)
            print(f"{ef:5.2f} {pr:6.2f} {sp:8.1f} {r['n']:6d} {r['z_end']:6.2f} {r['drop']:6.2f} "
                  f"{r['v_end']:6.2f} {r['v_min']:6.2f} {r['ncon']:5.1f}")
