"""Place the ball ON the wall with a chosen speed. Can the gait hold it there?

This removes the approach from the question. If a hand-placed ball cannot
hold, the controller is wrong. If it can, the problem is the run-up.
"""
import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import wall_of_death, surface_normal

def trial(v0, z0=1.60, climb=0.0, press=0.45, steps=600, ext_frac=0.55, seed=1, log=False):
    cfg = load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg, False)
    sc = generate_scenario("motordrome", cfg, seed=seed)
    md = sc.motordromes[0]; wall_r, mu = md[3], md[6]
    env = MujocoRadialSphereEnv(cfg, scenario=sc, randomize=False, max_steps=10**6)
    env.reset(seed=seed)
    # centre orbit radius = wall - foot base - part of the stroke
    r0 = wall_r - 0.15 - ext_frac*env.max_extend
    env.data.qpos[0]=r0; env.data.qpos[1]=0.0; env.data.qpos[2]=z0
    env.data.qpos[3:7]=[1,0,0,0]
    env.data.qvel[:]=0.0
    env.data.qvel[1]=v0                      # tangential (anticlockwise at +x)
    env.data.qvel[3:6]=[0,0,0]
    # spin the ball so it is already rolling on the wall: omega = v/R about -x
    env.data.qvel[3] = -v0/0.19
    mujoco.mj_forward(env.model, env.data)
    zs=[]; vs=[]; ms=[]
    for i in range(steps):
        pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
        n=surface_normal(env.model, env.data, pos)
        tg,info=wall_of_death(env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                              pos, vel, normal=n, wall_radius=wall_r, wall_friction=mu,
                              ccw=True, climb_rate=climb, press=press)
        env.step(tg); zs.append(info["z"]); vs.append(info["speed"]); ms.append(info["margin"])
        if log and i%100==0:
            nn = n if n is not None else np.zeros(3)
            print(f"    {i:4d} z={info['z']:5.2f} r={info['r']:4.2f} v={info['speed']:5.2f} "
                  f"m={info['margin']:4.2f} wall={info['wallness']:4.2f} {info['mode']}")
        if info["z"] < 0.5: break
    env.close()
    return dict(steps=len(zs), z_end=zs[-1], z_drop=z0-zs[-1], v_end=vs[-1],
                v_mean=float(np.mean(vs)), m_mean=float(np.mean(ms)))

print("Hold test: ball placed on the wall at z=1.60, level orbit, 6 s\n")
print(f"{'v0':>5} {'needed':>7} {'steps':>6} {'z_end':>6} {'drop':>6} {'v_end':>6} {'margin':>7}")
cfg = load_config("configs/rl/motordrome.yaml")
sc = generate_scenario("motordrome", cfg, seed=1); md=sc.motordromes[0]
for v0 in (2.6, 3.0, 3.4, 3.8, 4.2, 4.6):
    r = trial(v0)
    need = np.sqrt(9.81*(md[3]-0.15-0.55*0.26)/md[6])
    print(f"{v0:5.1f} {need:7.2f} {r['steps']:6d} {r['z_end']:6.2f} {r['z_drop']:6.2f} "
          f"{r['v_end']:6.2f} {r['m_mean']:7.2f}")

print("\n\nPress sweep at v0=4.2, level orbit, 6 s (drop < 0.2 m = holding)")
print(f"{'press':>6} {'z_end':>6} {'drop':>6} {'v_end':>6}")
for pr in (0.45, 0.65, 0.80, 0.90, 1.00):
    r = trial(4.2, press=pr)
    print(f"{pr:6.2f} {r['z_end']:6.2f} {r['z_drop']:6.2f} {r['v_end']:6.2f}")
