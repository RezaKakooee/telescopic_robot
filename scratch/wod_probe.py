"""Headless wall-of-death probe: does the ball climb and stay up?"""
import os, sys, argparse
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills.wall_of_death import wall_of_death, surface_normal

def run(steps=3000, climb=0.55, press=0.45, seed=1, verbose=True, **over):
    cfg = load_config("configs/rl/motordrome.yaml")
    OmegaConf.set_struct(cfg, False)
    for k, v in over.items():
        setattr(cfg.scenario, k, v)
    sc = generate_scenario("motordrome", cfg, seed=seed)
    md = sc.motordromes[0]
    floor_r, wall_r, apron_h, total_h, mu = md[2], md[3], md[4], md[5], md[6]
    env = MujocoRadialSphereEnv(cfg, scenario=sc, randomize=False, max_steps=10**6)
    env.reset(seed=seed)
    env.data.qpos[0] = floor_r * 0.55; env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.15 + 0.15 * env.max_extend + 0.01
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    max_z = 0.0; max_v = 0.0; laps = 0.0
    prev_th = 0.0; hist = []
    for i in range(steps):
        pos = env.data.qpos[:3].copy(); vel = env.data.qvel[:3].copy()
        n = surface_normal(env.model, env.data, pos)
        tg, info = wall_of_death(env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                                 pos, vel, normal=n, wall_radius=wall_r,
                                 wall_friction=mu, ccw=True, climb_rate=climb, press=press)
        env.step(tg)
        th = np.arctan2(pos[1], pos[0]); d = th - prev_th
        d = (d + np.pi) % (2*np.pi) - np.pi
        if i: laps += d/(2*np.pi)
        prev_th = th
        max_z = max(max_z, info["z"]); max_v = max(max_v, info["speed"])
        hist.append((i, info["z"], info["r"], info["speed"], info["margin"], info["mode"]))
        if verbose and i % 250 == 0:
            print(f"  {i:5d} z={info['z']:5.2f} r={info['r']:4.2f} v={info['speed']:5.2f} "
                  f"vt={info['v_t']:5.2f} margin={info['margin']:5.2f} {info['mode']:8s} "
                  f"n=({n[0]:5.2f},{n[1]:5.2f},{n[2]:5.2f})" if n is not None else "  air")
        if info["r"] > wall_r + 0.6: print("  ESCAPED"); break
    env.close()
    last = hist[-600:]
    return dict(max_z=max_z, max_v=max_v, laps=abs(laps),
                z_mean_last=float(np.mean([h[1] for h in last])),
                z_min_last=float(np.min([h[1] for h in last])),
                v_mean_last=float(np.mean([h[3] for h in last])),
                margin_last=float(np.mean([h[4] for h in last])))

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--steps", type=int, default=3000)
    a.add_argument("--climb", type=float, default=0.55); a.add_argument("--press", type=float, default=0.45)
    r = run(**vars(a.parse_args()))
    print("\n" + " | ".join(f"{k}={v:.2f}" if isinstance(v,float) else f"{k}={v}" for k,v in r.items()))
