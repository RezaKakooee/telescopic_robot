import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, mujoco
from omegaconf import OmegaConf
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.mjcf import SLEEVE_STUB, TIP_GAP, FOOT_RADIUS, rolling_radius
from skills.wall_of_death import wall_of_death, surface_normal

cfg = load_config("configs/rl/motordrome.yaml"); OmegaConf.set_struct(cfg, False)
sc = generate_scenario("motordrome", cfg, seed=1); md=sc.motordromes[0]
wall_r, mu = md[3], md[6]
env = MujocoRadialSphereEnv(cfg, scenario=sc, randomize=False, max_steps=10**6); env.reset(seed=1)
E = env.max_extend
print(f"SLEEVE_STUB={SLEEVE_STUB} TIP_GAP={TIP_GAP} FOOT_RADIUS={FOOT_RADIUS}")
tip0 = 0.15 + SLEEVE_STUB + TIP_GAP
print(f"foot centre at rest = {tip0:.4f}; outer surface = {tip0+FOOT_RADIUS:.4f}")
print(f"full stroke {E}: outer surface = {tip0+FOOT_RADIUS+E:.4f}")
print(f"rolling_radius(0.15*E) = {rolling_radius(0.15,0.15*E):.4f}")
print(f"wall at {wall_r}; centre must be within {wall_r-(tip0+FOOT_RADIUS+E):.3f}..{wall_r-(tip0+FOOT_RADIUS):.3f}")

# now trace a placed ball
r0 = wall_r - (tip0+FOOT_RADIUS) - 0.5*E
env.data.qpos[0]=r0; env.data.qpos[1]=0; env.data.qpos[2]=1.60
env.data.qpos[3:7]=[1,0,0,0]; env.data.qvel[:]=0.0
env.data.qvel[1]=4.2; env.data.qvel[3]=-4.2/0.19
mujoco.mj_forward(env.model, env.data)
print(f"\nplaced at r={r0:.3f}, z=1.60, vt=4.2\n")
print(f"{'i':>4} {'z':>6} {'r':>5} {'vz':>6} {'vt':>5} {'ncon':>4} {'nx':>5} {'ny':>5} {'nz':>5} {'maxext':>6} {'nrods>0.1':>9}")
for i in range(160):
    pos=env.data.qpos[:3].copy(); vel=env.data.qvel[:3].copy()
    n=surface_normal(env.model, env.data, pos)
    tg,info=wall_of_death(env.data.qpos[3:7].copy(), env.dirs_body, E, pos, vel,
                          normal=n, wall_radius=wall_r, wall_friction=mu,
                          ccw=True, climb_rate=0.0, press=0.45)
    if i%10==0:
        th=np.arctan2(pos[1],pos[0]); t_hat=np.array([-np.sin(th),np.cos(th),0])
        nn = n if n is not None else np.zeros(3)
        print(f"{i:4d} {pos[2]:6.2f} {info['r']:5.2f} {vel[2]:6.2f} {np.dot(vel,t_hat):5.2f} "
              f"{env.data.ncon:4d} {nn[0]:5.2f} {nn[1]:5.2f} {nn[2]:5.2f} {tg.max():6.3f} {(tg>0.1).sum():9d}")
    env.step(tg)
env.close()
