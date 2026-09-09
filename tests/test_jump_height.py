import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill

def main():
    cfg = load_config("configs/rl/standing_jump_showcase.yaml")
    env = MujocoRadialSphereEnv(cfg, scenario=generate_scenario("goal", cfg, seed=42), randomize=False, max_steps=200)

    # 1. jump_up: 20cm vs 50cm
    heights = {}
    for target_h in (20.0, 50.0):
        env.reset(seed=42)
        start_z = env.data.qpos[2]
        max_z = start_z
        for step in range(60):
            quat = env.data.qpos[3:7]
            phase = "crouch" if step < 20 else ("takeoff" if step < 32 else "airborne")
            targets = execute_skill("jump_up", quat, env.dirs_body, env.max_extend,
                                    phase=phase, jump_height_cm=target_h)
            env.step(targets)
            max_z = max(max_z, env.data.qpos[2])
        heights[target_h] = (max_z - start_z) * 100.0
    print(f"jump_up: 20cm request -> {heights[20.0]:.1f}cm, 50cm request -> {heights[50.0]:.1f}cm")
    assert heights[20.0] < heights[50.0] - 15.0, "FAIL: 20cm jump must be significantly lower than 50cm jump"

    # 2. jump_forward_while_stopped: 20cm vs 45cm
    for target_h in (20.0, 45.0):
        env.reset(seed=42)
        start_z = env.data.qpos[2]
        max_z = start_z
        for step in range(80):
            quat = env.data.qpos[3:7]
            phase = "crouch" if step < 20 else ("takeoff" if step < 32 else ("airborne" if env.data.qpos[2] > 0.28 else "landing"))
            targets = execute_skill("jump_forward_while_stopped", quat, env.dirs_body, env.max_extend,
                                    d_hat=np.array([1., 0.]), phase=phase, jump_height_cm=target_h)
            env.step(targets)
            max_z = max(max_z, env.data.qpos[2])
        heights[target_h] = (max_z - start_z) * 100.0
    print(f"jump_fwd_stop: 20cm request -> {heights[20.0]:.1f}cm, 45cm request -> {heights[45.0]:.1f}cm")
    assert heights[20.0] < heights[45.0] - 15.0, "FAIL: 20cm jump must be significantly lower than 45cm jump"

    # 3. jump_forward_while_moving: 20cm vs 50cm with running runup
    for target_h in (20.0, 50.0):
        env.reset(seed=42)
        start_z = env.data.qpos[2]
        orig_y = env.data.qpos[1]
        max_z = start_z
        dt = float(env.model.opt.timestep * env.action_repeat)
        for step in range(round(3.0 / dt)):
            t = step * dt
            quat = env.data.qpos[3:7]
            if t < 1.5:
                targets = execute_skill("move", quat, env.dirs_body, env.max_extend,
                                        d_hat=np.array([1., 0.]), speed=1.2,
                                        lin_vel=env.data.qvel[:2].copy(),
                                        cross_track_error=float(env.data.qpos[1] - orig_y),
                                        rod_mechanism="multi_stage")
            elif t < 1.57:
                targets = execute_skill("jump_forward_while_moving", quat, env.dirs_body, env.max_extend,
                                        d_hat=np.array([1., 0.]), phase="dip")
            elif t < 1.70:
                targets = execute_skill("jump_forward_while_moving", quat, env.dirs_body, env.max_extend,
                                        d_hat=np.array([1., 0.]), phase="launch", jump_height_cm=target_h)
            elif env.data.qpos[2] > 0.28:
                targets = execute_skill("jump_forward_while_moving", quat, env.dirs_body, env.max_extend,
                                        d_hat=np.array([1., 0.]), phase="airborne")
            else:
                targets = execute_skill("jump_forward_while_moving", quat, env.dirs_body, env.max_extend,
                                        d_hat=np.array([1., 0.]), phase="landing")
            env.step(targets)
            if t >= 1.5:
                max_z = max(max_z, env.data.qpos[2])
        heights[target_h] = (max_z - start_z) * 100.0
    print(f"jump_fwd_mov: 20cm request -> {heights[20.0]:.1f}cm, 50cm request -> {heights[50.0]:.1f}cm")
    assert heights[20.0] < heights[50.0] - 15.0, "FAIL: 20cm jump must be significantly lower than 50cm jump"

    env.close()
    print("ALL JUMP HEIGHT CONTROL CHECKS PASSED ✅")

if __name__ == "__main__":
    main()
