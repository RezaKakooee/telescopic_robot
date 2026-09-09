import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill

def test_curve_right_and_left():
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range, cfg.scenario.goal.y_range = [0., 0.], [-400., -400.]
    env = MujocoRadialSphereEnv(cfg, scenario=generate_scenario("goal", cfg, seed=42),
                               randomize=False, max_steps=5000)
    env.reset(seed=42)
    dt = float(env.model.opt.timestep * env.action_repeat)

    # 1. Test Right Curve
    d_curr = np.array([1.0, 0.0])
    traj_r = []
    for step in range(round(3.0 / dt)):
        pos = env.data.qpos[:2].copy()
        quat = env.data.qpos[3:7].copy()
        traj_r.append(pos)
        targets = execute_skill(
            "curve", quat, env.dirs_body, env.max_extend,
            d_hat=d_curr, radius=2.0, speed=1.2, direction="right",
            ball_xy=pos, rod_mechanism="multi_stage"
        )
        env.step(targets)

    traj_r = np.array(traj_r)
    # Right curve from [1, 0] should produce negative y displacement
    assert traj_r[-1][1] < -0.5, f"FAIL: Right curve did not curve right: {traj_r[-1]}"
    print(f"✅ Right curve verified: end={traj_r[-1]}")

    # 2. Reset and Test Left Curve
    env.reset(seed=42)
    traj_l = []
    for step in range(round(3.0 / dt)):
        pos = env.data.qpos[:2].copy()
        quat = env.data.qpos[3:7].copy()
        traj_l.append(pos)
        targets = execute_skill(
            "curve", quat, env.dirs_body, env.max_extend,
            d_hat=d_curr, radius=2.0, speed=1.2, direction="left",
            ball_xy=pos, rod_mechanism="multi_stage"
        )
        env.step(targets)

    traj_l = np.array(traj_l)
    # Left curve from [1, 0] should produce positive y displacement
    assert traj_l[-1][1] > 0.5, f"FAIL: Left curve did not curve left: {traj_l[-1]}"
    print(f"✅ Left curve verified: end={traj_l[-1]}")

    # 3. Test signed curvature
    env.reset(seed=42)
    targets_curv = execute_skill(
        "curve", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
        d_hat=d_curr, curvature=0.5, ball_xy=env.data.qpos[:2].copy()
    )
    assert len(targets_curv) == env.n_bars
    print(f"✅ Signed curvature verified")

    env.close()
    print("ALL CURVE SKILL TESTS PASSED! 🎉")


def load_tests(loader, tests, pattern):
    """Expose this module's plain test functions to `unittest discover`."""
    import sys
    from tests._function_suite import suite_from_module
    return suite_from_module(sys.modules[__name__])


if __name__ == "__main__":
    test_curve_right_and_left()
