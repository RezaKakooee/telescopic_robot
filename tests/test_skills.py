"""Verification test: Run every skill for ~30 steps and verify it produces the expected motion."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario, skill_course_platform
from skills import SKILL_REGISTRY, execute_skill


def test_wall_push_in_maze(n_steps=80, seed=3):
    """Physics check for push_against_wall: find a real maze wall with lidar,
    push against it, and confirm the robot drives itself away from that wall."""
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    scenario = generate_scenario("maze", cfg, seed=seed)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=600)
    env.reset(seed=seed)

    # Locate the nearest wall from the 16-ray lidar (ray 0 along world +x).
    rays = env.raycast_lidar(n_rays=16, max_range=3.0, g=np.array([1.0, 0.0]))
    k = int(np.argmin(rays))
    angle = k / 16 * 2 * np.pi
    to_wall = np.array([np.cos(angle), np.sin(angle)])   # robot → wall
    wall_normal = -to_wall                               # wall → robot

    start_xy = env.data.qpos[0:2].copy()
    for _ in range(n_steps):
        targets = execute_skill("push_against_wall", env.data.qpos[3:7].copy(),
                                env.dirs_body, env.max_extend, wall_normal=wall_normal)
        env.step(targets)
    disp = env.data.qpos[0:2].copy() - start_xy
    away = float(np.dot(disp, wall_normal))
    env.close()

    print(f"8b. push_wall(maze): wall at {float(rays[k]) * 3.0:.3f}m  "
          f"pushed {away * 100:+.1f}cm away from it")
    assert away > 0.01, f"FAIL: Expected to push away from the wall, got {away:+.4f}m"


def run_skill_test(skill_name, env, n_steps=30, **skill_kwargs):
    """Run a skill for n_steps, return (dx, dy, dz, final_vx, final_vy, final_vz)."""
    obs, info = env.reset(seed=42)
    start_pos = env.data.qpos[0:3].copy()

    for step in range(n_steps):
        quat = env.data.qpos[3:7]
        targets = execute_skill(skill_name, quat, env.dirs_body, env.max_extend, **skill_kwargs)
        obs, rew, term, trunc, info = env.step(targets)

    end_pos = env.data.qpos[0:3].copy()
    end_vel = env.data.qvel[0:3].copy()
    delta = end_pos - start_pos
    return delta, end_vel


def test_fall_down_off_platform(seed=1):
    """Physics check for fall_down: place the robot on the course platform,
    step it off the edge and confirm it lands upright and under control."""
    import mujoco

    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    scenario = generate_scenario("skill_course", cfg, seed=seed)
    plat = skill_course_platform(cfg)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=seed)

    # Stand the robot on top of the platform, near its trailing edge.
    px, py = float(plat["xy"][0]), float(plat["xy"][1])
    env.data.qpos[0] = px - 0.35
    env.data.qpos[1] = py
    env.data.qpos[2] = plat["height"] + 0.20
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(60):
        env.step(execute_skill("stop", env.data.qpos[3:7].copy(),
                               env.dirs_body, env.max_extend))

    start_z = float(env.data.qpos[2])
    deck = plat["height"] + 0.19
    d = np.array([1.0, 0.0])
    phase, max_drop_speed = "edge", 0.0
    for _ in range(1200):
        z = float(env.data.qpos[2])
        max_drop_speed = max(max_drop_speed, -float(env.data.qvel[2]))
        if phase == "edge" and z < deck - 0.05:
            phase = "freefall"
        elif phase == "freefall" and z < 0.26:
            phase = "absorb"
        elif phase == "absorb" and z < 0.215:
            phase = "settle"
        env.step(execute_skill("fall_down", env.data.qpos[3:7].copy(),
                               env.dirs_body, env.max_extend, d_hat=d, phase=phase))
        if phase == "settle" and float(env.data.qpos[0]) > px + plat["half_depth"] + 0.4:
            break

    end_z = float(env.data.qpos[2])
    end_x = float(env.data.qpos[0])
    env.close()

    print(f"12. fall_down:     {start_z:.3f}m -> {end_z:.3f}m  "
          f"(dropped {(start_z - end_z) * 100:.0f}cm)  "
          f"peak fall speed {max_drop_speed:.2f} m/s  x={end_x:.2f}m")
    assert end_z < deck - 0.10, (
        f"FAIL: Expected to end below the deck ({deck - 0.10:.2f}m), got {end_z:.3f}m")
    assert 0.15 < end_z < 0.30, f"FAIL: Expected to land upright, got z={end_z:.3f}m"
    assert end_x > px + plat["half_depth"], (
        f"FAIL: Expected to end past the platform edge, got x={end_x:.2f}m")


def main():
    cfg = load_config("configs/rl/standing_jump_showcase.yaml")
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=200)

    d_forward = np.array([1.0, 0.0], dtype=np.float32)
    d_right   = np.array([0.0, -1.0], dtype=np.float32)  # +x is forward, -y is right

    print("=" * 70)
    print("  SKILLS LIBRARY VERIFICATION TEST")
    print("=" * 70)

    results = {}

    # 1. move_forward
    delta, vel = run_skill_test("move_forward", env, 40, d_hat=d_forward)
    results["move_forward"] = (delta, vel)
    print(f"\n1. move_forward:  Δx={delta[0]:+.3f}m  Δy={delta[1]:+.3f}m  Δz={delta[2]:+.3f}m  vx={vel[0]:+.2f}m/s")
    assert delta[0] > 0.02, f"FAIL: Expected forward motion (+x), got Δx={delta[0]:.3f}"

    # 2. move_right
    delta, vel = run_skill_test("move_right", env, 40, d_hat=d_forward)
    results["move_right"] = (delta, vel)
    print(f"2. move_right:    Δx={delta[0]:+.3f}m  Δy={delta[1]:+.3f}m  Δz={delta[2]:+.3f}m  vy={vel[1]:+.2f}m/s")
    assert delta[1] < 0.0, f"FAIL: Expected rightward motion (-y), got Δy={delta[1]:.3f}"

    # 3. move_left
    delta, vel = run_skill_test("move_left", env, 40, d_hat=d_forward)
    results["move_left"] = (delta, vel)
    print(f"3. move_left:     Δx={delta[0]:+.3f}m  Δy={delta[1]:+.3f}m  Δz={delta[2]:+.3f}m  vy={vel[1]:+.2f}m/s")
    assert delta[1] > 0.0, f"FAIL: Expected leftward motion (+y), got Δy={delta[1]:.3f}"

    # 4. stop — passive stance, then active braking from full speed
    delta, vel = run_skill_test("stop", env, 30)
    results["stop"] = (delta, vel)
    print(f"4. stop (stance): Δx={delta[0]:+.3f}m  Δy={delta[1]:+.3f}m  Δz={delta[2]:+.3f}m  |v|={np.linalg.norm(vel):.3f}m/s")
    assert abs(delta[0]) < 0.10, f"FAIL: Expected minimal drift, got Δx={delta[0]:.3f}"

    # 4b. stop as a brake: build speed with go_fast, then stop and measure decay.
    env.reset(seed=42)
    for _ in range(50):
        env.step(execute_skill("go_fast", env.data.qpos[3:7].copy(),
                               env.dirs_body, env.max_extend, d_hat=d_forward))
    v_before = float(np.linalg.norm(env.data.qvel[0:2]))
    x_brake0 = float(env.data.qpos[0])
    for _ in range(160):
        env.step(execute_skill("stop", env.data.qpos[3:7].copy(),
                               env.dirs_body, env.max_extend,
                               lin_vel=env.data.qvel[0:2].copy()))
    v_after = float(np.linalg.norm(env.data.qvel[0:2]))
    coast = float(env.data.qpos[0]) - x_brake0
    # Stopping distance scales with entry speed, so bound it per m/s rather
    # than as a fixed number of metres.
    coast_per_speed = coast / max(v_before, 1e-6)
    print(f"4b. stop (brake): {v_before:.2f} → {v_after:.3f} m/s  "
          f"({100 * (1 - v_after / max(v_before, 1e-6)):.0f}% cut)  "
          f"coast={coast:.3f}m ({coast_per_speed:.2f} m per m/s)")
    assert v_before > 0.5, f"FAIL: brake test needs speed first, got {v_before:.3f}"
    assert v_after < 0.15, f"FAIL: Expected near-standstill after braking, got {v_after:.3f} m/s"
    assert coast_per_speed < 0.50, (
        f"FAIL: Expected under 0.50 m of coast per m/s, got {coast_per_speed:.2f}")

    # 5. go_fast
    delta_fast, vel_fast = run_skill_test("go_fast", env, 40, d_hat=d_forward)
    results["go_fast"] = (delta_fast, vel_fast)
    print(f"5. go_fast:       Δx={delta_fast[0]:+.3f}m  Δy={delta_fast[1]:+.3f}m  vx={vel_fast[0]:+.2f}m/s")
    assert delta_fast[0] > results["move_forward"][0][0], f"FAIL: go_fast should be faster than move_forward"

    # 6. go_slow
    delta_slow, vel_slow = run_skill_test("go_slow", env, 40, d_hat=d_forward)
    results["go_slow"] = (delta_slow, vel_slow)
    print(f"6. go_slow:       Δx={delta_slow[0]:+.3f}m  Δy={delta_slow[1]:+.3f}m  vx={vel_slow[0]:+.2f}m/s")
    assert delta_slow[0] < delta_fast[0], f"FAIL: go_slow should be slower than go_fast"

    # 7. reverse
    delta, vel = run_skill_test("reverse", env, 40, d_hat=d_forward)
    results["reverse"] = (delta, vel)
    print(f"7. reverse:       Δx={delta[0]:+.3f}m  Δy={delta[1]:+.3f}m  vx={vel[0]:+.2f}m/s")
    assert delta[0] < 0.0, f"FAIL: Expected backward motion (-x), got Δx={delta[0]:.3f}"

    # 8. push_against_wall — kinematic check: only the wall-facing rods extend.
    # Wall on the robot's right (-y side) → wall_normal points from wall to robot.
    wall_normal_right = np.array([0.0, 1.0])
    env.reset(seed=42)
    quat = env.data.qpos[3:7].copy()
    dirs_world = env.dirs_body @ quat_to_rotmat(quat).T
    proj = dirs_world[:, 0] * wall_normal_right[0] + dirs_world[:, 1] * wall_normal_right[1]
    targets = execute_skill("push_against_wall", quat, env.dirs_body, env.max_extend,
                            wall_normal=wall_normal_right)
    ext_toward = float(targets[proj < -0.30].mean())   # rods facing the wall
    ext_away = float(targets[proj > 0.30].mean())      # rods facing away
    print(f"8. push_wall(R):  wall-side rods={ext_toward:.3f}m  far-side rods={ext_away:.3f}m")
    assert ext_toward > 2.0 * ext_away, (
        f"FAIL: wall-side rods must extend far more than far-side; "
        f"got {ext_toward:.3f} vs {ext_away:.3f}")

    # 9. jump_up (crouch → takeoff → airborne)
    obs, info = env.reset(seed=42)
    start_z = env.data.qpos[2]
    max_z = start_z
    for step in range(60):
        quat = env.data.qpos[3:7]
        if step < 20:
            phase = "crouch"
        elif step < 32:
            phase = "takeoff"
        else:
            phase = "airborne"
        targets = execute_skill("jump_up", quat, env.dirs_body, env.max_extend, phase=phase)
        env.step(targets)
        max_z = max(max_z, env.data.qpos[2])
    jump_height = max_z - start_z
    print(f"9. jump_up:       peak_z={max_z:.3f}m  net_lift=+{jump_height*100:.1f}cm")
    assert jump_height > 0.20, f"FAIL: Expected >20cm vertical jump, got {jump_height*100:.1f}cm"

    # 10. jump_forward_while_stopped (crouch → takeoff → airborne)
    obs, info = env.reset(seed=42)
    start_pos = env.data.qpos[0:3].copy()
    max_z = start_pos[2]
    for step in range(80):
        quat = env.data.qpos[3:7]
        if step < 20:
            phase = "crouch"
        elif step < 32:
            phase = "takeoff"
        elif env.data.qpos[2] > 0.28:
            phase = "airborne"
        else:
            phase = "landing"
        targets = execute_skill("jump_forward_while_stopped", quat, env.dirs_body, env.max_extend,
                                d_hat=d_forward, phase=phase)
        env.step(targets)
        max_z = max(max_z, env.data.qpos[2])
    end_pos = env.data.qpos[0:3].copy()
    fwd_dist = end_pos[0] - start_pos[0]
    print(f"10. jump_fwd_stop: peak_z={max_z:.3f}m  Δx={fwd_dist:+.3f}m")
    assert max_z > 0.30, f"FAIL: Expected peak > 0.30m, got {max_z:.3f}"

    # 11. jump_forward_while_moving (sprint → dip → launch → airborne → landing)
    obs, info = env.reset(seed=42)
    start_pos = env.data.qpos[0:3].copy()
    max_z = start_pos[2]
    for step in range(200):
        quat = env.data.qpos[3:7]
        if step < 55:
            phase = "sprint"
        elif step < 62:
            phase = "dip"
        elif step < 75:
            phase = "launch"
        elif env.data.qpos[2] > 0.28:
            phase = "airborne"
        else:
            phase = "landing"
        targets = execute_skill("jump_forward_while_moving", quat, env.dirs_body, env.max_extend,
                                d_hat=d_forward, phase=phase)
        env.step(targets)
        max_z = max(max_z, env.data.qpos[2])
    end_pos = env.data.qpos[0:3].copy()
    fwd_dist = end_pos[0] - start_pos[0]
    print(f"11. jump_fwd_mov:  peak_z={max_z:.3f}m  Δx={fwd_dist:+.3f}m  vx={env.data.qvel[0]:+.2f}m/s")
    assert max_z > 0.35, f"FAIL: Expected peak > 0.35m, got {max_z:.3f}"
    assert fwd_dist > 0.50, f"FAIL: Expected >0.50m forward, got {fwd_dist:.3f}m"

    env.close()

    # 8b. push_against_wall against a real maze wall (separate maze env).
    test_wall_push_in_maze()

    # 12. fall_down off the skill-course platform (separate course env).
    test_fall_down_off_platform()

    print("\n" + "=" * 70)
    print("  ✅ ALL 12 SKILLS VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
