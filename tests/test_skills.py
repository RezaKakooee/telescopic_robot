"""Verification test: Run every skill for ~30 steps and verify it produces the expected motion."""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
from pathlib import Path
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.geometry import quat_to_rotmat
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import (
    generate_scenario, skill_course_platform, stairs_course_geometry,
)
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


def test_fall_down_pillar_rolloff(seed=1):
    """Physics check for fall_down on narrow pillars: roll off pillar 2 (1.05m)
    onto pillar 3 (0.65m) and verify it lands squarely on the 0.9m pad without
    pole-vaulting or overshooting."""
    from radial_sphere.scenario import pillar_course_columns

    cfg = load_config("configs/rl/pillar_course.yaml")
    scenario = generate_scenario("pillar_course", cfg, seed=1)
    cols = pillar_course_columns(cfg)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=seed)

    cur, target = cols[2], cols[3]
    drop = cur["height"] - target["height"]
    deck_core = target["height"] + 0.19
    env.data.qpos[0] = cur["far"] - 0.25
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = cur["height"] + 0.19
    env.data.qvel[:] = 0

    phase, ps = "edge", 0
    d_fwd = np.array([1.0, 0.0])
    for _ in range(1500):
        z = float(env.data.qpos[2])
        if phase == "edge" and z < cur["height"] + 0.19 - 0.06:
            phase, ps = "freefall", 0
        elif phase == "freefall" and z < deck_core + 0.10:
            phase, ps = "absorb", 0
        elif phase == "absorb" and z < deck_core + 0.03:
            phase, ps = "brake", 0
        ps += 1

        if phase == "brake":
            t = execute_skill("stop", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                              lin_vel=env.data.qvel[0:2].copy(), stop_distance=0.15)
        else:
            t = execute_skill("fall_down", env.data.qpos[3:7].copy(), env.dirs_body,
                              env.max_extend, d_hat=d_fwd, phase=phase,
                              drop_height=drop, edge_speed=0.35, gear=0.5)
        env.step(t)
        if phase == "brake" and ps > 80:
            break

    for _ in range(60):
        env.step(execute_skill("stop", env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
                               lin_vel=env.data.qvel[0:2].copy()))

    x, y, z = float(env.data.qpos[0]), float(env.data.qpos[1]), float(env.data.qpos[2])
    on = (target["near"] < x < target["far"]
          and target["height"] + 0.10 < z < target["height"] + 0.55)
    env.close()

    print(f"12b. fall_down(pillar): landed x={x:.2f}m y={y:+.2f}m z={z:.2f}m "
          f"on pillar 3 [{target['near']:.2f}, {target['far']:.2f}]m")
    assert on, f"FAIL: Expected to land on pillar 3 [{target['near']:.2f}, {target['far']:.2f}], got x={x:.2f}m"


def test_circle_trajectory(radius=1.8, target_speed=1.0, steps=600):
    """Physics check for circle: drive in a circle of radius R, verify circularity
    (radial variance < 10cm, mean radius within 5% of target)."""
    cfg = load_config("configs/rl/standing_jump_showcase.yaml")
    scenario = generate_scenario("goal", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    env.data.qpos[0] = radius
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.20
    env.data.qvel[:] = 0

    radii = []
    prev_th = 0.0
    accum_th = 0.0

    for step in range(steps):
        pos = env.data.qpos[0:2].copy()
        quat = env.data.qpos[3:7].copy()
        r = float(np.linalg.norm(pos))
        radii.append(r)

        th = np.arctan2(pos[1], pos[0])
        d_th = th - prev_th
        if d_th > np.pi: d_th -= 2 * np.pi
        elif d_th < -np.pi: d_th += 2 * np.pi
        accum_th += abs(d_th)
        prev_th = th

        targets = execute_skill("circle", quat, env.dirs_body, env.max_extend,
                                ball_xy=pos, radius=radius, speed=target_speed)
        env.step(targets)

    env.close()

    r_arr = np.array(radii[50:])
    mean_r = float(np.mean(r_arr))
    std_r = float(np.std(r_arr))
    laps = accum_th / (2 * np.pi)

    print(f"14. circle:        target R={radius:.2f}m -> achieved {mean_r:.3f}±{std_r:.3f}m  "
          f"laps={laps:.2f}  (radius error {abs(mean_r - radius)*100:.1f}cm)")
    assert abs(mean_r - radius) < 0.15, f"FAIL: Expected mean radius near {radius:.2f}m, got {mean_r:.3f}m"
    assert std_r < 0.15, f"FAIL: Expected tight circular orbit, got std={std_r:.3f}m"
    assert laps > 0.5, f"FAIL: Expected at least 0.5 lap, got {laps:.2f}"


def test_straddle_gap_traverse(gap_width=0.22, box_height=0.25, steps=500):
    """Physics check for straddle_gap: robot spans two parallel ledges (Box 1 and Box 2)
    with a deep central chasm directly underneath. Verify it stays elevated on top of the
    ledges, keeps centered over the gap, and moves forward continuously."""
    import mujoco

    cfg = load_config("configs/rl/gap_bridge.yaml")
    scenario = generate_scenario("gap_bridge", cfg, seed=42)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=42)

    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = box_height + 0.19
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    for _ in range(25):
        env.step(execute_skill("straddle_gap", env.data.qpos[3:7].copy(), env.dirs_body,
                               env.max_extend, speed=0.0))

    d_fwd = np.array([1.0, 0.0])
    for _ in range(steps):
        pos = env.data.qpos[0:3].copy()
        quat = env.data.qpos[3:7].copy()
        targets = execute_skill("straddle_gap", quat, env.dirs_body, env.max_extend,
                                d_hat=d_fwd, speed=1.3, lateral_offset=float(pos[1]))
        env.step(targets)

    end_x = float(env.data.qpos[0])
    end_y = float(env.data.qpos[1])
    end_z = float(env.data.qpos[2])
    env.close()

    print(f"15. straddle_gap:  Δx=+{end_x:.2f}m  y={end_y:+.3f}m  z={end_z:.3f}m  "
          f"(deck={box_height:.2f}m, gap={gap_width*100:.0f}cm)")
    assert end_x > 1.5, f"FAIL: Expected forward progress > 1.5m, got {end_x:.2f}m"
    assert end_z > box_height + 0.10, f"FAIL: Expected to stay on top of boxes (z > {box_height+0.10:.2f}m), got z={end_z:.3f}m"
    assert abs(end_y) < 0.12, f"FAIL: Expected to stay centered over gap (|y| < 0.12m), got y={end_y:.3f}m"



def test_chimney_climb_vertical(seed=5):
    """Wall-jump up a 0.40 m chimney under free physics, burst out over the
    lower wall, land on its top and stop there. No pinned state."""
    import mujoco
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "skills"))
    from run_chimney import climb_chimney

    cfg = load_config("configs/rl/chimney.yaml")
    scenario = generate_scenario("chimney", cfg, seed=1)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=100_000)
    env.reset(seed=1)
    env.data.qpos[0:3] = [0.0, 0.0, 0.20]
    qq = np.random.default_rng(seed).normal(size=4)
    env.data.qpos[3:7] = qq / np.linalg.norm(qq)
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(40):
        env.step(np.zeros(60, dtype=np.float32))

    boxes = np.asarray(scenario.steps, dtype=float)
    low = boxes[int(np.argmin(boxes[:, 4]))]
    top, low_sign = float(low[4]), int(np.sign(low[1]))
    box_y = (float(abs(low[1]) - low[3]), float(abs(low[1]) + low[3]))
    r = climb_chimney(env, top=top, box_y=box_y, low_sign=low_sign)
    y, z = float(env.data.qpos[1]), float(env.data.qpos[2])
    env.close()

    print(f"16. chimney_climb: peak {r['peak']:.2f}m, cleared the {top:.1f}m lip {r['reached']}, "
          f"on the box top {r['on_top']} at y {y:+.2f} z {z:.2f}  (t {r['t_down']}s)")
    assert r["reached"], f"FAIL: never cleared the lip, peak {r['peak']:.2f}"
    assert r["on_top"] and top + 0.10 < z < top + 0.55, f"FAIL: not on the box top, z {z:.2f}"
    assert box_y[0] < low_sign * y < box_y[1], f"FAIL: off the box top, y {y:+.2f}"

def test_wall_of_death(seconds=65.0, seed=1):
    """Spiral up the drome bowl and hold a high orbit.

    The bar is a sustained ride, not a peak. A robot that flings itself high
    once and falls back has not ridden anything, and every earlier attempt at
    this arena did exactly that.

    The spiral opens at 0.10 m of radius per second by design, so the climb
    takes most of a minute. Do not shorten this run: at 45 s it is still on
    its way up and the last-8-seconds window measures the climb, not the ride.
    """
    import mujoco
    from omegaconf import OmegaConf
    from skills.wall_of_death import Bowl, advance_radius, surface_frame, wall_of_death

    cfg = load_config("configs/rl/motordrome.yaml")
    OmegaConf.set_struct(cfg, False)
    scenario = generate_scenario("motordrome", cfg, seed=seed)
    md = scenario.motordromes[0]
    bowl = Bowl.from_motordrome(md)

    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10 ** 7)
    env.reset(seed=seed)
    env.data.qpos[0] = 0.55
    env.data.qpos[1] = 0.0
    env.data.qpos[2] = 0.16 + 0.20 * env.max_extend
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    r_cmd, laps, prev_th, peak_z = 0.55, 0.0, 0.0, 0.0
    zs, vs = [], []
    for i in range(int(seconds * 100)):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        n_hat, _d = surface_frame(env.model, env.data, pos)
        targets, info = wall_of_death(
            env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend, pos, vel,
            r_cmd=r_cmd, bowl=bowl, normal=n_hat, wall_radius=float(md[3]), ccw=True,
            steer_gain=0.30,
        )
        r_cmd = advance_radius(r_cmd, info["r"], info["speed"], bowl)
        env.step(targets)
        th = float(np.arctan2(pos[1], pos[0]))
        if i:
            laps += ((th - prev_th + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi)
        prev_th = th
        peak_z = max(peak_z, info["z"])
        zs.append(info["z"])
        vs.append(info["speed"])
    env.close()

    tail_z = zs[-800:]
    tail_v = vs[-800:]
    z_mean, z_min = float(np.mean(tail_z)), float(np.min(tail_z))
    v_mean = float(np.mean(tail_v))
    # The bar is the top of the *ride line*, not the rim. The boards curl up
    # past the ride line to meet the vertical wall, and nothing rides that.
    ride_top = bowl.height_at(bowl.ride_radius(v_mean))

    print(f"17. wall_of_death: peak {peak_z:.2f}m, holds {z_mean:.2f}m "
          f"(min {z_min:.2f}) at {v_mean:.2f} m/s, {abs(laps):.1f} laps "
          f"(ride line tops out at {ride_top:.2f}m, bowl {bowl.rim_z:.2f}m deep)")

    assert abs(laps) > 6.0, f"FAIL: expected over 6 laps, got {abs(laps):.1f}"
    assert peak_z > 1.10, f"FAIL: expected to climb past 1.10m, peaked at {peak_z:.2f}m"
    assert v_mean > 3.5, f"FAIL: expected over 3.5 m/s on the bank, got {v_mean:.2f}"
    assert z_mean > 0.90 * ride_top, (
        f"FAIL: expected to hold near the {ride_top:.2f}m top of the ride line, "
        f"held {z_mean:.2f}m")
    assert z_min > 0.70 * ride_top, (
        f"FAIL: fell back to {z_min:.2f}m; the ride is not sustained")


def test_wall_of_death_descent(seconds=75.0, descend_after=45.0):
    """Ride high, then wind the spiral in and park on the flat floor.

    Coming down is its own problem, not the climb played backwards. Going up,
    the bank has to be earned with speed. Coming down, the speed has to be
    given away, and a bank only lets go of it as fast as the robot can shed it.
    Ask for the descent too fast and the robot is left going quicker than the
    circle it has been handed, so it runs wide and climbs again.

    The test is that the robot is in charge the whole way down: no stretch of
    free fall, and a stop at the bottom rather than a stop against something.
    """
    from scripts.skills.run_motordrome_wall_of_death import run

    r = run(seconds=seconds, descend_after=descend_after, record_video=False)

    free_fall = float(np.sqrt(2 * 9.81 * r["peak_z"]))
    print(f"18. wall_of_death descent: from {r['peak_z']:.2f}m, fastest drop "
          f"{r['max_drop']:.2f} m/s (free fall would be {free_fall:.1f}), "
          f"down in {r['down_time']:.1f}s, ends z {r['z_end']:.2f}m at "
          f"{r['v_end']:.2f} m/s, parked {r['parked']}")

    assert r["peak_z"] > 1.5, (
        f"FAIL: never got high enough to test a descent, peaked at {r['peak_z']:.2f}m")
    assert r["max_drop"] < 0.45 * free_fall, (
        f"FAIL: dropped at {r['max_drop']:.2f} m/s against {free_fall:.1f} for a "
        f"free fall; that is falling, not descending")
    assert r["z_end"] < 0.35, f"FAIL: ended at {r['z_end']:.2f}m, not on the floor"
    assert r["v_end"] < 0.25, f"FAIL: ended still moving at {r['v_end']:.2f} m/s"
    assert r["parked"], "FAIL: never reached the flat floor to park"


def test_wall_run(seconds=18.0, repeats=2):
    """Jump at a wall, ride it on momentum, push off, land on an open cage.

    Tests the modular horizontal wall run skill:
    1. Standard multi-repeat flat wall run down the 62 m arena
    2. Curved arc wall run with extended wall contact
    3. Banked wall run
    """
    from scripts.skills.run_wall_run import run
    from types import SimpleNamespace
    from skills.wall_run import get_wall_frame

    # Geometry is part of the controller: wall segments are geom centre lines,
    # while rod reach must be measured to the near surface.
    flat_sc = SimpleNamespace(
        wall_mode="flat", walls=np.array([[0.0, 1.30, 10.0, 1.30]]),
        wall_thickness=0.12,
    )
    flat_gap, flat_n, flat_t = get_wall_frame(np.array([2.0, 0.0, 1.0]), flat_sc)
    assert np.isclose(flat_gap, 1.24), f"FAIL: flat surface gap {flat_gap:.3f}, expected 1.240"
    assert np.allclose(flat_n, [0.0, 1.0, 0.0])
    assert np.allclose(flat_t, [1.0, 0.0, 0.0])

    bank_sc = SimpleNamespace(
        wall_mode="banked", walls=np.array([[0.0, 1.30, 10.0, 1.30]]),
        wall_thickness=0.12, wall_height=3.0, wall_bank_deg=12.0,
    )
    bank_gap, bank_n, _ = get_wall_frame(np.array([2.0, 0.0, 1.0]), bank_sc)
    expected_bank_gap = 1.30 * np.cos(np.radians(12.0)) + 0.50 * np.sin(np.radians(12.0)) - 0.06
    assert np.isclose(bank_gap, expected_bank_gap)
    assert bank_n[2] > 0.0, "FAIL: banked wall normal must match the compiled upward-tilted +Y face"

    # 1. Multi-repeat flat wall run down the 62 m lane
    r_flat = run(mode="flat_multistep", seconds=seconds, repeats=repeats, record_video=False)

    print(f"19. wall_run (flat): {len(r_flat['runs'])} of {repeats} runs reached the wall; "
          + "; ".join(f"{g['secs']:.2f}s {g['along']:.2f}m at z{g['z'][0]:.2f}"
                      for g in r_flat["runs"])
          + f"; rods folded {r_flat['squash']*1000:.0f}mm, push off {r_flat['exit_v']:.2f} m/s, "
            f"settles at {r_flat['end_z']:.2f}m")

    assert len(r_flat["runs"]) == repeats, (
        f"FAIL: only {len(r_flat['runs'])} of {repeats} runs reached the wall")
    for k, g in enumerate(r_flat["runs"], 1):
        assert g["along"] > 0.45, (
            f"FAIL: run {k} moved only {g['along']:.2f} m along the wall")
        assert g["entry_z"] > 0.75, (
            f"FAIL: run {k} met the wall at {g['entry_z']:.2f} m, too low")
        assert g["airborne_fraction"] > 0.90, (
            f"FAIL: run {k} was only {g['airborne_fraction']:.0%} airborne during wall contact")
        assert g["contact_rods"] >= 6, (
            f"FAIL: run {k} used only {g['contact_rods']} wall rods; no multi-contact sequence")
        assert not g["core_contact"], f"FAIL: run {k} hit the wall with the core"
    assert 0.05 < r_flat["squash"] < 0.30, (
        f"FAIL: rods folded {r_flat['squash']*1000:.0f} mm; expected 50-300")
    assert r_flat["min_gap"] > 0.16, (
        f"FAIL: closest approach {r_flat['min_gap']:.3f} m puts the core into the wall")
    assert r_flat["exit_v"] > 1.0, (
        f"FAIL: pushed off at only {r_flat['exit_v']:.2f} m/s")
    assert r_flat["end_z"] < 0.50, (
        f"FAIL: ended at {r_flat['end_z']:.2f} m, still up on its landing cage")

    # 2. Curved arc wall run
    r_curv = run(mode="curved", seconds=6.0, repeats=1, record_video=False)
    assert len(r_curv["runs"]) == 1, "FAIL: curved wall run failed to contact wall"
    g_curv = r_curv["runs"][0]
    assert g_curv["along"] > 0.45, f"FAIL: curved run along={g_curv['along']:.2f}m < 0.45m"
    assert g_curv["entry_z"] > 0.70 and g_curv["airborne_fraction"] > 0.90
    assert g_curv["contact_rods"] >= 8 and not g_curv["core_contact"]

    # 3. Banked wall run
    r_bank = run(mode="banked", seconds=6.0, repeats=1, record_video=False)
    assert len(r_bank["runs"]) == 1, "FAIL: banked wall run failed to contact wall"
    g_bank = r_bank["runs"][0]
    assert g_bank["along"] > 0.50, f"FAIL: banked run along={g_bank['along']:.2f}m < 0.50m"
    assert g_bank["entry_z"] > 0.90 and g_bank["airborne_fraction"] > 0.90
    assert g_bank["contact_rods"] >= 6 and not g_bank["core_contact"]


def test_training_cones():
    """Slalom weave cleanly between 10 linear training cones with zero collisions."""
    from scripts.skills.run_training_cones import run

    r = run(speed=1.1, lateral_offset=0.80, lookahead=0.40, lateral_gain=5.0, record_video=False)

    print(f"20. training_cones: {r['n_cleared']} of {r['n_cones']} cones cleared, "
          f"contacts={r['contacts']}, min clearance={r['min_clearance']:.3f}m, "
          f"reached finish at x={r['final_pos'][0]:.2f}m")

    assert r["n_cleared"] == r["n_cones"], (
        f"FAIL: only cleared {r['n_cleared']} of {r['n_cones']} cones")
    assert r["contacts"] == 0, (
        f"FAIL: had {r['contacts']} cone collisions; clean slalom required")
    assert r["min_clearance"] >= 0.40, (
        f"FAIL: min clearance {r['min_clearance']:.3f}m is dangerously close (<0.40m)")
def test_curved_training_cones():
    """Slalom weave cleanly between 10 unevenly spaced cones along an S-curve track."""
    from scripts.skills.run_curved_training_cones import run

    r = run(speed=1.1, lateral_offset=0.80, lookahead=0.40, lateral_gain=5.0, record_video=False)

    print(f"21. curved_training_cones: {r['n_cleared']} of {r['n_cones']} cones cleared, "
          f"contacts={r['contacts']}, min clearance={r['min_clearance']:.3f}m, "
          f"reached finish at x={r['final_pos'][0]:.2f}m, y={r['final_pos'][1]:.2f}m")

    assert r["n_cleared"] == r["n_cones"], (
        f"FAIL: only cleared {r['n_cleared']} of {r['n_cones']} cones")
    assert r["contacts"] == 0, (
        f"FAIL: had {r['contacts']} cone collisions on curved course; clean slalom required")
    assert r["min_clearance"] >= 0.40, (
        f"FAIL: min clearance {r['min_clearance']:.3f}m is dangerously close (<0.40m)")
    assert r["final_pos"][0] >= 26.5, (
        f"FAIL: final x={r['final_pos'][0]:.2f}m did not reach the finish gate")


def test_stairs_climb():
    """Verify three real tread landings up and three controlled drops down."""
    from scripts.skills.run_stairs import run

    r = run(record_video=False)
    stair_geo = stairs_course_geometry(load_config("configs/rl/stairs_course.yaml"))

    print(f"22. stairs (step-by-step vaulting): climbed={r['steps_climbed']}/3, "
          f"descended={r['steps_descended']}/3, "
          f"peak z={r['peak_z']:.3f}m, final pos=({r['final_pos'][0]:.2f}, {r['final_pos'][1]:.2f})m, "
          f"core impacts={r['core_impacts']}")

    assert r["success"], "FAIL: composed stair traversal did not complete"
    assert r["steps_climbed"] == 3, f"FAIL: only climbed {r['steps_climbed']}/3 steps"
    assert r["steps_descended"] == 3, f"FAIL: only descended {r['steps_descended']}/3 steps"
    assert [x["geom"] for x in r["climbs"]] == [
        "stair_0_0", "stair_0_1", "stair_0_2",
    ]
    assert [x["geom"] for x in r["descents"]] == [
        "stair_1_1", "stair_1_2", "floor",
    ]
    assert all(x["verified"] for x in r["climbs"] + r["descents"]), (
        "FAIL: a milestone was counted without a verified target contact")
    assert all(x.get("attempts") == 1 for x in r["climbs"]), (
        f"FAIL: stair jumps required retries: {[x.get('attempts') for x in r['climbs']]}")
    assert r["peak_z"] >= 0.90, f"FAIL: peak height {r['peak_z']:.3f}m < 0.90m"
    assert r["final_pos"][0] >= stair_geo["finish_x"] - 0.50, (
        f"FAIL: final x={r['final_pos'][0]:.2f}m did not reach the configured finish")
    assert abs(r["final_pos"][1]) < 0.25, (
        f"FAIL: final lateral drift y={r['final_pos'][1]:.2f}m")
    assert r["core_impacts"] == 0, f"FAIL: had {r['core_impacts']} core impacts on stairs"
    assert r["end_speed"] < 0.20, f"FAIL: ended moving at {r['end_speed']:.2f}m/s"
    assert r["end_spin"] < 0.40, f"FAIL: ended spinning at {r['end_spin']:.2f}rad/s"


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

    # 12b. fall_down pillar roll-off (narrow pillar ladder).
    test_fall_down_pillar_rolloff()

    # 14. circle trajectory tracking.
    test_circle_trajectory()

    # 15. straddle_gap chasm traversal.
    test_straddle_gap_traverse()

    # 16. chimney_climb vertical traversal.
    test_chimney_climb_vertical()

    # 17. wall_of_death: spiral up the drome and hold the high orbit.
    test_wall_of_death()

    # 18. wall_of_death: and come back down under control.
    test_wall_of_death_descent()

    # 19. wall_run: the horizontal parkour wall run.
    test_wall_run()

    # 20. training_cones: the slalom weave between 10 obstacle cones.
    test_training_cones()

    # 21. curved_training_cones: slalom weave through 10 uneven cones on an S-curve.
    test_curved_training_cones()

    # 22. stairs: multi-step flight ascent, plateau, and compliant descent.
    test_stairs_climb()

    print("\n" + "=" * 70)
    print("  ✅ ALL 22 SKILL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
