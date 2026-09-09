"""Tests for the stay_in_boundary skill primitive and scenario.

Verifies:
1. Pure-function behavior across center roaming and perimeter containment regimes.
2. Skill registry aliases (stay_in_boundary, stay_within_boundary, boundary_containment).
3. Wall-less boundary scenario generation (zero walls, high-density visual floor markings).
4. Closed-loop physics simulation: robot roams slowly, takes diverse actions, and
   strictly stays inside the circular boundary with zero wall collisions.
"""

import numpy as np

from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario, Scenario
from skills import execute_skill, SKILL_REGISTRY


def test_stay_in_boundary_pure_function():
    """Verify stay_in_boundary produces valid targets and handles regimes."""
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    dirs_body = np.random.randn(60, 3)
    dirs_body /= np.linalg.norm(dirs_body, axis=1, keepdims=True)
    max_extend = 0.20

    # 1. Roaming regime at center: diverse actions across step counts
    sub_skills_seen = set()
    for step in [10, 100, 180, 240, 300]:
        targets, meta = execute_skill(
            "stay_in_boundary",
            quat, dirs_body, max_extend,
            ball_xy=np.array([0.0, 0.0]),
            lin_vel=np.array([0.2, 0.0]),
            step_count=step,
            return_metadata=True,
        )
        assert targets.shape == (60,)
        assert np.all(targets >= 0.02)
        assert np.all(targets <= max_extend + 1e-4)
        assert not meta["is_boundary_active"]
        sub_skills_seen.add(meta["sub_skill"])

    assert len(sub_skills_seen) >= 3, f"Expected diverse actions, got: {sub_skills_seen}"

    # 2. Critical perimeter approach heading outward -> emergency brake
    targets, meta = execute_skill(
        "stay_in_boundary",
        quat, dirs_body, max_extend,
        ball_xy=np.array([1.88, 0.0]),
        lin_vel=np.array([0.35, 0.0]),
        boundary_radius=2.0,
        return_metadata=True,
    )
    assert meta["is_boundary_active"]
    assert meta["sub_skill"] == "stop"
    assert meta["action_name"] == "boundary_emergency_brake"

    # 3. Perimeter approach requiring turn inward
    targets, meta = execute_skill(
        "stay_in_boundary",
        quat, dirs_body, max_extend,
        ball_xy=np.array([1.65, 0.0]),
        lin_vel=np.array([0.25, 0.25]),
        boundary_radius=2.0,
        return_metadata=True,
    )
    assert meta["is_boundary_active"]
    assert meta["sub_skill"] in {"turn", "curve", "move"}


def test_stay_in_boundary_registry_aliases():
    """Verify registry aliases resolve to the same function."""
    assert "stay_in_boundary" in SKILL_REGISTRY
    assert "stay_within_boundary" in SKILL_REGISTRY
    assert "boundary_containment" in SKILL_REGISTRY
    assert SKILL_REGISTRY["stay_in_boundary"] is SKILL_REGISTRY["stay_within_boundary"]
    assert SKILL_REGISTRY["stay_in_boundary"] is SKILL_REGISTRY["boundary_containment"]


def test_boundary_scenario_generation():
    """Verify wall-less boundary scenario has zero physical walls and visual decals."""
    sc = generate_scenario("boundary", None, radius=2.0)
    assert isinstance(sc, Scenario)
    assert sc.kind == "boundary"
    # STRICT REQUIREMENT: ZERO physical walls
    assert len(sc.walls) == 0, f"Expected 0 walls, found {len(sc.walls)}"
    # High-density floor decals
    assert len(sc.yardlines) >= 64
    assert sc.path_pts.shape[1] == 2


def test_boundary_physics_containment():
    """Run closed-loop physics simulation and verify ball stays inside boundary."""
    radius = 2.0
    sc = generate_scenario("boundary", None, radius=radius)
    env = MujocoRadialSphereEnv(scenario=sc, render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    radii = []
    speeds = []
    sub_skills_observed = set()

    for step in range(350):
        quat = env.data.qpos[3:7].copy()
        ball_xy = env.data.qpos[0:2].copy()
        lin_vel = env.data.qvel[0:2].copy()

        r = float(np.linalg.norm(ball_xy))
        speed = float(np.linalg.norm(lin_vel))
        radii.append(r)
        speeds.append(speed)

        targets, meta = execute_skill(
            "stay_in_boundary",
            quat,
            env.dirs_body,
            env.max_extend,
            ball_xy=ball_xy,
            lin_vel=lin_vel,
            boundary_radius=radius,
            step_count=step,
            return_metadata=True,
        )

        sub_skills_observed.add(meta["sub_skill"])
        env.step(targets)

    env.close()

    max_r = max(radii)
    mean_speed = np.mean(speeds)

    print(f"Max radius reached: {max_r:.3f}m / {radius:.2f}m boundary")
    print(f"Mean speed: {mean_speed:.3f}m/s")
    print(f"Sub-skills observed during run: {sub_skills_observed}")

    # Guarantees: strictly within boundary (r < radius)
    assert max_r < radius, f"Robot violated boundary! max_r={max_r:.3f} >= {radius}"
    assert mean_speed < 0.65, f"Robot moved too fast! mean_speed={mean_speed:.3f}"
    assert len(sub_skills_observed) >= 3, f"Expected diverse actions, observed: {sub_skills_observed}"


def load_tests(loader, tests, pattern):
    """Expose this module's plain test functions to `unittest discover`."""
    import sys
    from tests._function_suite import suite_from_module
    return suite_from_module(sys.modules[__name__])


if __name__ == "__main__":
    test_stay_in_boundary_pure_function()
    test_stay_in_boundary_registry_aliases()
    test_boundary_scenario_generation()
    test_boundary_physics_containment()
    print("\n✅ All boundary skill tests passed successfully!")
