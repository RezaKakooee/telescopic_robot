"""Radial-sphere locomotion package.

Public API:
    from radial_sphere import RadialSphereEnv, GymCompatWrapper, make_compat_env
    from radial_sphere import load_config, load_config_dict
    # or via the gym registry:
    import gymnasium as gym; env = gym.make("RadialSphere-v0")

Agents (root scripts): ``random_agent.py``, ``heuristic_agent.py``.

Modular pieces (see each module's docstring):
    config       — load config.yaml (single source of truth) into a namespace
    geometry     — Fibonacci sphere, sinusoidal path, quaternion math (pure)
    mjcf         — MJCF XML generation for the radial-sphere robot
    controller   — scripted open-loop locomotion policy (pure)
    action       — action space + normalised action ↔ dof-target mapping
    observation  — observation space + observation builder
    reward       — progress + success reward
    render       — TensorState → uint8 RGB frame (handler camera)
    scenario     — task specs: path navigation / goal finding (+ generators)
    snapshot     — storage_local run dirs + code/config snapshot
    radial_sphere — the gym ``RadialSphereEnv`` composing the above
"""
from __future__ import annotations

from .action import ActionModel
from .config import load_config, load_config_dict
from .controller import bar_targets, desired_direction
from .geometry import (
    PATH_AMPLITUDE,
    PATH_LENGTH,
    PATH_WAVES,
    fibonacci_sphere,
    path_xy,
    quat_to_rotmat,
    sample_path,
    sample_roundtrip,
)
from .mjcf import build_robot_mjcf, rolling_radius
from .observation import ObservationModel
from .radial_sphere import RadialSphereEnv, GymCompatWrapper, make_compat_env
from .render import Renderer, VideoRecorder
from .reward import RewardModel
from .scenario import (Scenario, generate_scenario, path_scenario, goal_scenario,
                       roundtrip_scenario, KINDS)
from .snapshot import make_run_dir, save_code
from ._gym import gym

__all__ = [
    "RadialSphereEnv", "GymCompatWrapper", "make_compat_env",
    "load_config", "load_config_dict",
    "Scenario", "generate_scenario", "path_scenario", "goal_scenario",
    "roundtrip_scenario", "KINDS",
    "make_run_dir", "save_code",
    "ActionModel", "ObservationModel", "RewardModel", "Renderer", "VideoRecorder",
    "fibonacci_sphere", "path_xy", "sample_path", "sample_roundtrip", "quat_to_rotmat",
    "PATH_LENGTH", "PATH_AMPLITUDE", "PATH_WAVES",
    "build_robot_mjcf", "rolling_radius", "desired_direction", "bar_targets",
]


def _register():
    try:
        ids = {spec.id for spec in gym.envs.registry.values()}
    except Exception:
        ids = set()
    max_steps = int(load_config().env.max_steps)
    if "RadialSphere-v0" not in ids:
        gym.register(id="RadialSphere-v0",
                     entry_point="radial_sphere:RadialSphereEnv", max_episode_steps=max_steps)
    if "RadialSphere-v0-compat" not in ids:
        gym.register(id="RadialSphere-v0-compat",
                     entry_point="radial_sphere:make_compat_env", max_episode_steps=max_steps)


_register()
