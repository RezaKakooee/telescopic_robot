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
from .config import load_config, load_config_cli, load_config_dict
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
from .observation import ObservationModel
try:
    from .render import Renderer, VideoRecorder, MultiVideoRecorder
except (ModuleNotFoundError, ImportError):
    Renderer = VideoRecorder = MultiVideoRecorder = None
try:
    from .radial_sphere import RadialSphereEnv, GymCompatWrapper, make_compat_env
except ModuleNotFoundError:
    RadialSphereEnv = GymCompatWrapper = make_compat_env = None
from .reward import RewardModel
from .scenario import (Scenario, generate_scenario, path_scenario, goal_scenario,
                       roundtrip_scenario, obstacle_scenario, maze_scenario, KINDS)
from .log import setup_logging
from .run_id import build_run_id, normalize_name
from .snapshot import make_run_dir, save_code
try:
    from .steering import SteeringEnv
except ModuleNotFoundError:
    SteeringEnv = None
from .mujoco_mjcf import build_mujoco_scene_mjcf
from .mujoco_env import MujocoRadialSphereEnv
from .mujoco_steering import MujocoSteeringEnv
from .mujoco_lowlevel_env import MujocoLowLevelEnv
from .vision_wrapper import RoboBallVisionWrapper, make_vision_env
try:
    from .skill_arbitration_env import SkillArbitrationEnv
except (ModuleNotFoundError, ImportError):
    SkillArbitrationEnv = None
from ._gym import gym

__all__ = [
    "RadialSphereEnv", "GymCompatWrapper", "make_compat_env", "SteeringEnv",
    "MujocoRadialSphereEnv", "MujocoSteeringEnv", "MujocoLowLevelEnv", "SkillArbitrationEnv",
    "RoboBallVisionWrapper", "make_vision_env",
    "build_mujoco_scene_mjcf",
    "load_config", "load_config_cli", "load_config_dict",
    "setup_logging", "build_run_id", "normalize_name",
    "Scenario", "generate_scenario", "path_scenario", "goal_scenario",
    "roundtrip_scenario", "obstacle_scenario", "maze_scenario", "KINDS",
    "make_run_dir", "save_code",
    "ActionModel", "ObservationModel", "RewardModel", "Renderer", "VideoRecorder", "MultiVideoRecorder",
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
    if "RadialSphereMujoco-v0" not in ids:
        gym.register(id="RadialSphereMujoco-v0",
                     entry_point="radial_sphere:MujocoRadialSphereEnv", max_episode_steps=max_steps)


_register()
