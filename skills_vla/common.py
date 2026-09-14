"""Shared geometry and the bridge from a `RobotState` into the `skills` library.

`skills_vla` owns the policy-facing API (egocentric heading, bounded params,
hidden jump phases). The rod mechanics come from `skills/`, which is
calibrated and tested. Nothing in this package computes rod targets itself.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from radial_sphere.geometry import quat_to_rotmat
from skills.runner import PHASE_SCHEDULES
from .base import RobotState


def ego_to_world_heading(ego_angle: float, camera_heading: float = 0.0) -> np.ndarray:
    """Convert an egocentric steering angle (relative to camera forward) to a world 2D vector.

    Parameters
    ----------
    ego_angle : float
        Steering angle in radians relative to camera view (0 = forward, +pi/2 = left, -pi/2 = right).
    camera_heading : float
        World yaw of the active camera in radians.
    """
    theta = float(camera_heading) + float(ego_angle)
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)


def resolve_heading(camera_heading: float, heading_ego: float, kwargs: dict[str, Any]) -> np.ndarray:
    """World heading from either an explicit `d_world`/`d_hat` or the egocentric angle."""
    d = kwargs.pop("d_world", None)
    if d is None:
        d = kwargs.pop("d_hat", None)
    if d is None:
        return ego_to_world_heading(heading_ego, camera_heading)
    d = np.asarray(d, dtype=np.float64)[:2]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-6 else np.array([1.0, 0.0])


def rods_world(state: RobotState) -> np.ndarray:
    """Return (n_bars, 3) rod direction vectors in the world reference frame."""
    return np.asarray(state.dirs_body) @ quat_to_rotmat(state.quat).T


def surface_frame(state: RobotState, along: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project all rods onto the local surface frame: (u_long, u_lat, u_into)."""
    n = state.floor_normal()
    dirs = rods_world(state)
    a = np.asarray(along, dtype=np.float64)
    if a.shape[0] == 2:
        a = np.array([a[0], a[1], 0.0], dtype=np.float64)
    a = a - np.dot(a, n) * n
    mag = float(np.linalg.norm(a))
    if mag < 1e-9:
        seed = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(seed, n)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        a = seed - np.dot(seed, n) * n
        mag = float(np.linalg.norm(a))
    a = a / mag
    lat = np.cross(n, a)
    return dirs @ a, dirs @ lat, dirs @ n


def finish_targets(targets: np.ndarray, state: RobotState) -> np.ndarray:
    """Clamp rod extension targets strictly within hardware/simulation bounds."""
    t = np.asarray(targets, dtype=np.float32)
    return np.clip(t, state.min_offset, state.max_extend)


def call_skill(fn: Callable[..., np.ndarray], state: RobotState, **kwargs: Any) -> np.ndarray:
    """Call a pure `skills` function with the positional state it expects."""
    out = fn(np.asarray(state.quat, dtype=np.float64), np.asarray(state.dirs_body, dtype=np.float64),
             float(state.max_extend), **kwargs)
    return finish_targets(out, state)


def jump_phase(schedule_name: str, substep: int, state: RobotState) -> tuple[str, int]:
    """Phase name from the verified `skills.runner` schedule, plus its step budget.

    The runner schedules read the core height above flat ground. Subtracting
    `ground_z` keeps elevated platforms from looking like perpetual flight.
    """
    phase_fn, budget = PHASE_SCHEDULES[schedule_name]
    core_z = float(state.core_z if state.core_z is not None else 0.22) - float(state.ground_z)
    return phase_fn(int(substep), core_z), budget


def state_from_env(env, ground_z: float = 0.0) -> RobotState:
    """Build a `RobotState` from a live `MujocoRadialSphereEnv`."""
    cfg_robot = getattr(getattr(env, "cfg", None), "robot", None)
    contact = env.get_rod_contact_forces() if hasattr(env, "get_rod_contact_forces") else None
    clear = env.get_terrain_clearances() if hasattr(env, "get_terrain_clearances") else None
    return RobotState(
        quat=env.data.qpos[3:7].copy(),
        dirs_body=env.dirs_body,
        max_extend=float(env.max_extend),
        lin_vel=env.data.qvel[0:3].copy(),
        core_z=float(env.data.qpos[2]),
        core_vz=float(env.data.qvel[2]),
        contact_forces=contact,
        terrain_clearances=clear,
        rod_mechanism=str(getattr(cfg_robot, "rod_mechanism", "multi_stage")),
        ground_z=float(ground_z),
    )
