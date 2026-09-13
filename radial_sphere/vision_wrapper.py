"""Vision observation wrapper for RoboBall: provides multimodal Gym observation space.

Wraps any RoboBall environment (e.g. SkillArbitrationEnv, MujocoRadialSphereEnv)
so that reset() and step() return a spaces.Dict containing:
- 'image': uint8 RGB image (H, W, 3) or (3, H, W) from a virtual camera (front, birdview, chase).
- 'state': low-dim float32 kinematics (11-D) or wrapped environment state.
- 'language_instruction': string instruction for Vision-Language-Action (VLA) models.
"""
from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import mujoco

from ._gym import gym, spaces

logger = logging.getLogger("vision_wrapper")


class RoboBallVisionWrapper(gym.Wrapper):
    """Gym wrapper that adds visual camera observations to the observation space."""

    def __init__(
        self,
        env: gym.Env,
        camera_name: str = "front",
        image_size: tuple[int, int] = (256, 256),
        channels_first: bool = False,
        state_type: str = "kinematics",
        language_instruction: str = "Navigate through the course to reach the goal pad.",
    ):
        super().__init__(env)
        self.camera_name = camera_name.lower()
        self.height, self.width = image_size
        self.channels_first = bool(channels_first)
        self.state_type = state_type
        self.language_instruction = language_instruction

        self._cam_heading: float | None = None
        self._renderer: mujoco.Renderer | None = None

        # Resolve underlying MuJoCo model & data
        self._underlying = self._find_mujoco_env()

        # Build Dict observation space
        img_shape = (3, self.height, self.width) if self.channels_first else (self.height, self.width, 3)
        image_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)

        if self.state_type == "kinematics":
            # 11-D vector: [pos_x, pos_y, vel_x, vel_y, rel_goal_x, rel_goal_y, dist_goal, quat_w, quat_x, quat_y, quat_z]
            state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        elif self.state_type == "raw":
            state_space = self.env.observation_space
        elif self.state_type == "both":
            state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown state_type '{self.state_type}', expected 'kinematics', 'raw', or 'both'")

        obs_dict: dict[str, spaces.Space] = {
            "image": image_space,
            "state": state_space,
        }
        if self.state_type == "both":
            obs_dict["raw_state"] = self.env.observation_space

        self.observation_space = spaces.Dict(obs_dict)

    def _find_mujoco_env(self):
        cur = self.env
        while cur is not None:
            if hasattr(cur, "model") and hasattr(cur, "data"):
                return cur
            if hasattr(cur, "env"):
                cur = cur.env
            else:
                break
        raise AttributeError("Could not find underlying MuJoCo environment with 'model' and 'data'")

    @property
    def renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._underlying.model, height=self.height, width=self.width)
        elif self._renderer.height != self.height or self._renderer.width != self.width:
            self._renderer.close()
            self._renderer = mujoco.Renderer(self._underlying.model, height=self.height, width=self.width)
        return self._renderer

    def _get_goal_xy(self) -> np.ndarray:
        if hasattr(self._underlying, "scenario") and hasattr(self._underlying.scenario, "goal"):
            return np.asarray(self._underlying.scenario.goal, dtype=np.float32)[:2]
        return np.array([0.0, 0.0], dtype=np.float32)

    def _get_kinematics(self) -> np.ndarray:
        data = self._underlying.data
        pos = data.qpos[:3].astype(np.float32)
        vel = data.qvel[:3].astype(np.float32)
        quat = data.qpos[3:7].astype(np.float32)
        ball_xy = pos[:2]
        goal_xy = self._get_goal_xy()

        rel_goal = goal_xy - ball_xy
        dist_goal = float(np.linalg.norm(rel_goal))

        return np.concatenate([
            ball_xy,
            vel[:2],
            rel_goal,
            np.array([dist_goal], dtype=np.float32),
            quat,
        ]).astype(np.float32)

    # Maps VLA wrapper camera names to MujocoRadialSphereEnv camera_name strings.
    # These use the proven working camera system in mujoco_env.py with correct
    # heading-relative azimuth, steadicam smoothing, and elevation angles.
    _CAMERA_MAP: dict[str, str] = {
        "chase": "chase",
        "rear": "chase",
        "behind": "chase",
        "front": "side_front",
        "forward": "side_front",
        "front_facing": "side_front",
        "birdview": "bird_chase",
        "bird_eye": "bird_chase",
        "topdown": "bird_chase",
        "iso": "fixed_angle_close_3d",
        "fixed_angle_close_3d": "fixed_angle_close_3d",
        "cinematic": "cinematic_chase_3d",
    }

    def render_camera_frame(self) -> np.ndarray:
        """Render RGB frame for the active camera mode.

        Delegates to MujocoRadialSphereEnv.render() for known cameras,
        which has proven-correct heading-relative azimuth and steadicam smoothing.
        """
        # Look up the underlying camera name
        mujoco_cam_name = self._CAMERA_MAP.get(self.camera_name, self.camera_name)

        # Save & temporarily override the underlying env's render_size so we
        # get the resolution we want, then restore it afterward.
        underlying = self._underlying
        orig_size = getattr(underlying, "render_size", None)
        orig_renderer = underlying.renderer

        # Force our desired resolution
        underlying.render_size = (self.height, self.width)
        if orig_renderer is not None and (orig_renderer.height != self.height or orig_renderer.width != self.width):
            underlying.renderer = None  # force re-creation at correct size

        try:
            frame = underlying.render(camera_name=mujoco_cam_name)
        finally:
            # Restore original state
            underlying.render_size = orig_size
            # Don't close the new renderer — let it be reused next frame

        if self.channels_first:
            frame = np.transpose(frame, (2, 0, 1))

        return frame

    def _build_observation(self, raw_obs: np.ndarray) -> dict[str, Any]:
        image = self.render_camera_frame()
        kinematics = self._get_kinematics()

        obs: dict[str, Any] = {
            "image": image,
        }

        if self.state_type == "kinematics":
            obs["state"] = kinematics
        elif self.state_type == "raw":
            obs["state"] = raw_obs
        elif self.state_type == "both":
            obs["state"] = kinematics
            obs["raw_state"] = raw_obs

        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._cam_heading = None
        raw_obs, info = self.env.reset(seed=seed, options=options)
        info["language_instruction"] = self.language_instruction
        obs = self._build_observation(raw_obs)
        return obs, info

    def step(self, action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        info["language_instruction"] = self.language_instruction
        obs = self._build_observation(raw_obs)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        super().close()


def make_vision_env(
    cfg=None,
    scenario=None,
    camera_name: str = "front",
    image_size: tuple[int, int] = (256, 256),
    channels_first: bool = False,
    state_type: str = "kinematics",
    language_instruction: str = "Navigate through the course to reach the goal pad.",
    **env_kwargs,
) -> RoboBallVisionWrapper:
    """Create a Vision-Language-Action (VLA) ready environment with multimodal observation space.

    Returns:
        RoboBallVisionWrapper around SkillArbitrationEnv.
    """
    from .skill_arbitration_env import SkillArbitrationEnv

    env = SkillArbitrationEnv(config=cfg, scenario=scenario, **env_kwargs)
    return RoboBallVisionWrapper(
        env,
        camera_name=camera_name,
        image_size=image_size,
        channels_first=channels_first,
        state_type=state_type,
        language_instruction=language_instruction,
    )

