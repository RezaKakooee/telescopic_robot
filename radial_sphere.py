"""Radial-sphere locomotion — RoboVerse port of my_mujoco/radial_sphere.py.

A sphere with N telescoping bars (Fibonacci-distributed on the unit sphere)
rolls along a sinusoidal path on the floor.  Each bar is a prismatic joint
with a position-controlled motor.  The controller extends back-pointing bars
(push), retracts front-pointing bars (slack), and biases the bottom bars
extended so the ball "stands" instead of rolling on the floor sphere.

Mapping to RoboVerse:
  * The sphere + bars are generated as an MJCF and wrapped in a RobotCfg.
  * The sinusoidal target path is drawn with small static spheres.
  * The locomotion controller runs in Python and pushes commands every step
    through handler.set_dof_targets(); handler.simulate() advances physics.

Only --sim mujoco is supported: we ship only an MJCF for the custom robot,
not USD/URDF, so other backends would refuse to load it.

Usage:
    python new_exps/radial_sphere.py --headless
    python new_exps/radial_sphere.py --headless --n-bars 80 --n-sim-steps 6000
"""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import PrimitiveSphereCfg
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def fibonacci_sphere(n: int) -> np.ndarray:
    """Return n approximately-uniform unit vectors on the sphere, shape (n, 3)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=1,
    )


PATH_LENGTH = 6.0
PATH_AMPLITUDE = 0.9
PATH_WAVES = 1.5


def path_xy(s: float) -> np.ndarray:
    """Sinusoid in the xy plane, parameterised by x = s."""
    f = 2 * np.pi * PATH_WAVES / PATH_LENGTH
    return np.array([s, PATH_AMPLITUDE * np.sin(f * s)])


def sample_path(n: int = 240) -> np.ndarray:
    """Discretise the path into n points along x ∈ [0, PATH_LENGTH]."""
    s = np.linspace(0, PATH_LENGTH, n)
    return np.stack([path_xy(si) for si in s], axis=0)


# ---------------------------------------------------------------------------
# MJCF generation
# ---------------------------------------------------------------------------

def build_robot_mjcf(
    n_bars: int = 60,
    sphere_radius: float = 0.15,
    max_extend: float = 0.12,
    bar_length: float = 0.10,
    sleeve_radius: float = 0.010,
    inner_radius: float = 0.008,
) -> tuple[str, np.ndarray]:
    """Build a self-contained MJCF for the radial-sphere robot.

    RoboVerse provides the floor, lighting and skybox via the scenario, so we
    only emit the <worldbody>'s single core body + actuators.  The handler
    adds a freejoint when fix_base_link=False, so we deliberately do NOT
    include one here.
    """
    dirs = fibonacci_sphere(n_bars)
    bars: list[str] = []
    actuators: list[str] = []
    for k, (ux, uy, uz) in enumerate(dirs):
        u = np.array([ux, uy, uz])
        sleeve_from = (sphere_radius * 0.20) * u
        sleeve_to = (sphere_radius * 0.95) * u
        inner_to = sphere_radius * u
        inner_from = (sphere_radius - bar_length) * u
        bars.append(
            f"""
            <geom name="sleeve_{k}" type="capsule"
                  fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                          {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"
                  size="{sleeve_radius}" rgba="0.35 0.40 0.50 1" mass="0.005"
                  contype="0" conaffinity="0"/>
            <body name="inner_{k}" pos="0 0 0">
                <joint name="slide_{k}" type="slide"
                       axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                       range="0 {max_extend}" armature="0.02"/>
                <geom name="inner_geom_{k}" type="capsule"
                      fromto="{inner_from[0]:.5f} {inner_from[1]:.5f} {inner_from[2]:.5f}
                              {inner_to[0]:.5f}   {inner_to[1]:.5f}   {inner_to[2]:.5f}"
                      size="{inner_radius}" rgba="0.1 0.4 1.0 1" mass="0.01"
                      friction="4.0 0.05 0.002" condim="4"/>
            </body>
            """
        )
        # PD gains come from RobotCfg.actuators (BaseActuatorCfg). The MJCF
        # only needs the motor to exist so the handler can find the joint;
        # ctrlrange/forcerange still gate the commanded values.
        # NB: RoboVerse's mujoco handler looks up actuators by JOINT name,
        # so the actuator name must equal the joint name (slide_k), not
        # "motor_k" as in the standalone MuJoCo version.
        actuators.append(
            f'<position name="slide_{k}" joint="slide_{k}" '
            f'ctrlrange="0 {max_extend}" forcerange="-80 80"/>'
        )

    xml = f"""<mujoco model="radial_sphere">
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
    <worldbody>
        <body name="core" pos="0 0 0">
            <geom name="core_geom" type="sphere" size="{sphere_radius}"
                  rgba="1.0 0.85 0.1 1" mass="0.5"/>
            {''.join(bars)}
        </body>
    </worldbody>
    <actuator>
        {''.join(actuators)}
    </actuator>
</mujoco>
"""
    return xml, dirs


# ---------------------------------------------------------------------------
# Locomotion controller
# ---------------------------------------------------------------------------

def desired_direction(
    ball_xy: np.ndarray, path_pts: np.ndarray, lookahead: float = 0.9, goal_eps: float = 0.45
) -> tuple[np.ndarray, float]:
    """Pick a look-ahead point on the path and return a unit xy direction.

    Returns (d_hat, drive).  drive=0 once the ball is within goal_eps of the
    path's last point — the controller then freezes the bars instead of
    pushing past the goal.
    """
    dists = np.linalg.norm(path_pts - ball_xy[None, :], axis=1)
    closest = int(np.argmin(dists))
    end_dist = float(np.linalg.norm(path_pts[-1] - ball_xy))
    if end_dist < goal_eps:
        return np.array([1.0, 0.0]), 0.0

    target_idx = len(path_pts) - 1
    accum = 0.0
    for j in range(closest, len(path_pts) - 1):
        accum += np.linalg.norm(path_pts[j + 1] - path_pts[j])
        if accum >= lookahead:
            target_idx = j + 1
            break
    target = path_pts[target_idx]
    d = target - ball_xy
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.array([1.0, 0.0]), 0.0
    return d / n, 1.0


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """wxyz unit quaternion → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def bar_targets(
    quat: np.ndarray,
    dirs_body: np.ndarray,
    max_extend: float,
    d_hat: np.ndarray,
    drive: float = 1.0,
    back_gain: float = 0.9,
    down_gain: float = 0.4,
    base: float = 0.15,
) -> np.ndarray:
    """Per-bar extension targets in metres."""
    R = quat_to_rotmat(quat)
    dirs_world = dirs_body @ R.T
    align = dirs_world[:, 0] * d_hat[0] + dirs_world[:, 1] * d_hat[1]
    downward = np.clip(-dirs_world[:, 2], 0.0, 1.0)
    score = -back_gain * align + down_gain * downward
    frac = np.clip(base + 0.5 * drive * (1.0 + score), 0.0, 1.0)
    return frac * max_extend


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for the radial-sphere demo."""

        # Only mujoco for now — we don't have USD/URDF for this custom robot.
        sim: Literal["mujoco"] = "mujoco"
        n_bars: int = 60
        max_extend: float = 0.12
        sphere_radius: float = 0.15
        kp: float = 900.0
        kv: float = 22.0
        # Locomotion gains. Lower back_gain → gentler push → slower ball.
        # Original standalone MuJoCo demo used back_gain=0.9 (faster).
        back_gain: float = 0.5
        down_gain: float = 0.4
        # ~ 4 s of physics @ dt=0.002 by default.  Bump for longer runs.
        n_sim_steps: int = 2000
        # Capture one obs every N sim steps → controls effective video frame rate.
        frame_every: int = 17
        # Initial settle steps where bars are held at base extension.
        n_settle_steps: int = 300
        num_envs: int = 1
        headless: bool = True

        def __post_init__(self):
            """Post-initialization configuration."""
            log.info(f"Args: {self}")

    args = tyro.cli(Args)

    # ------------------------------------------------------------------
    # Generate the robot MJCF and write it to disk. Path is outside
    # roboverse_data/ so check_assets() doesn't try to fetch it from HF.
    # ------------------------------------------------------------------
    mjcf_xml, dirs_body = build_robot_mjcf(
        n_bars=args.n_bars, sphere_radius=args.sphere_radius, max_extend=args.max_extend
    )
    mjcf_dir = os.path.abspath("new_exps/output/radial_sphere")
    os.makedirs(mjcf_dir, exist_ok=True)
    mjcf_path = os.path.join(mjcf_dir, "radial_sphere.xml")
    with open(mjcf_path, "w") as f:
        f.write(mjcf_xml)
    log.info(f"Wrote MJCF → {mjcf_path}")

    slide_names = [f"slide_{k}" for k in range(args.n_bars)]
    base_ext = 0.15 * args.max_extend  # resting extension that the ball spawns at

    # RobotCfg for the radial sphere. The PD gains (stiffness/damping) here
    # override whatever MJCF defaults would be — they are what actually drives
    # joint torques every simulate() call.
    robot = RobotCfg(
        name="radial_sphere",
        num_joints=args.n_bars,
        mjcf_path=mjcf_path,
        # No usd/urdf — only mujoco backend is supported for this robot.
        usd_path=None,
        urdf_path=None,
        enabled_gravity=True,
        fix_base_link=False,           # let the ball roll
        enabled_self_collisions=False,
        actuators={n: BaseActuatorCfg(stiffness=args.kp, damping=args.kv) for n in slide_names},
        joint_limits={n: (0.0, args.max_extend) for n in slide_names},
        control_type={n: "position" for n in slide_names},
        default_joint_positions={n: base_ext for n in slide_names},
    )

    scenario = ScenarioCfg(
        robots=[robot],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )
    # Wide chase-style camera covering most of the path.
    scenario.cameras = [
        PinholeCameraCfg(
            name="chase",
            width=1280,
            height=720,
            pos=(3.0, -4.0, 2.5),
            look_at=(3.0, 0.0, 0.0),
        )
    ]

    # ------------------------------------------------------------------
    # Visualise the target path as a row of small red spheres on the floor.
    # We sample fewer points for the markers than for the controller's
    # look-up table — 40 spheres is enough to read at video resolution.
    # ------------------------------------------------------------------
    path_pts_markers = sample_path(40)
    scenario.objects = [
        PrimitiveSphereCfg(
            name=f"marker_{i}",
            radius=0.03,
            color=[1.0, 0.2, 0.2],
            physics=PhysicStateType.RIGIDBODY,
        )
        for i in range(len(path_pts_markers))
    ]

    log.info(f"Using simulator: {args.sim}")
    handler = get_handler(scenario)

    spawn = path_xy(0.0)
    init = {
        "objects": {
            f"marker_{i}": {
                # Sit the markers right on the floor; mass is tiny so they
                # don't perturb the ball if it brushes one.
                "pos": torch.tensor([float(pt[0]), float(pt[1]), 0.03]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            }
            for i, pt in enumerate(path_pts_markers)
        },
        "robots": {
            "radial_sphere": {
                # Spawn at path start; height = sphere + base bar extension so
                # the lower bars just touch the ground (no impact at t=0).
                "pos": torch.tensor(
                    [float(spawn[0]), float(spawn[1]), args.sphere_radius + base_ext + 0.005]
                ),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {n: base_ext for n in slide_names},
            },
        },
    }
    handler.set_states([init] * scenario.num_envs)

    # ------------------------------------------------------------------
    # Settle: hold motors at the base extension while gravity pulls the
    # ball into contact. Velocities damp out over a few hundred steps.
    # ------------------------------------------------------------------
    settle_actions = [
        {"radial_sphere": {"dof_pos_target": {n: base_ext for n in slide_names}}}
        for _ in range(scenario.num_envs)
    ]
    for _ in range(args.n_settle_steps):
        handler.set_dof_targets(settle_actions)
        handler.simulate()

    obs = handler.get_states(mode="tensor")
    os.makedirs("new_exps/output", exist_ok=True)
    obs_saver = ObsSaver(video_path=f"new_exps/output/radial_sphere_{args.sim}.mp4")
    obs_saver.add(obs)

    # Higher-resolution path table for the controller (smoother look-ahead).
    path_pts_ctrl = sample_path(240)

    # ------------------------------------------------------------------
    # Main control loop. One simulate() per iteration; a frame is captured
    # every frame_every steps so the resulting video isn't multi-GB.
    # ------------------------------------------------------------------
    for step in range(args.n_sim_steps):
        # Read env 0's root state. Layout: (pos[3], quat[4], lin_vel[3], ang_vel[3]).
        # Quaternion is wxyz (matches the convention used in init_states above).
        root = obs.robots["radial_sphere"].root_state[0].cpu().numpy()
        ball_xy = root[:2]
        quat = root[3:7]

        d_hat, drive = desired_direction(ball_xy, path_pts_ctrl)
        targets = bar_targets(
            quat,
            dirs_body,
            args.max_extend,
            d_hat,
            drive,
            back_gain=args.back_gain,
            down_gain=args.down_gain,
        )

        actions = [
            {
                "radial_sphere": {
                    "dof_pos_target": {slide_names[k]: float(targets[k]) for k in range(args.n_bars)}
                }
            }
            for _ in range(scenario.num_envs)
        ]
        handler.set_dof_targets(actions)
        handler.simulate()
        obs = handler.get_states(mode="tensor")

        if (step + 1) % args.frame_every == 0:
            obs_saver.add(obs)
        if (step + 1) % 200 == 0:
            log.info(
                f"step {step + 1:>5}/{args.n_sim_steps}  "
                f"ball=({ball_xy[0]:+.2f},{ball_xy[1]:+.2f})  drive={drive:.1f}"
            )

    obs_saver.save()
    handler.close()
