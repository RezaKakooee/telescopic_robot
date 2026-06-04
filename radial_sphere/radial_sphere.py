"""Main runner: wires geometry, MJCF, controller + RoboVerse scenario together.

    from radial_sphere import run, Args
    run(Args())
"""
from __future__ import annotations

import os

import numpy as np
import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import PrimitiveSphereCfg
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler

from .config import Args
from .controller import bar_targets, desired_direction
from .geometry import path_xy, sample_path
from .mjcf import build_robot_mjcf


def run(args: Args) -> None:
    """Run the radial-sphere locomotion demo end-to-end."""
    # ------------------------------------------------------------------
    # Generate the robot MJCF and write it to disk. Path is outside
    # roboverse_data/ so check_assets() doesn't try to fetch it from HF.
    # ------------------------------------------------------------------
    mjcf_xml, dirs_body = build_robot_mjcf(
        n_bars=args.n_bars,
        sphere_radius=args.sphere_radius,
        max_extend=args.max_extend,
    )
    mjcf_dir = os.path.abspath("new_exps/output/radial_sphere")
    os.makedirs(mjcf_dir, exist_ok=True)
    mjcf_path = os.path.join(mjcf_dir, "radial_sphere.xml")
    with open(mjcf_path, "w") as f:
        f.write(mjcf_xml)
    log.info(f"Wrote MJCF → {mjcf_path}")

    slide_names = [f"slide_{k}" for k in range(args.n_bars)]
    base_ext = 0.15 * args.max_extend  # resting extension at spawn

    # RobotCfg — PD gains (stiffness/damping) override MJCF defaults.
    robot = RobotCfg(
        name="radial_sphere",
        num_joints=args.n_bars,
        mjcf_path=mjcf_path,
        usd_path=None,
        urdf_path=None,
        enabled_gravity=True,
        fix_base_link=False,
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
    scenario.cameras = [
        PinholeCameraCfg(
            name="chase",
            width=1280,
            height=720,
            pos=(3.0, -4.0, 2.5),
            look_at=(3.0, 0.0, 0.0),
        )
    ]

    # Visualise the target path as a row of small red spheres on the floor.
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

    # Settle: hold motors at base extension while gravity pulls the ball into
    # contact. Velocities damp out over a few hundred steps.
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

    # Higher-resolution path table for smoother look-ahead.
    path_pts_ctrl = sample_path(240)

    for step in range(args.n_sim_steps):
        # root_state layout: (pos[3], quat[4], lin_vel[3], ang_vel[3])
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
