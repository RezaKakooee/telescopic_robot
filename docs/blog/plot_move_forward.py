"""Measure and illustrate move_forward on the current default robot.

Run from the repository root with PYTHONPATH=. and MUJOCO_GL=egl.
Outputs use centimetres; simulation and skill inputs use SI units.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from skills import execute_skill


def main():
    assets = Path(__file__).resolve().parent / "assets"
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.scenario.goal.x_range = [0.0, 0.0]
    cfg.scenario.goal.y_range = [-400.0, -400.0]
    env = MujocoRadialSphereEnv(
        cfg, scenario=generate_scenario("goal", cfg, seed=42),
        randomize=False, max_steps=10000,
    )
    env.reset(seed=42)
    origin = env.data.qpos[:3].copy()
    start_time = float(env.data.time)
    env.model.vis.global_.offwidth = 800
    env.model.vis.global_.offheight = 480
    renderer = mujoco.Renderer(env.model, height=480, width=800)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90, -20, 1.8
    rows = [[0.0, 0.0, 0.0, float(env.data.qvel[0]) * 100]]
    requested_speed = 1.2
    steps = round(6.0 / (env.model.opt.timestep * env.action_repeat))
    video_stride = round(0.04 / (env.model.opt.timestep * env.action_repeat))
    try:
        with imageio.get_writer(assets / "move-forward.mp4", fps=25) as writer:
            for step in range(steps):
                targets = execute_skill(
                    "move", env.data.qpos[3:7].copy(),
                    env.dirs_body, env.max_extend,
                    d_hat=np.array([1.0, 0.0]), speed=requested_speed,
                    lin_vel=env.data.qvel[:2].copy(),
                    cross_track_error=float(env.data.qpos[1] - origin[1]),
                    rod_mechanism=str(cfg.robot.rod_mechanism),
                )
                _, _, terminated, truncated, _ = env.step(targets)
                if terminated or truncated:
                    raise RuntimeError("Run ended before the six-second measurement")
                delta = (env.data.qpos[:3] - origin) * 100
                rows.append([env.data.time - start_time, delta[0], delta[1],
                             env.data.qvel[0] * 100])
                if step % video_stride == 0:
                    camera.lookat[:] = env.data.qpos[:3]
                    renderer.update_scene(env.data, camera=camera)
                    frame = renderer.render()
                    writer.append_data(frame)
                    if step == steps // 2:
                        imageio.imwrite(assets / "move-forward-frame.png", frame)
        data = np.asarray(rows)
        np.savetxt(assets / "move-forward.csv", data, delimiter=",",
                   header="time_s,forward_cm,sideways_cm,forward_speed_cm_s", comments="")
        results = dict(
            config="configs/rl/config.yaml", seed=42, mechanism=str(cfg.robot.rod_mechanism),
            requested_speed_cm_s=120, duration_s=float(data[-1, 0]),
            distance_cm=float(data[-1, 1]), sideways_cm=float(data[-1, 2]),
            mean_speed_last_2s_cm_s=float(data[data[:, 0] >= 4, 3].mean()),
            physics_step_s=float(env.model.opt.timestep), action_repeat=env.action_repeat,
            mujoco_version=mujoco.__version__,
        )
        (assets / "move-forward-results.json").write_text(json.dumps(results, indent=2) + "\n")
        plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "legend.fontsize": 13})
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
        axes[0].plot(data[:, 0], data[:, 3], color="#1764d8", label="Measured forward speed")
        axes[0].axhline(120, color="#cf3c20", linestyle="--", label="Requested: 120 cm/s")
        axes[0].set(ylabel="Forward speed (cm/s)", title="Does it reach the requested speed?")
        axes[1].plot(data[:, 0], data[:, 1], color="#087f63", label="Forward displacement")
        axes[1].plot(data[:, 0], data[:, 2], color="#b51a7c", label="Sideways displacement")
        axes[1].set(ylabel="Displacement from start (cm)", title="Where does the ball actually go?")
        for ax in axes:
            ax.set(xlabel="Time after command (s)", xlim=(0, 6))
            ax.grid(alpha=0.2)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), frameon=False)
        fig.savefig(assets / "move-forward-performance.png", dpi=180)
        plt.close(fig)
        print(json.dumps(results, indent=2))
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
