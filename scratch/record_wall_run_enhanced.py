import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from pathlib import Path
from omegaconf import OmegaConf
import imageio

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.snapshot import make_run_dir
from radial_sphere.run_id import build_run_id
from skills.overlay import annotate
from skills.wall_run import FOOT_BASE, next_phase, _frame
from scratch.test_wall_run_upward_bias import wall_run_enhanced

def record_comparison(approach_angle=18.0, along_drive=0.40, upward_bias=0.20, slowmo=4, seconds=8.0):
    cfg = load_config("configs/rl/wall_run.yaml")
    OmegaConf.set_struct(cfg, False)
    scenario = generate_scenario("wall_run", cfg, seed=3)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10 ** 7)
    env.reset(seed=3)
    mujoco.mj_forward(env.model, env.data)

    # Set architectural concrete wall material for high visual clarity
    mat_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_MATERIAL, "wall_mat")
    if mat_id >= 0:
        env.model.mat_rgba[mat_id] = [0.68, 0.64, 0.58, 1.0]

    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "wall_0")
    pos_w = env.model.geom_pos[gid]
    size_w = env.model.geom_size[gid]
    n_hat = np.array([0.0, 1.0, 0.0])
    face_y = float(pos_w[1] - size_w[1])
    lane_gap = float(face_y - scenario.spawn_xy[1])
    travel = np.array([1.0, 0.0, 0.0])

    run_dir = make_run_dir(build_run_id("wall_run", "enhanced_slowmo"))
    out_video = Path(run_dir) / "renders" / "wall_run_enhanced_slowmo.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    fps = max(int(100 / slowmo), 1)
    writer = imageio.get_writer(str(out_video), fps=fps, codec="libx264")

    phase = "approach"
    for i in range(int(seconds * 100)):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        wall_dist = face_y - float(pos[1])
        on_ground = any(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom1) == "floor" or
                        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom2) == "floor"
                        for c_i in range(env.data.ncon))

        nxt = next_phase(
            phase, wall_dist=wall_dist, lin_vel=vel, height=float(pos[2]),
            on_ground=on_ground, max_extend=env.max_extend, speed_ready=4.0,
            launch_gap=1.05, lane_gap=lane_gap, wall_normal=n_hat,
        )
        phase = nxt

        targets = wall_run_enhanced(
            env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
            phase=phase, wall_normal=n_hat, wall_dist=wall_dist,
            travel=travel, lin_vel=vel, speed=4.0,
            approach_angle=approach_angle, recover_speed=1.5,
            launch_in=2.1, launch_up=2.8,
            along_drive=along_drive, upward_bias=upward_bias,
        )
        env.step(targets)

        if env.renderer is None:
            env.render(camera_name="fixed_angle_close_3d")

        cam = mujoco.MjvCamera()
        eye = np.array([pos[0] - 2.4, face_y - 6.4, pos[2] + 3.4])
        look = np.array([pos[0] + 0.9, 0.5 * (pos[1] + face_y), pos[2] + 0.35])
        from scripts.skills.run_wall_run import _aim
        _aim(cam, eye, look)
        env.renderer.update_scene(env.data, camera=cam)
        frame = annotate(
            env.renderer.render(),
            f"Horizontal Wall Run (Enhanced {slowmo}x Slowmo)",
            [
                f"phase    {phase}",
                f"gap      {wall_dist:.2f} m to wall",
                f"height   {pos[2]:.2f} m",
                f"speed    {np.linalg.norm(vel):.2f} m/s  (into wall {np.dot(vel, n_hat):+.2f})",
            ],
            margin=14,
        )
        writer.append_data(frame)

    writer.close()
    env.close()
    print(f"Video saved to: {out_video}")

    # Copy to artifact path
    dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/wall_run_enhanced.mp4")
    import shutil
    shutil.copyfile(str(out_video), str(dst))
    print(f"Copied to artifact path: {dst}")

if __name__ == "__main__":
    record_comparison()
