import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from pathlib import Path
from omegaconf import OmegaConf
import imageio

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import Scenario, generate_scenario
from radial_sphere.snapshot import make_run_dir
from radial_sphere.run_id import build_run_id
from skills.overlay import annotate
from skills.wall_run import FOOT_BASE, next_phase, _frame
from scratch.test_wall_run_upward_bias import wall_run_enhanced

def create_wall_run_ledge_scenario(cfg):
    wall_y = 1.30
    length = 24.0
    lane = 2.60

    # Main wall
    walls = np.array([[0.0, wall_y, length, wall_y]], dtype=np.float32)
    spawn = np.array([1.0, wall_y - lane], dtype=np.float32)
    goal = np.array([12.0, wall_y - lane + 1.0], dtype=np.float32)

    # Elevated ledge platform: x in [5.6, 9.6], y in [-0.50, 0.50], height = 0.35m
    # Box geom pos: cx=7.6, cy=0.0, z=0.175; half-sizes: sx=2.0, sy=0.50, sz=0.175
    steps = [[7.60, 0.0, 2.00, 0.50, 0.35]]

    pts = np.linspace(spawn, goal, 40).astype(np.float32)

    return Scenario(
        kind="wall_run",
        name="wall_run_to_ledge",
        spawn_xy=spawn,
        goal=goal,
        path_pts=pts,
        markers=np.empty((0, 2), dtype=np.float32),
        path_length=float(np.linalg.norm(goal - spawn)),
        walls=walls,
        steps=steps,
    )

def run_wall_run_to_ledge(record_video=True, slowmo=2, seconds=7.0):
    cfg = load_config("configs/rl/wall_run.yaml")
    OmegaConf.set_struct(cfg, False)
    scenario = create_wall_run_ledge_scenario(cfg)
    env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False, max_steps=10 ** 7)
    env.reset(seed=3)
    mujoco.mj_forward(env.model, env.data)

    # Set architectural concrete wall material
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

    run_dir = make_run_dir(build_run_id("wall_run", "to_ledge"))
    out_video = Path(run_dir) / "renders" / "wall_run_to_ledge.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    fps = max(int(100 / slowmo), 1)
    writer = imageio.get_writer(str(out_video), fps=fps, codec="libx264") if record_video else None

    phase = "approach"
    landed_on_ledge = False
    max_z = 0.22

    print("=== Parkour Wall Run -> Ledge Traversal ===")

    for i in range(int(seconds * 100)):
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        wall_dist = face_y - float(pos[1])
        max_z = max(max_z, float(pos[2]))

        on_ground = any(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom1) in ("floor", "step_0") or
                        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom2) in ("floor", "step_0")
                        for c_i in range(env.data.ncon))

        nxt = next_phase(
            phase, wall_dist=wall_dist, lin_vel=vel, height=float(pos[2]),
            on_ground=on_ground, max_extend=env.max_extend, speed_ready=4.0,
            launch_gap=1.05, lane_gap=lane_gap, wall_normal=n_hat,
        )
        if nxt != phase:
            print(f"  t={i/100:5.2f}s  {phase} -> {nxt}   x={pos[0]:.2f} z={pos[2]:.2f} gap={wall_dist:.2f} v={np.linalg.norm(vel):.2f}")
            phase = nxt

        targets = wall_run_enhanced(
            env.data.qpos[3:7].copy(), env.dirs_body, env.max_extend,
            phase=phase, wall_normal=n_hat, wall_dist=wall_dist,
            travel=travel, lin_vel=vel, speed=4.0,
            approach_angle=18.0, recover_speed=1.5,
            launch_in=2.1, launch_up=2.8,
            along_drive=0.40, upward_bias=0.20,
        )
        env.step(targets)

        # Check if robot is on top of the ledge (step_0)
        on_ledge = any(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom1) == "step_0" or
                       mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[c_i].geom2) == "step_0"
                       for c_i in range(env.data.ncon))
        if on_ledge and pos[2] > 0.55:
            landed_on_ledge = True

        if writer is not None and i % 2 == 0:
            if env.renderer is None:
                env.render(camera_name="fixed_angle_close_3d")
            cam = mujoco.MjvCamera()
            eye = np.array([pos[0] - 2.8, face_y - 6.8, pos[2] + 3.8])
            look = np.array([pos[0] + 1.2, 0.5 * (pos[1] + face_y), pos[2] + 0.35])
            from scripts.skills.run_wall_run import _aim
            _aim(cam, eye, look)
            env.renderer.update_scene(env.data, camera=cam)
            frame = annotate(
                env.renderer.render(),
                "Parkour: Horizontal Wall Run -> Ledge Traversal",
                [
                    f"phase    {phase}",
                    f"gap      {wall_dist:.2f} m to wall",
                    f"height   {pos[2]:.2f} m (Ledge = 0.45m)",
                    f"speed    {np.linalg.norm(vel):.2f} m/s",
                    f"status   {'ON LEDGE' if landed_on_ledge else 'AIRBORNE/APPROACH'}",
                ],
                margin=14,
            )
            writer.append_data(frame)

    if writer is not None:
        writer.close()
    env.close()

    print(f"\nResult: Peak Z = {max_z:.2f}m | Landed on Ledge = {landed_on_ledge} | Final Z = {pos[2]:.2f}m")
    if record_video:
        dst = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/wall_run_to_ledge.mp4")
        import shutil
        shutil.copyfile(str(out_video), str(dst))
        print(f"Copied to artifact path: {dst}")

if __name__ == "__main__":
    run_wall_run_to_ledge()
