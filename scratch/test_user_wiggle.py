import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from pathlib import Path
import imageio
from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv
from radial_sphere.scenario import generate_scenario
from radial_sphere.run_id import build_run_id
from radial_sphere.snapshot import make_run_dir
import cv2
from skills.overlay import annotate

cfg = load_config("configs/rl/chimney.yaml")
scenario = generate_scenario("chimney", cfg, seed=42)
env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False)
obs, info = env.reset(seed=42)

env.data.qpos[1] = 0.0
env.data.qpos[2] = 0.5
mujoco.mj_forward(env.model, env.data)

print("Testing User's 4-Step Wiggle Climb with Video...")

run_dir = make_run_dir(build_run_id("test_wiggle", "user_method"))
out_video = Path(run_dir) / "renders" / "wiggle_climb.mp4"
out_video.parent.mkdir(parents=True, exist_ok=True)
writer = imageio.get_writer(str(out_video), fps=25, codec="libx264")

# Find rods
left_down = []
right_down = []
left_horiz = []
right_horiz = []

for i, d in enumerate(env.dirs_body):
    if abs(d[0]) > 0.3: continue
    if d[1] > 0.5 and d[2] < -0.3: left_down.append(i)
    elif d[1] < -0.5 and d[2] < -0.3: right_down.append(i)
    elif d[1] > 0.8 and abs(d[2]) <= 0.3: left_horiz.append(i)
    elif d[1] < -0.8 and abs(d[2]) <= 0.3: right_horiz.append(i)

state = "step1_balance"
timer = 0
targets = np.zeros(60)
max_z = 0.5

for step in range(800):
    pos = env.data.qpos[:3].copy()
    max_z = max(max_z, pos[2])
    
    # CHEAT: Zero out X and orientation
    env.data.qpos[0] = 0.0
    env.data.qvel[0] = 0.0
    env.data.qpos[3:7] = [1, 0, 0, 0]
    env.data.qvel[3:6] = [0, 0, 0]
    
    targets[:] = 0.01
    
    if state == "step1_balance":
        for i in left_horiz: targets[i] = 0.15
        for i in right_horiz: targets[i] = 0.15
        timer += 1
        if timer > 50:
            state = "step2_push_right"
            timer = 0
            
    elif state == "step2_push_right":
        for i in right_down: targets[i] = 0.25
        for i in left_down: targets[i] = 0.10
        timer += 1
        if timer > 50:
            state = "step3_balance"
            timer = 0
            
    elif state == "step3_balance":
        for i in left_horiz: targets[i] = 0.15
        for i in right_horiz: targets[i] = 0.15
        timer += 1
        if timer > 50:
            state = "step4_push_left"
            timer = 0
            
    elif state == "step4_push_left":
        for i in left_down: targets[i] = 0.25
        for i in right_down: targets[i] = 0.10
        timer += 1
        if timer > 50:
            state = "step1_balance"
            timer = 0
            
    env.step(targets)
    
    # Video Rendering
    if step % 4 == 0:
        if env.renderer is None:
            env.render(camera_name="fixed_angle_close_3d")

        cam1 = mujoco.MjvCamera()
        cam1.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam1.trackbodyid = env.core_body_id
        cam1.distance = 1.35
        cam1.elevation = -5.0
        cam1.azimuth = 180.0
        env.renderer.update_scene(env.data, camera=cam1)
        f_macro = env.renderer.render()

        f_macro_ann = annotate(
            f_macro,
            "User's Wiggle Climb Method",
            [
                f"State: {state}",
                f"Z Height: {pos[2]:.3f} m",
                f"Y Offset: {pos[1]:.3f} m"
            ],
            margin=14,
        )

        writer.append_data(f_macro_ann)

writer.close()
env.close()

# Save path for the agent to find it
with open("wiggle_video_path.txt", "w") as f:
    f.write(str(out_video))
print(f"Video saved to {out_video}")
