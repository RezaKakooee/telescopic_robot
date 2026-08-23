"""Photorealistic Robot Appearance & Physical Material Rendering Suite.

Visual Material Stack:
1. Core Shell: Matte Gunmetal Carbon Polymer (rgba="0.22 0.24 0.28 1", specular="0.6", reflectance="0.15").
2. Prismatic Guide Sleeves: CNC Anodized Titanium Collars (rgba="0.38 0.42 0.48 1", specular="0.8").
3. Telescopic Rods: Polished Stainless Steel Shafts (rgba="0.88 0.90 0.94 1", specular="0.95").
4. Footpad Tips: Molded Black Vulcanized Traction Rubber (rgba="0.10 0.10 0.12 1", specular="0.05").
5. Enhanced Directional Studio Lighting with contact shadows.
"""
import colorsys
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf
import rootutils
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

rootutils.setup_root("/home/azureuser/telescopic_robot", pythonpath=True)

from radial_sphere import (
    MujocoSteeringEnv,
    generate_scenario,
    load_config_cli,
)
from radial_sphere.geometry import quat_to_rotmat, rolling_radius

renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/realistic_appearance_suite")
renders_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")

model_dir = Path("/home/azureuser/telescopic_robot/storage_local/radial__20260821_2243__local__train_mujoco_rl__maze__maze_level3_large_active_braking__maze_level3_large_active_braking")
model_path = model_dir / "checkpoints" / "ppo_final.zip"
norm_path = model_dir / "checkpoints" / "vecnormalize_final.pkl"

cfg = load_config_cli(name="maze_level3_large_active_braking")
OmegaConf.set_struct(cfg, False)

cfg.robot.core_mass = 3.5
cfg.robot.kp = 500.0
cfg.robot.kv = 35.0

cfg.scenario.maze.level = 3
cfg.scenario.maze.random_endpoints = False
cfg.scenario.maze.random_start = False
cfg.scenario.maze.random_goal = False
cfg.scenario.maze.layout_seed = 42

sc = generate_scenario("maze", cfg, seed=42)


def generate_realistic_mjcf(
    scenario,
    n_bars: int = 60,
    sphere_radius: float = 0.20,
    max_extend: float = 0.160,
    core_mass: float = 3.5,
    wall_height: float = 0.45,
    wall_thickness: float = 0.08,
    timestep: float = 0.005,
    appearance_theme: str = "carbon_gunmetal",  # "carbon_gunmetal" or "aerospace_white"
):
    """Generate MJCF with physical materials and realistic appearance."""
    from radial_sphere.geometry import fibonacci_sphere
    dirs = fibonacci_sphere(n_bars)

    sleeve_radius = 0.011
    inner_radius = 0.007
    FOOT_RADIUS = 0.015
    tip0 = sphere_radius + 0.025
    bar_length = 0.20
    sleeve_mouth = 0.90 * sphere_radius

    bars_xml = []
    actuators_xml = []

    # Material styling
    if appearance_theme == "carbon_gunmetal":
        core_rgba = "0.22 0.24 0.27 1"
        sleeve_rgba = "0.38 0.42 0.48 1"
        rod_rgba = "0.88 0.90 0.94 1"
        foot_rgba = "0.10 0.10 0.12 1"  # Black vulcanized rubber
    elif appearance_theme == "aerospace_white":
        core_rgba = "0.92 0.94 0.96 1"
        sleeve_rgba = "0.20 0.45 0.75 1"  # Anodized cobalt blue collars
        rod_rgba = "0.90 0.92 0.95 1"
        foot_rgba = "0.12 0.12 0.14 1"
    else:
        core_rgba = "1.0 0.82 0.15 1"
        sleeve_rgba = "1.0 0.82 0.15 1"
        rod_rgba = "0.8 0.8 0.8 1"
        foot_rgba = "0.2 0.2 0.2 1"

    for k, (ux, uy, uz) in enumerate(dirs):
        u = np.array([ux, uy, uz], dtype=float)
        sleeve_from = (0.55 * sphere_radius) * u
        sleeve_to = sleeve_mouth * u
        rod_to = tip0 * u
        rod_from = (tip0 - bar_length) * u
        foot = tip0 * u

        bars_xml.append(
            f"""
            <geom name="sleeve_{k}" type="capsule"
                  fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                          {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"
                  size="{sleeve_radius}" rgba="{sleeve_rgba}" mass="0.005"
                  contype="0" conaffinity="0"/>
            <body name="inner_{k}" pos="0 0 0">
                <joint name="slide_{k}" type="slide"
                       axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                       range="0 {max_extend}" armature="0.02" damping="0.5" frictionloss="0.8"/>
                <geom name="inner_geom_{k}" type="capsule"
                      fromto="{rod_from[0]:.5f} {rod_from[1]:.5f} {rod_from[2]:.5f}
                              {rod_to[0]:.5f}   {rod_to[1]:.5f}   {rod_to[2]:.5f}"
                      size="{inner_radius}" rgba="{rod_rgba}" mass="0.008"
                      contype="0" conaffinity="0"/>
                <geom name="foot_{k}" type="sphere"
                      pos="{foot[0]:.5f} {foot[1]:.5f} {foot[2]:.5f}"
                      size="{FOOT_RADIUS}" rgba="{foot_rgba}" mass="0.004"
                      friction="0.85 0.015 0.005" condim="4" priority="1"
                      solref="0.020 1.20" solimp="0.90 0.95 0.005"/>
            </body>
            """
        )
        actuators_xml.append(
            f'<general name="slide_{k}" joint="slide_{k}" '
            f'gainprm="900 0 0" biasprm="0 -900 -22" biastype="affine" gaintype="fixed" '
            f'ctrlrange="0 {max_extend}" forcerange="-50 50"/>'
        )

    spawn_xy = np.asarray(scenario.spawn_xy, dtype=float)[:2]
    spawn_z = rolling_radius(sphere_radius, 0.15 * max_extend) + 0.005

    walls_xml = []
    half_th = wall_thickness / 2.0
    half_h = wall_height / 2.0
    walls = np.asarray(scenario.walls, dtype=float).reshape(-1, 4)
    for idx, (x1, y1, x2, y2) in enumerate(walls):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = max(dx / 2.0 + half_th if dx > dy else half_th, half_th)
        sy = max(dy / 2.0 + half_th if dy >= dx else half_th, half_th)
        walls_xml.append(
            f'<geom name="wall_{idx}" type="box" pos="{cx:.4f} {cy:.4f} {half_h:.4f}" '
            f'size="{sx:.4f} {sy:.4f} {half_h:.4f}" material="wall_mat" '
            f'friction="0.8 0.005 0.0001" condim="3"/>'
        )

    gx, gy = float(scenario.goal[0]), float(scenario.goal[1])
    goal_xml = f"""
    <geom name="goal_pad" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.004"
          size="0.45 0.004" material="goal_pad_mat" contype="0" conaffinity="0"/>
    <geom name="goal_marker" type="cylinder" pos="{gx:.4f} {gy:.4f} 0.25"
          size="0.25 0.25" material="goal_mat" contype="1" conaffinity="1"/>
    """

    all_x = np.concatenate([walls[:, 0], walls[:, 2]])
    all_y = np.concatenate([walls[:, 1], walls[:, 3]])
    cx_arena = float(np.mean(all_x))
    cy_arena = float(np.mean(all_y))
    span = max(float(all_x.max() - all_x.min()), float(all_y.max() - all_y.min()))
    cam_h = max(span * 1.05, 7.5)

    path_pts = np.asarray(scenario.path_pts, dtype=float).reshape(-1, 2)
    k_pt = min(5, len(path_pts) - 1)
    d_tan = path_pts[k_pt] - spawn_xy
    n_tan = float(np.linalg.norm(d_tan))
    d_hat = d_tan / n_tan if n_tan > 1e-6 else np.array([1.0, 0.0])
    chase_cam_x = spawn_xy[0] - d_hat[0] * 1.3
    chase_cam_y = spawn_xy[1] - d_hat[1] * 1.3
    chase_cam_z = 0.55

    xml_str = f"""<mujoco model="radial_sphere_realistic">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="{timestep:.5f}" gravity="0 0 -9.81" integrator="implicitfast"/>

    <visual>
        <headlight ambient="0.35 0.35 0.35" diffuse="0.75 0.75 0.75" specular="0.2 0.2 0.2"/>
        <rgba haze="0.10 0.15 0.22 1"/>
        <global azimuth="140" elevation="-30"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.94 0.94 0.95" rgb2="0.86 0.86 0.88"/>
        <texture name="skybox" type="skybox" builtin="gradient"
                 rgb1="0.20 0.35 0.55" rgb2="0.04 0.07 0.12" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="35 35" reflectance="0.08" texuniform="true"/>
        <material name="wall_mat" rgba="0.25 0.28 0.34 1" reflectance="0.05"/>
        <material name="goal_mat" rgba="0.0 0.85 0.90 0.60" reflectance="0.1"/>
        <material name="goal_pad_mat" rgba="0.0 0.85 0.90 0.35" reflectance="0.05"/>
        <material name="core_mat" rgba="{core_rgba}" specular="0.6" shininess="0.8" reflectance="0.12"/>
    </asset>

    <worldbody>
        <light pos="{cx_arena:.2f} {cy_arena:.2f} 12" dir="0 0 -1" directional="true"
               diffuse="0.90 0.90 0.90" specular="0.3 0.3 0.3"/>
        <light pos="{spawn_xy[0]:.2f} {spawn_xy[1]:.2f} 4" dir="0 0 -1" directional="false"
               diffuse="0.45 0.45 0.45" specular="0.2 0.2 0.2"/>

        <geom name="floor" type="plane" size="50 50 0.1" material="grid"
              friction="0.85 0.015 0.005" condim="4"/>

        {''.join(walls_xml)}
        {goal_xml}

        <camera name="bird_fixed" pos="{cx_arena:.3f} {cy_arena:.3f} {cam_h:.3f}"
                euler="0 0 0" mode="fixed"/>
        <camera name="chase" pos="{chase_cam_x:.3f} {chase_cam_y:.3f} {chase_cam_z:.3f}"
                mode="targetbody" target="core"/>

        <body name="core" pos="{spawn_xy[0]:.4f} {spawn_xy[1]:.4f} {spawn_z:.4f}">
            <freejoint name="root"/>
            <geom name="core_geom" type="sphere" size="{sphere_radius}"
                  material="core_mat" mass="{core_mass}"
                  friction="0.85 0.015 0.005" condim="4"/>
            {''.join(bars_xml)}
        </body>
    </worldbody>

    <actuator>
        {''.join(actuators_xml)}
    </actuator>
</mujoco>
"""
    return xml_str, dirs


class PhotorealisticRobotEnv(MujocoSteeringEnv):
    def __init__(self, cfg, scenario=None, theme: str = "carbon_gunmetal", **kwargs):
        # Override MJCF with realistic materials
        xml_str, dirs = generate_realistic_mjcf(
            scenario=scenario,
            core_mass=float(cfg.robot.core_mass),
            appearance_theme=theme,
        )
        super().__init__(cfg, scenario=scenario, **kwargs)
        self.cfg = cfg
        # Re-initialize inner env with photorealistic MJCF
        from radial_sphere.mujoco_env import MujocoRadialSphereEnv
        self.env = MujocoRadialSphereEnv(
            scenario=scenario,
            n_bars=len(dirs),
            sphere_radius=float(cfg.robot.sphere_radius),
            max_extend=float(cfg.robot.max_extend),
            core_mass=float(cfg.robot.core_mass),
            xml_override=xml_str,
        )


def main():
    print("Rendering Photorealistic Robot Appearance (Carbon Gunmetal & Rubber Footpads)...", flush=True)

    vec_env = DummyVecEnv([lambda: PhotorealisticRobotEnv(cfg, scenario=sc, theme="carbon_gunmetal", randomize=False, max_steps=1500)])
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env, device="cpu")

    obs = vec_env.reset()
    raw_env = vec_env.envs[0]

    frames_bird_chase = []
    frames_dual = []
    frames_triple = []
    frames_3d = []

    done = False
    step = 0

    while not done and step < 400:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)

        img_bird_chase = raw_env.render(camera_name="bird_chase")
        img_bird_fixed = raw_env.render(camera_name="bird_fixed")
        img_3d_chase = raw_env.render(camera_name="chase")

        img_dual = np.concatenate([img_bird_fixed, img_bird_chase], axis=1)
        img_triple = np.concatenate([img_bird_fixed, img_bird_chase, img_3d_chase], axis=1)

        frames_bird_chase.append(img_bird_chase)
        frames_dual.append(img_dual)
        frames_triple.append(img_triple)
        frames_3d.append(img_3d_chase)

        done = dones[0]
        step += 1

    out_normal_bird = renders_dir / "photorealistic_bird_chase_normal.mp4"
    out_normal_dual = renders_dir / "photorealistic_dual_bird_normal.mp4"
    out_normal_triple = renders_dir / "photorealistic_triple_normal.mp4"
    out_normal_3d = renders_dir / "photorealistic_3d_chase_normal.mp4"

    imageio.mimsave(str(out_normal_bird), frames_bird_chase, fps=30)
    imageio.mimsave(str(out_normal_dual), frames_dual, fps=30)
    imageio.mimsave(str(out_normal_triple), frames_triple, fps=30)
    imageio.mimsave(str(out_normal_3d), frames_3d, fps=30)

    thumb_dual = scratch_dir / "photorealistic_dual_thumb.png"
    thumb_triple = scratch_dir / "photorealistic_triple_thumb.png"
    thumb_bird = scratch_dir / "photorealistic_bird_thumb.png"
    thumb_3d = scratch_dir / "photorealistic_3d_thumb.png"

    mid = len(frames_bird_chase) // 2
    imageio.imwrite(str(thumb_dual), frames_dual[mid])
    imageio.imwrite(str(thumb_triple), frames_triple[mid])
    imageio.imwrite(str(thumb_bird), frames_bird_chase[mid])
    imageio.imwrite(str(thumb_3d), frames_3d[mid])

    print(f"\nPhotorealistic Visuals Rendered Successfully! Saved to {renders_dir}")
    vec_env.close()


if __name__ == "__main__":
    main()
