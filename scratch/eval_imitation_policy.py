"""Evaluate and Visualize Pure Imitation Learning (BC) Policy on Test Episodes."""
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
import rootutils
import torch

rootutils.setup_root("/home/azureuser/telescopic_robot", pythonpath=True)

from radial_sphere import (
    MujocoSteeringEnv,
    desired_direction,
    generate_scenario,
    load_config_cli,
)
from scripts.imitation.train_bc import HierarchicalImitationPolicy

ckpt_path = Path("/home/azureuser/telescopic_robot/storage_local/imitation_models/bc_hierarchical_best.pt")
assert ckpt_path.exists(), f"BC Checkpoint not found at {ckpt_path}"

ckpt = torch.load(str(ckpt_path), map_location="cpu")
obs_dim = ckpt.get("obs_dim", 163)

model = HierarchicalImitationPolicy(obs_dim=obs_dim, high_act_dim=3, low_act_dim=60, hidden_dim=256)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

scratch_dir = Path("/home/azureuser/.gemini/antigravity-ide/brain/6c6c10ba-f20d-4055-aff1-c75d2495e95b/scratch")
renders_dir = Path("/home/azureuser/telescopic_robot/storage_local/imitation_models/renders")
renders_dir.mkdir(parents=True, exist_ok=True)

test_scenarios = [
    {"name": "Test Ep 1: Level 1 Orthogonal Maze", "level": 1, "seed": 9901},
    {"name": "Test Ep 2: Level 2 Multi-Loop Braid Maze", "level": 2, "seed": 9902},
    {"name": "Test Ep 3: Level 3 Twisty Tree Maze", "level": 3, "seed": 9903},
]

results = []

for idx, sc_info in enumerate(test_scenarios):
    level = sc_info["level"]
    seed = sc_info["seed"]
    
    cfg_name = f"maze_level{level}_random_endpoints" if level in (2, 3) else "maze_level3_random_endpoints"
    try:
        cfg = load_config_cli(name=cfg_name)
    except Exception:
        cfg = load_config_cli(name="maze_level3_random_endpoints")
        
    OmegaConf.set_struct(cfg, False)
    if hasattr(cfg.scenario, "maze") and cfg.scenario.maze is not None:
        cfg.scenario.maze.level = level
        cfg.scenario.maze.random_endpoints = True
        cfg.scenario.maze.endpoint_min_route = 6.0

    sc = generate_scenario("maze", cfg, seed=seed)
    env = MujocoSteeringEnv(cfg, scenario=sc, randomize=False, max_steps=2000)
    obs, _ = env.reset(seed=seed)
    raw = env.env

    frames_dual = [raw.render(mode="dual")]
    frames_chase = [raw.render(mode="chase")]

    done = False
    step = 0
    total_reward = 0.0
    wall_contacts = 0
    
    print(f"\nEvaluating Pure IL Policy on {sc_info['name']} (Seed {seed})...", flush=True)

    while not done and step < 1500:
        ball_xy = raw.data.qpos[:2]
        quat = raw.data.qpos[3:7]
        g, expert_drive = desired_direction(ball_xy, sc.path_pts, lookahead=float(cfg.controller.lookahead))
        
        # Construct 163D observation
        lidar = raw.raycast_lidar(n_rays=24, max_range=3.0, g=g)
        v_fwd = float(raw.data.qvel[0] * g[0] + raw.data.qvel[1] * g[1])
        v_lat = float(g[0] * raw.data.qvel[1] - g[1] * raw.data.qvel[0])
        norm_dist = float(np.linalg.norm(sc.goal[:2] - ball_xy) / max(raw.path_length, 1.0))
        rel_goal = (sc.goal[:2] - ball_xy).astype(np.float32)
        norm_joint = (raw.data.qpos[7:7 + raw.n_bars] / raw.max_extend).astype(np.float32)

        obs_163 = np.concatenate([
            quat.astype(np.float32),
            np.array([v_fwd, v_lat, raw.data.qvel[2]], dtype=np.float32),
            raw.data.qvel[3:6].astype(np.float32),
            norm_joint,
            np.zeros(66, dtype=np.float32),
            rel_goal,
            np.array([norm_dist], dtype=np.float32),
            lidar.astype(np.float32),
        ])

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_163).unsqueeze(0).float()
            pred_high, pred_low = model(obs_tensor)
            act_high = pred_high.squeeze(0).numpy()
            act_low = pred_low.squeeze(0).numpy()

        # Step environment using the pure IL predicted action
        obs, r, terminated, truncated, info = env.step(act_high)
        done = terminated or truncated
        total_reward += float(r)

        if info.get("wall_contact", False):
            wall_contacts += 1

        if step % 2 == 0:
            frames_dual.append(raw.render(mode="dual"))
            frames_chase.append(raw.render(mode="chase"))

        step += 1

    final_dist = float(np.linalg.norm(raw.data.qpos[:2] - sc.goal[:2]))
    success = bool(final_dist < 0.50 or info.get("success", False))

    vid_dual = renders_dir / f"il_test_ep_{idx+1}_dual.mp4"
    vid_chase = renders_dir / f"il_test_ep_{idx+1}_chase.mp4"
    imageio.mimsave(str(vid_dual), frames_dual, fps=25)
    imageio.mimsave(str(vid_chase), frames_chase, fps=25)

    thumb = scratch_dir / f"il_test_ep_{idx+1}_mid.png"
    imageio.imwrite(str(thumb), frames_dual[len(frames_dual)//2])

    results.append({
        "name": sc_info["name"],
        "level": level,
        "seed": seed,
        "steps": step,
        "success": success,
        "goal_dist": final_dist,
        "wall_contacts": wall_contacts,
        "wall_pct": wall_contacts / max(step, 1) * 100.0,
        "total_reward": total_reward,
        "dual_video": str(vid_dual),
        "chase_video": str(vid_chase),
        "thumb": str(thumb),
    })

    print(f"[{sc_info['name']}] Steps={step} | Success={success} | GoalDist={final_dist:.2f}m | WallHits={wall_contacts} ({wall_contacts/max(step,1)*100:.1f}%) | Rew={total_reward:.2f}")
    env.close()

print("\n=========================================================================")
print("PURE IMITATION LEARNING TEST EVALUATION RESULTS")
print("=========================================================================")
for r in results:
    print(f"{r['name']:35s} | Success={str(r['success']):5s} | GoalDist={r['goal_dist']:4.2f}m | Steps={r['steps']:4d} | WallHits={r['wall_contacts']:3d} ({r['wall_pct']:4.1f}%) | Rew={r['total_reward']:7.2f}")
