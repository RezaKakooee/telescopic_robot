"""RL fine-tuning (PPO) of the Skills VLA policy on the 3D playground parkour.

Starts from the supervised (BC) checkpoint of ``SkillsVLAPolicy`` and fine-tunes
the discrete skill choice with PPO, using the same camera frame + 11-D state
inputs the BC policy was trained on.

Design
------
* Actor  : the BC ``skill_head`` (5 skill logits -> Categorical).
* Critic : a new ``value_head`` on the fused (vision + state) features.
* Env    : ``SkillArbitrationEnv`` with a one-hot macro-skill action, exactly
           like ``scripts/vla/eval_skills_vla.py``. The env ignores the 4-D
           continuous params, so the ``param_head`` is kept but not trained.
* Reward : the env's own shaped reward (progress, collisions, success bonus).
* Schedule: the ResNet backbone is frozen for the first ``--freeze-steps``
           env steps (only heads train), then unfrozen at ``--backbone-lr``.
* KL to BC: an optional penalty keeps the actor close to the BC policy so it
           does not forget the demonstrations early in training.

Run (headless):
    MUJOCO_GL=egl PYTHONPATH=. python scripts/vla/train_rl_skills_vla.py --total-steps 20000
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from radial_sphere.config import load_config_cli
from radial_sphere.run_id import build_run_id
from radial_sphere.scenario import generate_scenario
from radial_sphere.skill_arbitration_env import SkillArbitrationEnv
from radial_sphere.snapshot import make_run_dir, save_code
from scripts.data.generate_obstacle_vla_dataset import render_vla_frame
from scripts.vla.eval_skills_vla import evaluate_skills_vla
from scripts.vla.train_skills_vla import SkillsVLAPolicy
from skills_vla import ENV_SKILL_MAP, SKILL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_rl_skills_vla")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SkillsVLAActorCritic(SkillsVLAPolicy):
    """``SkillsVLAPolicy`` plus a value head. State-dict compatible with the BC checkpoint."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def features(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        vis_feat = self.visual_encoder(image).flatten(1)
        vis_emb = F.silu(self.visual_proj(vis_feat))
        state_emb = self.state_encoder(state)
        return torch.cat([vis_emb, state_emb], dim=1)

    def forward(self, image, state):
        fused = self.features(image, state)
        return self.skill_head(fused), self.param_head(fused), self.value_head(fused).squeeze(-1)

    def load_bc_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        # Only the fresh value head may be missing.
        bad = [k for k in missing if not k.startswith("value_head.")]
        if bad or unexpected:
            raise RuntimeError(f"BC checkpoint mismatch. missing={bad} unexpected={unexpected}")
        logger.info(f"Loaded BC weights from {path} (epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc')})")

    def set_backbone_trainable(self, flag: bool) -> None:
        for p in self.visual_encoder.parameters():
            p.requires_grad_(flag)


# --------------------------------------------------------------------------- #
# Env helpers (mirror eval_skills_vla.py so train/eval see the same inputs)
# --------------------------------------------------------------------------- #
def make_env(cfg, seed: int) -> SkillArbitrationEnv:
    scenario = generate_scenario("playground", cfg, seed=seed)
    env = SkillArbitrationEnv(cfg, scenario=scenario, seed=seed, max_steps=1500, training=True)
    return env


def observe(env: SkillArbitrationEnv, goal_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = render_vla_frame(env.env, width=256, height=256)
    pos = env.env.data.qpos[:3].copy()
    quat = env.env.data.qpos[3:7].copy()
    vel = env.env.data.qvel[:3].copy()
    rel_goal = goal_xy - pos[:2]
    state = np.concatenate([
        pos[:2].astype(np.float32),
        vel[:2].astype(np.float32),
        rel_goal.astype(np.float32),
        np.array([np.linalg.norm(rel_goal)], dtype=np.float32),
        quat.astype(np.float32),
    ])
    return frame, state


def skill_to_env_action(env: SkillArbitrationEnv, skill_idx: int) -> np.ndarray:
    env_skill = ENV_SKILL_MAP[SKILL_NAMES[skill_idx]]
    act = np.full(len(env.skill_names), -1.0, dtype=np.float32)
    act[env.skill_names.index(env_skill)] = 1.0
    return act


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #
class Rollout:
    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.device = device
        self.frames = np.zeros((n, 256, 256, 3), dtype=np.uint8)
        self.states = np.zeros((n, 11), dtype=np.float32)
        self.actions = np.zeros(n, dtype=np.int64)
        self.logp = np.zeros(n, dtype=np.float32)
        self.values = np.zeros(n, dtype=np.float32)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.dones = np.zeros(n, dtype=np.float32)
        self.bc_logits = np.zeros((n, len(SKILL_NAMES)), dtype=np.float32)

    def gae(self, last_value: float, gamma: float, lam: float):
        adv = np.zeros(self.n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self.n)):
            next_v = last_value if t == self.n - 1 else self.values[t + 1]
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_v * nonterminal - self.values[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        return adv, adv + self.values

    def to_tensors(self, idx: np.ndarray):
        img = torch.from_numpy(self.frames[idx]).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        st = torch.from_numpy(self.states[idx]).to(self.device)
        return img, st


def ppo_update(model, bc_model, optimizer, buf: Rollout, adv, ret, args, device):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv_t = torch.from_numpy(adv).to(device)
    ret_t = torch.from_numpy(ret).to(device)
    act_t = torch.from_numpy(buf.actions).to(device)
    old_logp_t = torch.from_numpy(buf.logp).to(device)
    bc_logits_t = torch.from_numpy(buf.bc_logits).to(device)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl_bc": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    n_batches = 0
    model.train()
    for _ in range(args.ppo_epochs):
        perm = np.random.permutation(buf.n)
        for start in range(0, buf.n, args.minibatch):
            idx = perm[start : start + args.minibatch]
            img, st = buf.to_tensors(idx)
            logits, _, values = model(img, st)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act_t[idx])
            ratio = torch.exp(logp - old_logp_t[idx])
            a = adv_t[idx]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
            v_loss = F.mse_loss(values, ret_t[idx])
            ent = dist.entropy().mean()
            # KL(BC || current) keeps the actor near the demonstrations.
            kl_bc = F.kl_div(F.log_softmax(logits, -1), F.softmax(bc_logits_t[idx], -1), reduction="batchmean")
            loss = pg + args.vf_coef * v_loss - args.ent_coef * ent + args.bc_kl_coef * kl_bc

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"] += pg.item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["kl_bc"] += kl_bc.item()
                stats["approx_kl"] += (old_logp_t[idx] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1).abs() > args.clip).float().mean().item()
            n_batches += 1
    return {k: v / max(1, n_batches) for k, v in stats.items()}


def build_optimizer(model, args, backbone_on: bool):
    heads = [p for n, p in model.named_parameters() if not n.startswith("visual_encoder.")]
    groups = [{"params": heads, "lr": args.lr}]
    if backbone_on:
        groups.append({"params": list(model.visual_encoder.parameters()), "lr": args.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=1e-5)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = make_run_dir(build_run_id("train_rl_skills_vla", tag=args.config_name))
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    cfg = load_config_cli(name=args.config_name)
    save_code(run_dir, __file__, cfg=cfg)
    logger.info(f"Run dir: {run_dir} | device: {device}")

    model = SkillsVLAActorCritic().to(device)
    model.load_bc_checkpoint(Path(args.bc_checkpoint))
    bc_model = copy.deepcopy(model).eval()
    for p in bc_model.parameters():
        p.requires_grad_(False)

    backbone_on = args.freeze_steps <= 0
    model.set_backbone_trainable(backbone_on)
    optimizer = build_optimizer(model, args, backbone_on)

    # One env for the whole run. Building a new env per episode leaks EGL
    # renderers and breaks rendering after ~150 episodes.
    env = make_env(cfg, seed=args.seed)
    ep_seed = args.seed
    obs_frame, obs_state = None, None

    def reset_env():
        nonlocal ep_seed, obs_frame, obs_state
        ep_seed += 1
        env.reset(seed=ep_seed)
        goal = np.asarray(env.scenario.goal, dtype=np.float32)[:2]
        obs_frame, obs_state = observe(env, goal)
        return goal

    goal_xy = reset_env()

    buf = Rollout(args.rollout_steps, device)
    history = []
    ep_returns, ep_successes, ep_lens, ep_hits = [], [], [], []
    cur_ret, cur_len = 0.0, 0
    best_score = -1e9  # success first, then fewer obstacle hits
    global_step = 0
    t0 = time.time()

    while global_step < args.total_steps:
        # Unfreeze the backbone once the heads have adapted.
        if not backbone_on and global_step >= args.freeze_steps:
            backbone_on = True
            model.set_backbone_trainable(True)
            optimizer = build_optimizer(model, args, backbone_on)
            logger.info(f"Step {global_step}: backbone unfrozen (lr={args.backbone_lr})")

        # ---- collect rollout ------------------------------------------------
        model.eval()
        for t in range(args.rollout_steps):
            with torch.no_grad():
                img = torch.from_numpy(obs_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                st = torch.from_numpy(obs_state).unsqueeze(0).to(device)
                logits, _, value = model(img, st)
                bc_logits, _, _ = bc_model(img, st)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            a = int(action.item())
            buf.frames[t] = obs_frame
            buf.states[t] = obs_state
            buf.actions[t] = a
            buf.logp[t] = logp.item()
            buf.values[t] = value.item()
            buf.bc_logits[t] = bc_logits[0].cpu().numpy()

            _, reward, terminated, truncated, info = env.step(skill_to_env_action(env, a))
            done = terminated or truncated or cur_len + 1 >= args.max_macro_steps
            buf.rewards[t] = reward
            buf.dones[t] = float(done)
            cur_ret += reward
            cur_len += 1
            global_step += 1

            if done:
                ep_returns.append(cur_ret)
                ep_lens.append(cur_len)
                ep_successes.append(float(info.get("success", False)))
                ep_hits.append(float(info.get("episode_hits", 0)))
                cur_ret, cur_len = 0.0, 0
                goal_xy = reset_env()
            else:
                obs_frame, obs_state = observe(env, goal_xy)

        with torch.no_grad():
            img = torch.from_numpy(obs_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            st = torch.from_numpy(obs_state).unsqueeze(0).to(device)
            _, _, last_value = model(img, st)
        adv, ret = buf.gae(last_value.item(), args.gamma, args.gae_lambda)

        # ---- update ---------------------------------------------------------
        stats = ppo_update(model, bc_model, optimizer, buf, adv, ret, args, device)

        recent_r = float(np.mean(ep_returns[-10:])) if ep_returns else float("nan")
        recent_s = float(np.mean(ep_successes[-10:])) if ep_successes else float("nan")
        recent_h = float(np.mean(ep_hits[-10:])) if ep_hits else float("nan")
        rec = {
            "step": global_step,
            "episodes": len(ep_returns),
            "mean_return_10": recent_r,
            "success_10": recent_s,
            "hits_10": recent_h,
            "mean_len_10": float(np.mean(ep_lens[-10:])) if ep_lens else float("nan"),
            "backbone_on": backbone_on,
            "elapsed_s": time.time() - t0,
            **stats,
        }
        history.append(rec)
        logger.info(
            f"step {global_step:6d} | eps {len(ep_returns):3d} | ret10 {recent_r:8.2f} | succ10 {recent_s:.2f} | hits10 {recent_h:4.1f} | "
            f"pg {stats['policy_loss']:.3f} vf {stats['value_loss']:.2f} ent {stats['entropy']:.3f} "
            f"klbc {stats['kl_bc']:.4f} | {rec['elapsed_s']:.0f}s"
        )

        torch.save({"model_state_dict": model.state_dict(), "step": global_step, "skills": list(SKILL_NAMES)},
                   ckpt_dir / "last_policy.pt")
        score = recent_s - 0.01 * recent_h if ep_successes else -1e9
        if len(ep_successes) >= 3 and score >= best_score:
            best_score = score
            torch.save({"model_state_dict": model.state_dict(), "step": global_step, "skills": list(SKILL_NAMES),
                        "success_10": recent_s, "hits_10": recent_h}, ckpt_dir / "best_policy.pt")

    env.close()
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump({"args": vars(args), "history": history, "best_score": best_score}, f, indent=2)
    logger.info(f"Training done in {time.time() - t0:.0f}s. Checkpoints in {ckpt_dir}")

    # ---- closed-loop eval with video ----------------------------------------
    if args.eval_episodes > 0:
        ckpt = ckpt_dir / ("best_policy.pt" if (ckpt_dir / "best_policy.pt").exists() else "last_policy.pt")
        evaluate_skills_vla(
            checkpoint_path=ckpt,
            config_name=args.config_name,
            output_dir=run_dir / "eval",
            episodes=args.eval_episodes,
            seed_offset=args.eval_seed,
        )


def main():
    p = argparse.ArgumentParser(description="PPO fine-tuning of the Skills VLA policy.")
    p.add_argument("--bc-checkpoint", default="storage_local/20260914_1317__local__train_skills_vla__v2_clean_oracle/best_policy.pt")
    p.add_argument("--config-name", default="playground_parkour_skills")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--total-steps", type=int, default=20000, help="env macro steps")
    p.add_argument("--rollout-steps", type=int, default=512)
    p.add_argument("--max-macro-steps", type=int, default=300)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4, help="heads lr")
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--freeze-steps", type=int, default=10**9, help="env steps with frozen ResNet (default: always frozen)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--bc-kl-coef", type=float, default=1.0)
    p.add_argument("--eval-episodes", type=int, default=2)
    p.add_argument("--eval-seed", type=int, default=150)
    train(p.parse_args())


if __name__ == "__main__":
    main()
