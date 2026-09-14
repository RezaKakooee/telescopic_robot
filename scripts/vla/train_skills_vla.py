"""Train and Fine-Tune Vision-Language-Action (VLA) Policy for Modular Parkour Skills.

Trains a multi-task multimodal policy network:
  Input:
    - RGB camera frame (256x256x3)
    - Proprioceptive state vector (11-D: pos, vel, rel_goal, dist_goal, quat)
  Outputs:
    - Skill classification logits (5 classes: roll, jump_forward, jump_gap, traverse_rough, brake_stop)
    - Continuous skill parameter predictions (4-D in [-1, 1])

Supports initializing weights from the pre-trained obstacle navigation VLA backbone.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models

from radial_sphere.run_id import build_run_id
from radial_sphere.snapshot import make_run_dir
from skills_vla import SKILL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_skills_vla")


class SkillsVLADataset(Dataset):
    """Loads RGB frames, robot state vectors, discrete skill targets, and continuous parameters."""

    def __init__(self, h5_path: Path, split: str = "all", train_ratio: float = 0.85):
        self.h5_path = h5_path

        frames_list = []
        states_list = []
        skills_list = []
        params_list = []

        with h5py.File(h5_path, "r") as f:
            ep_keys = sorted(list(f.keys()))
            n_train = max(1, int(len(ep_keys) * train_ratio))
            if split == "train":
                selected_keys = ep_keys[:n_train]
            elif split == "val":
                selected_keys = ep_keys[n_train:]
                if len(selected_keys) == 0:
                    selected_keys = ep_keys[-1:]
            else:
                selected_keys = ep_keys

            for ep_name in selected_keys:
                grp = f[ep_name]
                frames_list.append(grp["frames"][:])
                states_list.append(grp["states"][:])
                skills_list.append(grp["skills"][:])
                params_list.append(grp["params"][:])

        self.frames = np.concatenate(frames_list, axis=0)
        self.states = np.concatenate(states_list, axis=0)
        self.skills = np.concatenate(skills_list, axis=0)
        self.params = np.concatenate(params_list, axis=0)

        logger.info(f"Loaded {len(self.skills)} samples into memory for split='{split}' from {h5_path}")

    def __len__(self) -> int:
        return len(self.skills)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frame = self.frames[idx]
        state = self.states[idx]
        skill = self.skills[idx]
        params = self.params[idx]

        # Convert to tensor: (3, 256, 256) normalized to [0, 1]
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        state_tensor = torch.from_numpy(state).float()
        skill_tensor = torch.tensor(skill, dtype=torch.long)
        params_tensor = torch.from_numpy(params).float()

        return {
            "image": img_tensor,
            "state": state_tensor,
            "skill": skill_tensor,
            "params": params_tensor,
        }


class SkillsVLAPolicy(nn.Module):
    """Multimodal Vision-Language-Action Policy for Modular Parkour Skills.

    Features:
      - Visual Encoder: ResNet18 backbone extracting 512-D spatial features
      - State MLP: Encodes 11-D proprioception into 256-D embedding
      - Multi-Task Decision Heads:
          * Discrete Skill Head: 5-way classification logits
          * Continuous Param Head: 4-D normalized skill execution parameters in [-1, 1]
    """

    def __init__(
        self,
        state_dim: int = 11,
        num_skills: int = len(SKILL_NAMES),
        param_dim: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim

        # 1. Visual backbone: ResNet18 adapted for 256x256 images
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Output: (B, 512, 1, 1)
        self.visual_proj = nn.Linear(512, hidden_dim)

        # 2. State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. Discrete Skill Classification Head (roll, jump_forward, jump_gap, traverse_rough, brake_stop)
        self.skill_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_skills),
        )

        # 4. Continuous Parameter Regression Head in [-1, 1]
        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, param_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting skill classification logits and continuous parameters.

        Returns
        -------
        skill_logits : (B, num_skills)
        pred_params : (B, param_dim) in [-1, 1]
        """
        # Visual features
        vis_feat = self.visual_encoder(image).flatten(1)  # (B, 512)
        vis_emb = F.silu(self.visual_proj(vis_feat))       # (B, hidden_dim)

        # State features
        state_emb = self.state_encoder(state)             # (B, hidden_dim)

        # Multimodal fusion
        fused = torch.cat([vis_emb, state_emb], dim=1)    # (B, hidden_dim * 2)

        # Multi-task predictions
        skill_logits = self.skill_head(fused)
        pred_params = self.param_head(fused)

        return skill_logits, pred_params

    def load_pretrained_backbone(self, checkpoint_path: Path):
        """Load pretrained visual encoder and state encoder weights from continuous VLA checkpoint."""
        if not checkpoint_path.exists():
            logger.warning(f"Pretrained checkpoint {checkpoint_path} not found; initializing from scratch.")
            return

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)

        loaded_keys = []
        model_dict = self.state_dict()
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
                loaded_keys.append(k)

        self.load_state_dict(model_dict)
        logger.info(f"Loaded {len(loaded_keys)} pretrained weight tensors from {checkpoint_path}")


def train_skills_vla(
    dataset_path: Path,
    output_dir: Path | None = None,
    pretrained_checkpoint: Path | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    param_loss_weight: float = 0.5,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if output_dir is None:
        output_dir = make_run_dir(build_run_id("train_skills_vla", tag=dataset_path.parent.name))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_dataset = SkillsVLADataset(dataset_path, split="train", train_ratio=0.85)
    val_dataset = SkillsVLADataset(dataset_path, split="val", train_ratio=0.85)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = SkillsVLAPolicy().to(device)

    if pretrained_checkpoint and pretrained_checkpoint.exists():
        logger.info(f"Initializing backbone from {pretrained_checkpoint}...")
        model.load_pretrained_backbone(pretrained_checkpoint)

    # Class weighting to handle skill imbalance (e.g. roll is more frequent than jump)
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_ckpt_path = output_dir / "best_policy.pt"
    history = []

    logger.info(f"Starting training for {epochs} epochs (batch_size={batch_size}, lr={lr})...")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_mse = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            states = batch["state"].to(device, non_blocking=True)
            skills = batch["skill"].to(device, non_blocking=True)
            params = batch["params"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, pred_params = model(images, states)

            loss_ce = ce_loss_fn(logits, skills)
            loss_mse = mse_loss_fn(pred_params, params)
            total_loss = loss_ce + param_loss_weight * loss_mse

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = images.size(0)
            train_loss += total_loss.item() * bs
            train_ce += loss_ce.item() * bs
            train_mse += loss_mse.item() * bs

            preds = logits.argmax(dim=1)
            train_correct += (preds == skills).sum().item()
            train_total += bs

        train_loss /= max(1, train_total)
        train_ce /= max(1, train_total)
        train_mse /= max(1, train_total)
        train_acc = 100.0 * train_correct / max(1, train_total)

        scheduler.step()

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_ce = 0.0
        val_mse = 0.0
        val_correct = 0
        val_total = 0
        per_skill_correct = {name: 0 for name in SKILL_NAMES}
        per_skill_total = {name: 0 for name in SKILL_NAMES}

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                states = batch["state"].to(device, non_blocking=True)
                skills = batch["skill"].to(device, non_blocking=True)
                params = batch["params"].to(device, non_blocking=True)

                logits, pred_params = model(images, states)

                loss_ce = ce_loss_fn(logits, skills)
                loss_mse = mse_loss_fn(pred_params, params)
                total_loss = loss_ce + param_loss_weight * loss_mse

                bs = images.size(0)
                val_loss += total_loss.item() * bs
                val_ce += loss_ce.item() * bs
                val_mse += loss_mse.item() * bs

                preds = logits.argmax(dim=1)
                val_correct += (preds == skills).sum().item()
                val_total += bs

                for p, t in zip(preds.cpu().numpy(), skills.cpu().numpy()):
                    skill_name = SKILL_NAMES[t]
                    per_skill_total[skill_name] += 1
                    if p == t:
                        per_skill_correct[skill_name] += 1

        val_loss /= max(1, val_total)
        val_ce /= max(1, val_total)
        val_mse /= max(1, val_total)
        val_acc = 100.0 * val_correct / max(1, val_total)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_mse": train_mse,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_mse": val_mse,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.1f}%, MSE: {train_mse:.4f}) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.1f}%, MSE: {val_mse:.4f})"
        )

        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_mse": val_mse,
                    "skills": list(SKILL_NAMES),
                },
                best_ckpt_path,
            )
            logger.info(f"  ★ Saved new best policy (val_acc={val_acc:.1f}%) to {best_ckpt_path}")

    total_time = time.time() - start_time
    logger.info("=" * 65)
    logger.info(f"TRAINING COMPLETE in {total_time:.1f}s | Best Val Acc: {best_val_acc:.1f}%")
    logger.info(f"Best Checkpoint: {best_ckpt_path}")
    logger.info("=" * 65)

    metrics_file = output_dir / "training_summary.json"
    with open(metrics_file, "w") as f:
        json.dump(
            {
                "best_val_acc": best_val_acc,
                "total_time_s": total_time,
                "epochs": epochs,
                "batch_size": batch_size,
                "history": history,
                "skills": list(SKILL_NAMES),
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="Train / Fine-Tune Skills VLA Policy on Parkour Demos.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="storage_local/20260914_1250__local__generate_skills_vla_dataset__v2_clean_oracle/parkour_skills_vla_demos.h5",
        help="Path to HDF5 demonstration dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for checkpoints and summaries (default: a new timestamped run dir under storage_local/)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="storage_local/20260913_1525__local__train_vla__obstacle_navigation/checkpoints/best_policy.pt",
        help="Path to pretrained continuous VLA checkpoint for backbone initialization",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    train_skills_vla(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        pretrained_checkpoint=Path(args.pretrained) if args.pretrained else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
