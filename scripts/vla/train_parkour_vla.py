"""Train / Fine-Tune Parkour Vision-Language-Action (VLA) Policy.

Predicts macro skills (follow_path, jump_to, traverse_rough_terrain, etc.)
from RGB camera frames + robot state for the 3D Playground Parkour task.
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_parkour_vla")


class ParkourVLADataset(Dataset):
    """Loads visual frames and macro skill actions from HDF5 demonstrations."""

    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        self.samples = []

        with h5py.File(h5_path, "r") as f:
            for ep_name in f.keys():
                ep_grp = f[ep_name]
                n_steps = len(ep_grp["actions"])
                for step_idx in range(n_steps):
                    self.samples.append((ep_name, step_idx))

        logger.info(f"Loaded {len(self.samples)} transitions from {h5_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_name, step_idx = self.samples[idx]
        with h5py.File(self.h5_path, "r") as f:
            ep_grp = f[ep_name]
            frame = ep_grp["frames"][step_idx]  # (256, 256, 3) uint8
            state = ep_grp["states"][step_idx]  # (11,) float32
            action = ep_grp["actions"][step_idx]  # (10,) float32

        # Convert to tensor: (3, 256, 256) normalized to [0, 1]
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        st_tensor = torch.from_numpy(state).float()
        act_tensor = torch.from_numpy(action).float()

        return {"image": img_tensor, "state": st_tensor, "action": act_tensor}


class ParkourVLAPolicy(nn.Module):
    """Multimodal Vision-Language-Action Policy for Parkour Navigation."""

    def __init__(self, state_dim: int = 11, action_dim: int = 10, hidden_dim: int = 256):
        super().__init__()
        # Visual backbone: ResNet18 feature extractor
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.vision_encoder = resnet

        self.vision_proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Proprioceptive / guidance state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # Multimodal fusion & macro action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        vis_feat = self.vision_encoder(images)
        vis_embed = self.vision_proj(vis_feat)
        st_embed = self.state_encoder(states)

        fused = torch.cat([vis_embed, st_embed], dim=-1)
        actions = self.action_head(fused)
        return actions


def train_parkour_vla(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 3e-4,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Parkour VLA on device: {device}")

    full_dataset = ParkourVLADataset(dataset_path)
    val_size = max(int(len(full_dataset) * 0.15), 1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    policy = ParkourVLAPolicy(state_dim=11, action_dim=10).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    metrics_history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        policy.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            states = batch["state"].to(device, non_blocking=True)
            actions = batch["action"].to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = policy(images, states)
            loss = criterion(preds, actions)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                states = batch["state"].to(device, non_blocking=True)
                actions = batch["action"].to(device, non_blocking=True)

                preds = policy(images, states)
                loss = criterion(preds, actions)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_dataset)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "state_dim": 11,
                    "action_dim": 10,
                },
                output_dir / "best_parkour_policy.pt",
            )

        metrics_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "is_best": is_best})

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Best: {best_val_loss:.5f}{' (*)' if is_best else ''}"
        )

    # Save final model
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": policy.state_dict(),
            "val_loss": val_loss,
            "state_dim": 11,
            "action_dim": 10,
        },
        output_dir / "final_parkour_policy.pt",
    )

    with open(output_dir / "parkour_vla_training_metrics.json", "w") as f:
        json.dump({"best_val_loss": best_val_loss, "history": metrics_history}, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"PARKOUR VLA FINE-TUNING COMPLETE in {time.time() - start_time:.1f}s")
    logger.info(f"Best Val Loss: {best_val_loss:.5f}")
    logger.info(f"Checkpoint:    {output_dir / 'best_parkour_policy.pt'}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VLA policy for playground parkour.")
    parser.add_argument(
        "--data",
        type=str,
        default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/dataset/playground_vla_demos.h5",
        help="Dataset path",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="storage_local/20260912_2239__local_601729__train_rl__playground_parkour_skills/checkpoints",
        help="Checkpoints output directory",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    train_parkour_vla(
        dataset_path=Path(args.data),
        output_dir=Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
