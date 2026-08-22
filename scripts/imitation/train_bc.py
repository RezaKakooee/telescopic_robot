"""Imitation Learning (Behavioral Cloning) Pre-training for Radial-Sphere Robot.

Trains:
1. High-Level Steering BC Policy: obs (37D / LiDAR + goal) -> steering actions [sx, sy, drive]
2. Low-Level Actuator BC Policy: obs (163D / proprioception + LiDAR) -> 60D rod extension targets
3. Joint Hierarchical Policy: Shared encoder predicting both high-level navigation & low-level coordination
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import h5py
import numpy as np
import rootutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

rootutils.setup_root(__file__, pythonpath=True)

from radial_sphere import setup_logging

log = logging.getLogger("radial_sphere")
setup_logging()


class MazeDemonstrationDataset(Dataset):
    """PyTorch Dataset loading expert maze demonstrations from NPZ or HDF5."""

    def __init__(self, data_dir: Path, mode: str = "joint", split: str = "train", train_ratio: float = 0.9):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.mode = mode  # 'highlevel', 'lowlevel', 'joint'
        
        npz_files = sorted((self.data_dir / "episodes_npz").glob("ep_*.npz"))
        assert len(npz_files) > 0, f"No demonstration NPZ files found in {self.data_dir / 'episodes_npz'}"
        
        # Split train / val
        n_train = int(len(npz_files) * train_ratio)
        if split == "train":
            self.files = npz_files[:n_train]
        else:
            self.files = npz_files[n_train:]

        log.info(f"Loading {split} dataset: {len(self.files)} episodes from {self.data_dir} (mode={mode})")

        # Load all transitions into memory for blazing fast GPU training
        self.obs_highlevel = []
        self.obs_lowlevel = []
        self.act_highlevel = []
        self.act_lowlevel = []

        for f in self.files:
            data = np.load(str(f))
            self.obs_highlevel.append(data["obs_highlevel"])
            self.obs_lowlevel.append(data["obs_lowlevel"])
            self.act_highlevel.append(data["action_highlevel"])
            self.act_lowlevel.append(data["action_lowlevel"])

        self.obs_highlevel = np.concatenate(self.obs_highlevel, axis=0).astype(np.float32)
        self.obs_lowlevel = np.concatenate(self.obs_lowlevel, axis=0).astype(np.float32)
        self.act_highlevel = np.concatenate(self.act_highlevel, axis=0).astype(np.float32)
        self.act_lowlevel = np.concatenate(self.act_lowlevel, axis=0).astype(np.float32)

        self.n_samples = self.obs_highlevel.shape[0]
        log.info(f"{split.upper()} set loaded: {self.n_samples:,} transitions.")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "obs_highlevel": torch.from_numpy(self.obs_highlevel[idx]),
            "obs_lowlevel": torch.from_numpy(self.obs_lowlevel[idx]),
            "act_highlevel": torch.from_numpy(self.act_highlevel[idx]),
            "act_lowlevel": torch.from_numpy(self.act_lowlevel[idx]),
        }


class HierarchicalImitationPolicy(nn.Module):
    """Hierarchical Neural Network with shared representation and dual heads."""

    def __init__(self, obs_dim: int = 163, high_act_dim: int = 3, low_act_dim: int = 60, hidden_dim: int = 256):
        super().__init__()
        # Backbone encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # High-level steering & drive head
        self.highlevel_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, high_act_dim),
            nn.Tanh(),  # [-1, 1] range for steering & throttle
        )

        # Low-level 60D actuator extension head
        self.lowlevel_head = nn.Sequential(
            nn.Linear(hidden_dim + high_act_dim, 256),
            nn.SiLU(),
            nn.Linear(256, low_act_dim),
            nn.Sigmoid(),  # [0, 1] normalized extension -> scaled to [min_offset, max_extend]
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(obs)
        act_high = self.highlevel_head(feat)
        # Condition low-level head on both latent state + high-level action
        low_input = torch.cat([feat, act_high], dim=-1)
        norm_low = self.lowlevel_head(low_input)
        # Scale to [0.025, 0.20] physical range
        act_low = 0.025 + norm_low * (0.20 - 0.025)
        return act_high, act_low


def train_bc(
    data_dir: Path = Path("datasets/maze_demos"),
    output_dir: Path = Path("storage_local/imitation_models"),
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Starting Behavioral Cloning on device: {device}")

    train_ds = MazeDemonstrationDataset(data_dir, split="train")
    val_ds = MazeDemonstrationDataset(data_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    obs_dim = train_ds.obs_lowlevel.shape[1]
    model = HierarchicalImitationPolicy(obs_dim=obs_dim, high_act_dim=3, low_act_dim=60, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        train_high_loss = 0.0
        train_low_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            obs = batch["obs_lowlevel"].to(device)
            target_high = batch["act_highlevel"].to(device)
            target_low = batch["act_lowlevel"].to(device)

            pred_high, pred_low = model(obs)

            loss_high = F.mse_loss(pred_high, target_high)
            loss_low = F.mse_loss(pred_low, target_low) * 100.0  # Scale meter MSE to cm scale

            total_loss = loss_high + loss_low

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_high_loss += float(loss_high.item())
            train_low_loss += float(loss_low.item())
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_high_loss = 0.0
        val_low_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs_lowlevel"].to(device)
                target_high = batch["act_highlevel"].to(device)
                target_low = batch["act_lowlevel"].to(device)

                pred_high, pred_low = model(obs)

                val_high_loss += float(F.mse_loss(pred_high, target_high).item())
                val_low_loss += float(F.mse_loss(pred_low, target_low).item() * 100.0)
                n_val += 1

        avg_val_loss = (val_high_loss + val_low_loss) / max(n_val, 1)
        history.append({
            "epoch": ep,
            "train_high_mse": train_high_loss / n_batches,
            "train_low_mse": train_low_loss / n_batches,
            "val_high_mse": val_high_loss / max(n_val, 1),
            "val_low_mse": val_low_loss / max(n_val, 1),
            "val_total": avg_val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if ep % 5 == 0 or ep == epochs:
            log.info(
                f"Epoch [{ep:03d}/{epochs:03d}] | "
                f"Train High MSE: {train_high_loss/n_batches:.5f} | "
                f"Train Low MSE: {train_low_loss/n_batches:.5f} | "
                f"Val High MSE: {val_high_loss/max(n_val,1):.5f} | "
                f"Val Low MSE: {val_low_loss/max(n_val,1):.5f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = output_dir / "bc_hierarchical_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim,
                "epoch": ep,
                "val_loss": best_val_loss,
            }, str(ckpt_path))

    # Save final model & metrics
    final_path = output_dir / "bc_hierarchical_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "epoch": epochs,
        "val_loss": avg_val_loss,
    }, str(final_path))

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"\n[BEHAVIORAL CLONING COMPLETE]")
    log.info(f"Best Validation Loss: {best_val_loss:.5f} saved to: {output_dir / 'bc_hierarchical_best.pt'}")


def main():
    p = argparse.ArgumentParser(description="Imitation Learning BC Pre-training")
    p.add_argument("--data-dir", default="datasets/maze_demos", help="Demonstrations directory")
    p.add_argument("--out-dir", default="storage_local/imitation_models", help="Model checkpoint directory")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = p.parse_args()

    train_bc(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
