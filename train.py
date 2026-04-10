"""
CausalComp training script.

Usage:
    python train.py                          # default config
    python train.py --exp_name debug --num_epochs 5  # quick test
    python train.py --batch_size 64 --lr 2e-4        # override params
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import Config
from data.clevrer_dataset import CLEVRERDataset, clevrer_collate_fn
from models.causalcomp import CausalComp
from utils.logger import setup_logger, log_metrics
from utils.visualize import visualize_slots, visualize_graph


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config: Config) -> CausalComp:
    model = CausalComp(
        resolution=config.data.resolution,
        num_slots=config.slot.num_slots,
        slot_dim=config.slot.slot_dim,
        num_interaction_types=config.causal.num_interaction_types,
        encoder_channels=config.slot.encoder_channels,
        dynamics_hidden=config.dynamics.hidden_dim,
        num_message_passing=config.dynamics.num_message_passing,
        gumbel_temperature=config.causal.gumbel_temperature,
    )
    return model


def train_one_epoch(model, loader, optimizer, config, epoch, logger):
    model.train()
    total_losses = {}
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        video = batch["video"].to(config.device)  # [B, T, 3, H, W]

        # Forward
        output = model(video, rollout_steps=config.train.rollout_steps)
        losses = model.compute_loss(output, config.train)

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()
        if config.train.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

        # Accumulate losses
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()
        num_batches += 1

        # Log
        if batch_idx % config.train.log_interval == 0:
            avg = {k: v / num_batches for k, v in total_losses.items()}
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"loss={avg['total']:.4f} recon={avg['recon']:.4f} "
                f"dyn={avg['dynamics']:.4f} sparse={avg['sparsity']:.4f}"
            )

    return {k: v / num_batches for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(model, loader, config):
    model.eval()
    total_losses = {}
    num_batches = 0

    for batch in loader:
        video = batch["video"].to(config.device)
        output = model(video, rollout_steps=config.train.rollout_steps)
        losses = model.compute_loss(output, config.train)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()
        num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in total_losses.items()}


def main():
    parser = argparse.ArgumentParser(description="Train CausalComp")
    parser.add_argument("--exp_name", type=str, default="causalcomp_v1")
    parser.add_argument("--data_dir", type=str, default="./data/clevrer")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_slots", type=int, default=None)
    parser.add_argument("--num_interaction_types", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    # Build config with overrides
    config = Config()
    config.exp_name = args.exp_name
    config.seed = args.seed
    config.device = args.device if torch.cuda.is_available() else "cpu"
    config.data.data_dir = args.data_dir

    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    if args.num_epochs is not None:
        config.train.num_epochs = args.num_epochs
    if args.num_slots is not None:
        config.slot.num_slots = args.num_slots
    if args.num_interaction_types is not None:
        config.causal.num_interaction_types = args.num_interaction_types
    if args.resolution is not None:
        config.data.resolution = args.resolution
    if args.rollout_steps is not None:
        config.train.rollout_steps = args.rollout_steps

    set_seed(config.seed)

    # Experiment directory
    exp_dir = Path("experiments") / config.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    logger = setup_logger(config.exp_name, exp_dir / "train.log")
    logger.info(f"Config: {config}")
    logger.info(f"Device: {config.device}")

    # Wandb
    if args.wandb:
        import wandb
        wandb.init(project=config.train.wandb_project, name=config.exp_name)

    # Data — both train and val scan the same data_dir for mp4 files.
    # When full dataset is available, use separate dirs via --data_dir.
    # For now, we split the found videos 90/10.
    full_dataset = CLEVRERDataset(
        data_dir=config.data.data_dir,
        split="validation",  # split name only affects annotation loading
        num_frames=config.data.num_frames,
        frame_skip=config.data.frame_skip,
        resolution=config.data.resolution,
    )
    n_total = len(full_dataset)
    n_val = max(1, n_total // 10)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=clevrer_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=clevrer_collate_fn,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} videos, Val: {len(val_dataset)} videos")

    # Model
    model = create_model(config).to(config.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.num_epochs, eta_min=1e-6
    )

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, config.train.num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, config, epoch, logger
        )
        scheduler.step()

        # Evaluate
        if epoch % config.train.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, config)
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch}/{config.train.num_epochs} ({elapsed:.1f}s) | "
                f"Train loss={train_metrics['total']:.4f} | "
                f"Val loss={val_metrics['total']:.4f} | "
                f"Val recon={val_metrics['recon']:.4f} "
                f"Val dyn={val_metrics['dynamics']:.4f}"
            )

            if args.wandb:
                log_metrics(
                    {f"train/{k}": v for k, v in train_metrics.items()},
                    {f"val/{k}": v for k, v in val_metrics.items()},
                    epoch=epoch,
                )

            # Save best
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                }, ckpt_dir / "best.pt")
                logger.info(f"  -> New best model saved (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if epoch % config.train.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, ckpt_dir / f"epoch_{epoch}.pt")

    logger.info("Training complete.")
    logger.info(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
