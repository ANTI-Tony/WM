"""
Compositional generalization evaluation.

Trains on videos with seen collision-type pairs, evaluates on:
1. test_seen: held-out samples of training combinations
2. test_unseen: collision pairs NEVER seen during training

Reports: Seen MSE, Unseen MSE, Harmonic Mean, Compositional Gap.

Also trains ablation baselines:
- CausalComp (full): graph discovery + typed modules
- NoGraph: no graph, just self-dynamics (all edges off)
- FullGraph: all edges on with equal weight (no discovery)
- SingleModule: graph discovery but single shared dynamics (no types)

Usage:
    python eval_compositional.py --num_epochs 100 --num_videos 5000 --synthetic
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from configs import Config
from data.synthetic_dataset import SyntheticPhysicsDataset, synthetic_collate_fn
from data.compositional_split import (
    create_compositional_split, classify_video, ALL_TYPES
)
from train_gt import GTCausalComp, GT_STATE_DIM, compute_loss
from utils.logger import setup_logger


def split_dataset(dataset, split_info):
    """Split dataset into train/test_seen/test_unseen indices."""
    train_idx, test_unseen_idx = [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        props = sample["objects"]["properties"]
        label = classify_video(props, sample["events"], split_info)
        if label == "train":
            train_idx.append(i)
        else:
            test_unseen_idx.append(i)

    # Split train into train (90%) and test_seen (10%)
    rng = random.Random(42)
    rng.shuffle(train_idx)
    n_seen = max(1, len(train_idx) // 10)
    test_seen_idx = train_idx[-n_seen:]
    train_idx = train_idx[:-n_seen]

    return train_idx, test_seen_idx, test_unseen_idx


@torch.no_grad()
def evaluate_split(model, loader, device, collision_adj_available=True):
    """Evaluate state prediction MSE on a data split."""
    model.eval()
    total_mse = 0
    total_samples = 0

    for batch in loader:
        gt_states = batch["gt_states"].to(device)
        output = model(gt_states, rollout_steps=8)

        pred = output["pred_states"]
        target = output["target_states"]
        T = min(pred.shape[1], target.shape[1])
        if T > 0:
            mse = F.mse_loss(pred[:, :T], target[:, :T], reduction="sum")
            total_mse += mse.item()
            total_samples += pred[:, :T].numel()

    return total_mse / max(total_samples, 1)


def train_model(model, train_loader, val_loader, config, device, logger, tag=""):
    """Train a model and return it."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0
        n = 0
        for batch in train_loader:
            gt_states = batch["gt_states"].to(device)
            collision_adj = batch["collision_adj"].to(device)
            output = model(gt_states, rollout_steps=config.rollout_steps)
            losses = compute_loss(output, collision_adj=collision_adj)

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += losses["total"].item()
            n += 1

        scheduler.step()

        if epoch % 20 == 0 or epoch == config.num_epochs:
            val_mse = evaluate_split(model, val_loader, device)
            logger.info(f"  [{tag}] Epoch {epoch}: loss={epoch_loss/n:.4f} val_mse={val_mse:.6f}")

    return model


class NoGraphDynamics(nn.Module):
    """Baseline: no interaction graph, only self-dynamics."""

    def __init__(self, state_dim=8, slot_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.dynamics = nn.Sequential(
            nn.Linear(slot_dim, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )
        self.decoder = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, state_dim))

    def forward(self, gt_states, rollout_steps=5):
        B, T, K, D = gt_states.shape
        slots = self.encoder(gt_states[:, 0])
        preds = []
        for _ in range(min(T - 1, rollout_steps)):
            slots = self.dynamics(slots)
            preds.append(self.decoder(slots))
            slots = self.encoder(preds[-1])
        return {
            "pred_states": torch.stack(preds, dim=1),
            "target_states": gt_states[:, 1:rollout_steps+1],
            "graph_infos": [],
        }


class FullGraphDynamics(nn.Module):
    """Baseline: all edges equally weighted (no discovery)."""

    def __init__(self, state_dim=8, slot_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.self_dyn = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.inter_dyn = nn.Sequential(nn.Linear(slot_dim * 2, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.update = nn.Sequential(nn.Linear(slot_dim * 2, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.decoder = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, state_dim))

    def forward(self, gt_states, rollout_steps=5):
        B, T, K, D = gt_states.shape
        slots = self.encoder(gt_states[:, 0])
        preds = []
        for _ in range(min(T - 1, rollout_steps)):
            delta_self = self.self_dyn(slots)
            # All-to-all interaction (uniform weight)
            si = slots.unsqueeze(2).expand(B, K, K, -1)
            sj = slots.unsqueeze(1).expand(B, K, K, -1)
            effects = self.inter_dyn(torch.cat([si, sj], dim=-1))
            delta_inter = effects.mean(dim=1)  # average over all senders
            slots = self.update(torch.cat([delta_self, delta_inter], dim=-1))
            preds.append(self.decoder(slots))
            slots = self.encoder(preds[-1])
        return {
            "pred_states": torch.stack(preds, dim=1),
            "target_states": gt_states[:, 1:rollout_steps+1],
            "graph_infos": [],
        }


def compute_baseline_loss(output):
    """Simple MSE loss for baselines without graph."""
    pred = output["pred_states"]
    target = output["target_states"]
    T = min(pred.shape[1], target.shape[1])
    step_losses = []
    for t in range(T):
        step_losses.append((1.0 + t) * F.mse_loss(pred[:, t], target[:, t]))
    return {"total": 5.0 * torch.stack(step_losses).mean(),
            "state_pred": torch.stack(step_losses).mean(),
            "edge_sup": torch.tensor(0.0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_videos", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_dir = Path("experiments/comp_eval")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("comp_eval", exp_dir / "eval.log")

    # Create compositional split
    split_info = create_compositional_split(seed=args.seed)

    # Generate dataset
    dataset = SyntheticPhysicsDataset(
        num_videos=args.num_videos, num_frames=16, resolution=64, seed=args.seed,
    )

    # Split
    logger.info("Splitting dataset...")
    train_idx, seen_idx, unseen_idx = split_dataset(dataset, split_info)
    logger.info(f"  Train: {len(train_idx)}, Test-seen: {len(seen_idx)}, Test-unseen: {len(unseen_idx)}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=2, collate_fn=synthetic_collate_fn, drop_last=True)
    seen_loader = DataLoader(Subset(dataset, seen_idx), batch_size=args.batch_size,
                             shuffle=False, num_workers=2, collate_fn=synthetic_collate_fn)
    unseen_loader = DataLoader(Subset(dataset, unseen_idx), batch_size=args.batch_size,
                               shuffle=False, num_workers=2, collate_fn=synthetic_collate_fn)

    # ========== Train and evaluate all models ==========
    results = {}

    # --- 1. CausalComp (full) ---
    logger.info("=== Training CausalComp (full) ===")
    model_full = GTCausalComp(state_dim=8, slot_dim=64, num_interaction_types=4).to(device)
    # Use compute_loss from train_gt for full model
    original_compute = compute_loss
    train_model(model_full, train_loader, seen_loader, args, device, logger, "CausalComp")
    seen_mse = evaluate_split(model_full, seen_loader, device)
    unseen_mse = evaluate_split(model_full, unseen_loader, device)
    results["CausalComp"] = {"seen": seen_mse, "unseen": unseen_mse}

    # --- 2. NoGraph baseline ---
    logger.info("=== Training NoGraph baseline ===")
    model_ng = NoGraphDynamics(state_dim=8, slot_dim=64).to(device)
    # Train with simple MSE loss
    opt_ng = torch.optim.AdamW(model_ng.parameters(), lr=args.lr)
    sched_ng = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ng, T_max=args.num_epochs)
    for epoch in range(1, args.num_epochs + 1):
        model_ng.train()
        for batch in train_loader:
            gt = batch["gt_states"].to(device)
            out = model_ng(gt, rollout_steps=args.rollout_steps)
            loss = compute_baseline_loss(out)["total"]
            opt_ng.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model_ng.parameters(), 1.0); opt_ng.step()
        sched_ng.step()
        if epoch % 20 == 0:
            logger.info(f"  [NoGraph] Epoch {epoch}: loss={loss.item():.4f}")
    seen_mse = evaluate_split(model_ng, seen_loader, device)
    unseen_mse = evaluate_split(model_ng, unseen_loader, device)
    results["NoGraph"] = {"seen": seen_mse, "unseen": unseen_mse}

    # --- 3. FullGraph baseline ---
    logger.info("=== Training FullGraph baseline ===")
    model_fg = FullGraphDynamics(state_dim=8, slot_dim=64).to(device)
    opt_fg = torch.optim.AdamW(model_fg.parameters(), lr=args.lr)
    sched_fg = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fg, T_max=args.num_epochs)
    for epoch in range(1, args.num_epochs + 1):
        model_fg.train()
        for batch in train_loader:
            gt = batch["gt_states"].to(device)
            out = model_fg(gt, rollout_steps=args.rollout_steps)
            loss = compute_baseline_loss(out)["total"]
            opt_fg.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model_fg.parameters(), 1.0); opt_fg.step()
        sched_fg.step()
        if epoch % 20 == 0:
            logger.info(f"  [FullGraph] Epoch {epoch}: loss={loss.item():.4f}")
    seen_mse = evaluate_split(model_fg, seen_loader, device)
    unseen_mse = evaluate_split(model_fg, unseen_loader, device)
    results["FullGraph"] = {"seen": seen_mse, "unseen": unseen_mse}

    # ========== Report Results ==========
    logger.info("\n" + "=" * 70)
    logger.info("COMPOSITIONAL GENERALIZATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Method':<20} {'Seen MSE':>10} {'Unseen MSE':>12} {'HM':>10} {'Comp Gap':>10}")
    logger.info("-" * 70)

    for name, r in results.items():
        s, u = r["seen"], r["unseen"]
        # Harmonic mean (lower MSE is better, so invert for HM)
        # HM of accuracies: use 1/MSE as "accuracy"
        if s > 0 and u > 0:
            hm = 2 * (1/s) * (1/u) / (1/s + 1/u)  # HM of inverse-MSE
            hm = 1 / hm  # back to MSE scale
        else:
            hm = 0
        gap = u - s  # positive = worse on unseen
        gap_pct = 100 * gap / max(s, 1e-8)

        logger.info(f"{name:<20} {s:>10.6f} {u:>12.6f} {hm:>10.6f} {gap_pct:>+9.1f}%")
        r["hm"] = hm
        r["gap_pct"] = gap_pct

    logger.info("=" * 70)
    logger.info("Lower MSE = better. Comp Gap = (Unseen-Seen)/Seen × 100%")
    logger.info("Good compositional generalization → small positive gap")

    # Save results
    torch.save(results, exp_dir / "results.pt")
    logger.info(f"\nResults saved to {exp_dir}/results.pt")


if __name__ == "__main__":
    main()
