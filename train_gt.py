"""
Train graph discovery + dynamics using GT object states (no encoder/decoder).
This validates the core idea: can modular causal dynamics discover meaningful
interaction graphs and improve compositional generalization?

Usage:
    python train_gt.py --exp_name gt_v1 --num_epochs 100 --synthetic --num_videos 3000
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import Config
from data.synthetic_dataset import SyntheticPhysicsDataset, synthetic_collate_fn
from models.causal_graph import CausalGraphDiscovery
from models.modular_dynamics import ModularCausalDynamics
from utils.logger import setup_logger


GT_STATE_DIM = 8  # [x, y, vx, vy, r, g, b, radius]


class GTCausalComp(nn.Module):
    """CausalComp operating on GT object states instead of learned slots.

    No encoder/decoder needed. Just graph discovery + dynamics.
    Input: GT state vectors per object per frame.
    Output: predicted next-step state vectors.
    """

    def __init__(self, state_dim: int = GT_STATE_DIM, slot_dim: int = 64,
                 num_interaction_types: int = 4, max_objects: int = 6):
        super().__init__()
        self.max_objects = max_objects
        self.slot_dim = slot_dim

        # Project GT state to slot space
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        # Graph discovery
        self.graph_discovery = CausalGraphDiscovery(
            slot_dim=slot_dim,
            num_interaction_types=num_interaction_types,
            hidden_dim=slot_dim,
            gumbel_temperature=0.5,
        )

        # Modular dynamics
        self.dynamics = ModularCausalDynamics(
            slot_dim=slot_dim,
            num_interaction_types=num_interaction_types,
            hidden_dim=slot_dim,
            num_message_passing=2,
        )

        # Decode back to state space (predict position/velocity changes)
        self.state_decoder = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, state_dim),
        )

    def forward(self, gt_states: torch.Tensor, rollout_steps: int = 5):
        """
        Args:
            gt_states: [B, T, max_obj, state_dim] GT object states
            rollout_steps: steps to predict autoregressively
        Returns: dict with predictions and graph info
        """
        B, T, K, D = gt_states.shape

        # Encode all frames
        all_slots = self.state_encoder(gt_states)  # [B, T, K, slot_dim]

        # Autoregressive prediction from frame 0
        current_slots = all_slots[:, 0]  # [B, K, slot_dim]
        pred_states_list = []
        graph_info_list = []

        for t in range(min(T - 1, rollout_steps)):
            # Discover graph
            edge_probs, edge_types, graph_info = self.graph_discovery(current_slots)
            graph_info_list.append(graph_info)

            # Predict next slots
            next_slots = self.dynamics(current_slots, edge_probs, edge_types)

            # Decode to state space
            pred_state = self.state_decoder(next_slots)
            pred_states_list.append(pred_state)

            # Autoregressive: re-encode predicted state for next step
            current_slots = self.state_encoder(pred_state)

        pred_states = torch.stack(pred_states_list, dim=1)  # [B, T', K, D]
        target_states = gt_states[:, 1:rollout_steps+1]      # [B, T', K, D]

        return {
            "pred_states": pred_states,
            "target_states": target_states,
            "graph_infos": graph_info_list,
        }


def compute_loss(output, collision_adj=None):
    """Compute losses for GT mode."""
    losses = {}
    dev = output["pred_states"].device

    # State prediction loss (main signal)
    pred = output["pred_states"]
    target = output["target_states"]
    T = min(pred.shape[1], target.shape[1])
    if T > 0:
        step_losses = []
        for t in range(T):
            weight = 1.0 + t  # later steps weighted more
            step_losses.append(weight * F.mse_loss(pred[:, t], target[:, t]))
        losses["state_pred"] = torch.stack(step_losses).mean()
    else:
        losses["state_pred"] = torch.tensor(0.0, device=dev)

    # Edge supervision (collision adjacency, NO matching needed in GT mode!)
    losses["edge_sup"] = torch.tensor(0.0, device=dev)
    if collision_adj is not None and output["graph_infos"]:
        edge_probs = output["graph_infos"][0]["edge_probs"]  # [B, K, K]
        B, K, _ = edge_probs.shape
        max_obj = collision_adj.shape[-1]

        # In GT mode, slot i = object i (no permutation!)
        col_any = collision_adj.any(dim=1).float().to(dev)  # [B, max_obj, max_obj]
        if max_obj >= K:
            col_target = col_any[:, :K, :K]
        else:
            col_target = F.pad(col_any, (0, K - max_obj, 0, K - max_obj))

        losses["edge_sup"] = F.binary_cross_entropy(edge_probs, col_target)

    # Graph losses
    for graph_info in output["graph_infos"]:
        gl = CausalGraphDiscovery.compute_loss(None, graph_info)
        for k, v in gl.items():
            if k not in losses:
                losses[k] = v
            else:
                losses[k] = losses[k] + v

    n_steps = max(len(output["graph_infos"]), 1)
    for k in ["sparsity", "type_entropy", "min_connect"]:
        if k in losses:
            losses[k] = losses[k] / n_steps

    # Total
    losses["total"] = (
        5.0 * losses["state_pred"] +
        2.0 * losses["edge_sup"] +      # tuned: 2.0 works better than 10.0
        0.001 * losses.get("sparsity", torch.tensor(0.0)) +
        0.01 * losses.get("type_entropy", torch.tensor(0.0))
    )

    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="gt_v1")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_videos", type=int, default=3000)
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--num_interaction_types", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_dir = Path("experiments") / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    logger = setup_logger(args.exp_name, exp_dir / "train.log")
    logger.info(f"GT Mode Training | device={device}")

    # Data
    dataset = SyntheticPhysicsDataset(
        num_videos=args.num_videos, num_frames=16, resolution=64, seed=args.seed,
    )
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=synthetic_collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, collate_fn=synthetic_collate_fn)

    logger.info(f"Train: {n_train}, Val: {n_val}")

    # Model
    model = GTCausalComp(
        state_dim=GT_STATE_DIM, slot_dim=64,
        num_interaction_types=args.num_interaction_types, max_objects=6,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    best_val = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_losses = {}
        n_batches = 0

        for batch in train_loader:
            gt_states = batch["gt_states"].to(device)
            collision_adj = batch["collision_adj"].to(device)

            output = model(gt_states, rollout_steps=args.rollout_steps)
            losses = compute_loss(output, collision_adj=collision_adj)

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v.item()
            n_batches += 1

        scheduler.step()
        avg = {k: v / n_batches for k, v in epoch_losses.items()}

        # Evaluate
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_losses = {}
            n_val_b = 0
            with torch.no_grad():
                for batch in val_loader:
                    gt_states = batch["gt_states"].to(device)
                    collision_adj = batch["collision_adj"].to(device)
                    output = model(gt_states, rollout_steps=args.rollout_steps)
                    losses = compute_loss(output, collision_adj=collision_adj)
                    for k, v in losses.items():
                        val_losses[k] = val_losses.get(k, 0) + v.item()
                    n_val_b += 1

                # Check edge differentiation
                ep = output["graph_infos"][0]["edge_probs"]
                edge_std = ep.std().item()
                edge_mean = ep.mean().item()
                edges_high = (ep > 0.5).float().sum(dim=(-1, -2)).mean().item()
                edges_low = (ep < 0.1).float().sum(dim=(-1, -2)).mean().item()

            val_avg = {k: v / max(n_val_b, 1) for k, v in val_losses.items()}

            logger.info(
                f"Epoch {epoch}/{args.num_epochs} | "
                f"train_loss={avg['total']:.4f} state={avg['state_pred']:.4f} "
                f"edgeSup={avg['edge_sup']:.4f} | "
                f"val_loss={val_avg['total']:.4f} | "
                f"edges: mean={edge_mean:.3f} std={edge_std:.3f} "
                f"high(>0.5)={edges_high:.1f} low(<0.1)={edges_low:.1f}"
            )

            if val_avg["total"] < best_val:
                best_val = val_avg["total"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val,
                }, exp_dir / "checkpoints" / "best.pt")
                logger.info(f"  -> Best model saved")
        else:
            logger.info(
                f"Epoch {epoch}/{args.num_epochs} | "
                f"train_loss={avg['total']:.4f} state={avg['state_pred']:.4f} "
                f"edgeSup={avg['edge_sup']:.4f}"
            )

    logger.info(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
