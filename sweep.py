"""
Hyperparameter sweep to maximize CausalComp's compositional advantage.

Key hypotheses to test:
1. edge_sup weight too high (10.0) → constrains dynamics learning
2. More interaction types → finer-grained modules → better generalization
3. Larger slot_dim → more capacity for dynamics
4. More data → better generalization

Usage:
    python sweep.py --synthetic
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.synthetic_dataset import SyntheticPhysicsDataset, synthetic_collate_fn
from data.compositional_split import create_compositional_split, classify_video
from train_gt import GTCausalComp, GT_STATE_DIM, compute_loss
from eval_full import NoGraphModel, FullGraphModel, eval_mse, simple_loss
from utils.logger import setup_logger


def make_compute_loss_fn(edge_sup_weight):
    """Create a compute_loss with custom edge_sup weight."""
    def fn(output, collision_adj=None):
        losses = {}
        dev = output["pred_states"].device
        pred, target = output["pred_states"], output["target_states"]
        T = min(pred.shape[1], target.shape[1])

        step_losses = [(1.0 + t) * F.mse_loss(pred[:, t], target[:, t]) for t in range(T)]
        losses["state_pred"] = torch.stack(step_losses).mean()

        losses["edge_sup"] = torch.tensor(0.0, device=dev)
        if collision_adj is not None and output["graph_infos"]:
            ep = output["graph_infos"][0]["edge_probs"]
            B, K, _ = ep.shape
            max_obj = collision_adj.shape[-1]
            col_any = collision_adj.any(dim=1).float().to(dev)
            if max_obj >= K:
                ct = col_any[:, :K, :K]
            else:
                ct = F.pad(col_any, (0, K - max_obj, 0, K - max_obj))
            losses["edge_sup"] = F.binary_cross_entropy(ep, ct)

        from models.causal_graph import CausalGraphDiscovery
        for gi in output["graph_infos"]:
            gl = CausalGraphDiscovery.compute_loss(None, gi)
            for k2, v2 in gl.items():
                losses[k2] = losses.get(k2, torch.tensor(0.0, device=dev)) + v2

        losses["total"] = (
            5.0 * losses["state_pred"] +
            edge_sup_weight * losses["edge_sup"] +
            0.001 * losses.get("sparsity", torch.tensor(0.0, device=dev))
        )
        return losses
    return fn


def train_and_eval(model, train_ld, seen_ld, unseen_ld, loss_fn, device, num_epochs, lr, rollout_steps):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_ld:
            gt = batch["gt_states"].to(device)
            col = batch["collision_adj"].to(device)
            out = model(gt, rollout_steps=rollout_steps)
            losses = loss_fn(out, collision_adj=col)
            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    s = eval_mse(model, seen_ld, device, rollout_steps)
    u = eval_mse(model, unseen_ld, device, rollout_steps)
    gap = 100 * (u - s) / max(s, 1e-8)
    return s, u, gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--num_videos", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    exp_dir = Path("experiments/sweep")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("sweep", exp_dir / "sweep.log")

    # Data
    split_info = create_compositional_split(seed=args.seed)
    dataset = SyntheticPhysicsDataset(num_videos=args.num_videos, num_frames=16, resolution=64, seed=args.seed)

    train_idx, seen_idx, unseen_idx = [], [], []
    for i in range(len(dataset)):
        s = dataset[i]
        label = classify_video(s["objects"]["properties"], s["events"], split_info)
        if label == "train":
            train_idx.append(i)
        else:
            unseen_idx.append(i)
    rng = random.Random(args.seed); rng.shuffle(train_idx)
    n_seen = max(1, len(train_idx) // 10)
    seen_idx = train_idx[-n_seen:]
    train_idx = train_idx[:-n_seen]

    train_ld = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
                          num_workers=2, collate_fn=synthetic_collate_fn, drop_last=True)
    seen_ld = DataLoader(Subset(dataset, seen_idx), batch_size=args.batch_size, shuffle=False,
                         num_workers=2, collate_fn=synthetic_collate_fn)
    unseen_ld = DataLoader(Subset(dataset, unseen_idx), batch_size=args.batch_size, shuffle=False,
                           num_workers=2, collate_fn=synthetic_collate_fn)

    logger.info(f"Train={len(train_idx)} Seen={len(seen_idx)} Unseen={len(unseen_idx)}")

    # NoGraph baseline (run once)
    logger.info("=== Baseline: NoGraph ===")
    torch.manual_seed(args.seed)
    ng = NoGraphModel(state_dim=8, slot_dim=64).to(device)
    s, u, g = train_and_eval(ng, train_ld, seen_ld, unseen_ld, lambda out, **kw: simple_loss(out),
                              device, args.num_epochs, 3e-4, 8)
    logger.info(f"  NoGraph: seen={s:.4f} unseen={u:.4f} gap={g:+.1f}%")
    baseline_unseen = u

    # Sweep configs
    configs = [
        # (name, num_types, slot_dim, edge_sup_weight, lr)
        ("T4_S64_E10",  4, 64,  10.0, 3e-4),   # original
        ("T4_S64_E2",   4, 64,  2.0,  3e-4),    # lower edge_sup
        ("T4_S64_E0.5", 4, 64,  0.5,  3e-4),    # much lower edge_sup
        ("T8_S64_E2",   8, 64,  2.0,  3e-4),    # more types
        ("T4_S128_E2",  4, 128, 2.0,  3e-4),    # larger slots
        ("T8_S128_E2",  8, 128, 2.0,  3e-4),    # more types + larger
        ("T4_S64_E5",   4, 64,  5.0,  3e-4),    # medium edge_sup
        ("T2_S64_E2",   2, 64,  2.0,  3e-4),    # fewer types
    ]

    results = []
    for name, nt, sd, esw, lr in configs:
        logger.info(f"\n=== {name}: types={nt} slot_dim={sd} edge_sup={esw} ===")
        torch.manual_seed(args.seed)
        model = GTCausalComp(state_dim=8, slot_dim=sd, num_interaction_types=nt).to(device)
        loss_fn = make_compute_loss_fn(esw)
        s, u, g = train_and_eval(model, train_ld, seen_ld, unseen_ld, loss_fn,
                                  device, args.num_epochs, lr, 8)
        improvement = 100 * (baseline_unseen - u) / baseline_unseen
        logger.info(f"  seen={s:.4f} unseen={u:.4f} gap={g:+.1f}% (vs NoGraph: {improvement:+.1f}%)")
        results.append({"name": name, "seen": s, "unseen": u, "gap": g, "vs_baseline": improvement})

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"SWEEP RESULTS (sorted by unseen MSE)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Config':<18} {'Seen':>8} {'Unseen':>8} {'Gap':>8} {'vs NoGraph':>10}")
    logger.info(f"{'-'*80}")

    results.sort(key=lambda r: r["unseen"])
    for r in results:
        logger.info(f"{r['name']:<18} {r['seen']:.4f}   {r['unseen']:.4f}   {r['gap']:+.1f}%   {r['vs_baseline']:+.1f}%")

    logger.info(f"\nNoGraph baseline:  unseen={baseline_unseen:.4f}")
    logger.info(f"{'='*80}")

    torch.save(results, exp_dir / "sweep_results.pt")


if __name__ == "__main__":
    main()
