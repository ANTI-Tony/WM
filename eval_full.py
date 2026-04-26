"""
Full evaluation: multi-seed + SingleModule ablation.

Runs 3 seeds × 4 methods = 12 training runs.
Reports mean ± std for all metrics.

Usage:
    python eval_full.py --num_epochs 150 --num_videos 5000 --synthetic
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
from models.causal_graph import CausalGraphDiscovery
from models.modular_dynamics import ModularCausalDynamics
from utils.logger import setup_logger


# ============ Baselines ============

class NoGraphModel(nn.Module):
    """No interaction graph, self-dynamics only."""
    def __init__(self, state_dim=8, slot_dim=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(state_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.dyn = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.dec = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, state_dim))
    def forward(self, gt_states, rollout_steps=8):
        B, T, K, D = gt_states.shape
        h = self.enc(gt_states[:, 0])
        preds = []
        for _ in range(min(T-1, rollout_steps)):
            h = self.dyn(h)
            preds.append(self.dec(h))
            h = self.enc(preds[-1])
        return {"pred_states": torch.stack(preds, 1), "target_states": gt_states[:, 1:rollout_steps+1], "graph_infos": []}


class FullGraphModel(nn.Module):
    """All edges on, equal weight, no discovery."""
    def __init__(self, state_dim=8, slot_dim=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(state_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.f_self = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.f_inter = nn.Sequential(nn.Linear(slot_dim*2, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.update = nn.Sequential(nn.Linear(slot_dim*2, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.dec = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, state_dim))
    def forward(self, gt_states, rollout_steps=8):
        B, T, K, D = gt_states.shape
        h = self.enc(gt_states[:, 0])
        preds = []
        for _ in range(min(T-1, rollout_steps)):
            ds = self.f_self(h)
            si = h.unsqueeze(2).expand(B,K,K,-1)
            sj = h.unsqueeze(1).expand(B,K,K,-1)
            di = self.f_inter(torch.cat([si, sj], -1)).mean(1)
            h = self.update(torch.cat([ds, di], -1))
            preds.append(self.dec(h))
            h = self.enc(preds[-1])
        return {"pred_states": torch.stack(preds, 1), "target_states": gt_states[:, 1:rollout_steps+1], "graph_infos": []}


class SingleModuleModel(nn.Module):
    """Graph discovery YES, but only ONE shared dynamics module (no type specialization).
    Ablation: proves that typed modules matter, not just the graph."""
    def __init__(self, state_dim=8, slot_dim=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(state_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.graph = CausalGraphDiscovery(slot_dim=slot_dim, num_interaction_types=1, hidden_dim=slot_dim)
        self.f_self = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.f_inter = nn.Sequential(nn.Linear(slot_dim*2, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.update = nn.Sequential(nn.LayerNorm(slot_dim*2), nn.Linear(slot_dim*2, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.dec = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, state_dim))
    def forward(self, gt_states, rollout_steps=8):
        B, T, K, D = gt_states.shape
        h = self.enc(gt_states[:, 0])
        preds, ginfos = [], []
        for _ in range(min(T-1, rollout_steps)):
            ep, et, gi = self.graph(h)
            ginfos.append(gi)
            ds = self.f_self(h)
            si = h.unsqueeze(2).expand(B,K,K,-1)
            sj = h.unsqueeze(1).expand(B,K,K,-1)
            effects = self.f_inter(torch.cat([si, sj], -1))
            di = (effects * ep.unsqueeze(-1)).sum(1)
            h = self.update(torch.cat([ds, di], -1))
            preds.append(self.dec(h))
            h = self.enc(preds[-1])
        return {"pred_states": torch.stack(preds, 1), "target_states": gt_states[:, 1:rollout_steps+1], "graph_infos": ginfos}


# ============ Training helpers ============

def simple_loss(output):
    pred, target = output["pred_states"], output["target_states"]
    T = min(pred.shape[1], target.shape[1])
    losses = [(1.0+t) * F.mse_loss(pred[:,t], target[:,t]) for t in range(T)]
    return {"total": 5.0 * torch.stack(losses).mean()}


def train_one(model, train_loader, config, device, use_graph_loss=False):
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.num_epochs)
    for epoch in range(config.num_epochs):
        model.train()
        for batch in train_loader:
            gt = batch["gt_states"].to(device)
            col = batch["collision_adj"].to(device)
            out = model(gt, rollout_steps=config.rollout_steps)
            if use_graph_loss:
                losses = compute_loss(out, collision_adj=col)
            else:
                losses = simple_loss(out)
            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()


@torch.no_grad()
def eval_mse(model, loader, device, rollout_steps=8):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        gt = batch["gt_states"].to(device)
        out = model(gt, rollout_steps=rollout_steps)
        p, t = out["pred_states"], out["target_states"]
        T = min(p.shape[1], t.shape[1])
        total += F.mse_loss(p[:,:T], t[:,:T], reduction="sum").item()
        count += p[:,:T].numel()
    return total / max(count, 1)


# ============ Main ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--num_videos", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--seeds", type=str, default="42,123,777")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path("experiments/full_eval")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("full_eval", exp_dir / "eval.log")

    methods = {
        "CausalComp":   lambda: GTCausalComp(state_dim=8, slot_dim=128, num_interaction_types=8),
        "SingleModule": lambda: SingleModuleModel(state_dim=8, slot_dim=128),
        "FullGraph":    lambda: FullGraphModel(state_dim=8, slot_dim=128),
        "NoGraph":      lambda: NoGraphModel(state_dim=8, slot_dim=128),
    }
    uses_graph_loss = {"CausalComp": True, "SingleModule": False, "FullGraph": False, "NoGraph": False}

    all_results = {m: {"seen": [], "unseen": [], "gap": []} for m in methods}

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED = {seed}")
        logger.info(f"{'='*60}")

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Data
        split_info = create_compositional_split(seed=seed)
        dataset = SyntheticPhysicsDataset(num_videos=args.num_videos, num_frames=16, resolution=64, seed=seed)

        train_idx, seen_idx, unseen_idx = [], [], []
        for i in range(len(dataset)):
            s = dataset[i]
            label = classify_video(s["objects"]["properties"], s["events"], split_info)
            if label == "train":
                train_idx.append(i)
            else:
                unseen_idx.append(i)
        rng = random.Random(seed)
        rng.shuffle(train_idx)
        n_seen = max(1, len(train_idx) // 10)
        seen_idx = train_idx[-n_seen:]
        train_idx = train_idx[:-n_seen]

        logger.info(f"  Train={len(train_idx)} Seen={len(seen_idx)} Unseen={len(unseen_idx)}")

        train_ld = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=synthetic_collate_fn, drop_last=True)
        seen_ld = DataLoader(Subset(dataset, seen_idx), batch_size=args.batch_size, shuffle=False,
                             num_workers=2, collate_fn=synthetic_collate_fn)
        unseen_ld = DataLoader(Subset(dataset, unseen_idx), batch_size=args.batch_size, shuffle=False,
                               num_workers=2, collate_fn=synthetic_collate_fn)

        for name, make_model in methods.items():
            logger.info(f"  Training {name}...")
            torch.manual_seed(seed)  # same init per seed
            model = make_model().to(device)
            train_one(model, train_ld, args, device, use_graph_loss=uses_graph_loss[name])
            s_mse = eval_mse(model, seen_ld, device, args.rollout_steps)
            u_mse = eval_mse(model, unseen_ld, device, args.rollout_steps)
            gap = 100 * (u_mse - s_mse) / max(s_mse, 1e-8)
            all_results[name]["seen"].append(s_mse)
            all_results[name]["unseen"].append(u_mse)
            all_results[name]["gap"].append(gap)
            logger.info(f"    {name}: seen={s_mse:.6f} unseen={u_mse:.6f} gap={gap:+.1f}%")

    # ============ Final Table ============
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL RESULTS (mean ± std over {len(seeds)} seeds)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Method':<16} {'Seen MSE':>14} {'Unseen MSE':>14} {'Comp Gap':>14}")
    logger.info(f"{'-'*80}")

    for name in methods:
        r = all_results[name]
        s_m, s_s = np.mean(r["seen"]), np.std(r["seen"])
        u_m, u_s = np.mean(r["unseen"]), np.std(r["unseen"])
        g_m, g_s = np.mean(r["gap"]), np.std(r["gap"])
        logger.info(f"{name:<16} {s_m:.4f}±{s_s:.4f} {u_m:.4f}±{u_s:.4f} {g_m:+.1f}±{g_s:.1f}%")

    logger.info(f"{'='*80}")
    logger.info("Lower MSE = better. Lower Comp Gap = better compositional generalization.")

    torch.save(all_results, exp_dir / "results_full.pt")
    logger.info(f"Saved to {exp_dir}/results_full.pt")


if __name__ == "__main__":
    main()
