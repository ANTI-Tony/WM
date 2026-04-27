"""
NeurIPS-ready evaluation: 2 environments × 5 methods × 3 seeds.

Environments:
  1. SimplePhysics: elastic collisions only (6 colors × 3 sizes = 18 types)
  2. MultiPhysics: collisions + gravity + charge (6 colors × 3 masses × 3 charges = 54 types)

Methods:
  1. NoGraph: self-dynamics only
  2. FullGraph: all edges equally weighted
  3. SingleModule: graph + 1 shared dynamics module
  4. SlotFormer-style: autoregressive transformer over slots (no graph structure)
  5. CausalComp: graph + M typed dynamics modules (ours)

Usage:
    python eval_neurips.py --synthetic --num_epochs 150 --num_videos 5000
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
from data.multi_physics_dataset import MultiPhysicsDataset, multi_physics_collate_fn
from data.compositional_split import create_compositional_split, classify_video
from train_gt import GTCausalComp, compute_loss
from eval_full import NoGraphModel, FullGraphModel, SingleModuleModel, eval_mse, simple_loss
from models.causal_graph import CausalGraphDiscovery
from utils.logger import setup_logger


# ============ SlotFormer-style baseline ============

class SlotFormerBaseline(nn.Module):
    """Autoregressive Transformer over object slots.
    Mimics SlotFormer's approach: no explicit graph structure,
    Transformer attention implicitly handles interactions.
    """
    def __init__(self, state_dim=8, slot_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(state_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        # Transformer processes all object slots jointly
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=slot_dim, nhead=num_heads, dim_feedforward=slot_dim*2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dec = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, state_dim))

    def forward(self, gt_states, rollout_steps=8):
        B, T, K, D = gt_states.shape
        h = self.enc(gt_states[:, 0])  # [B, K, slot_dim]
        preds = []
        for _ in range(min(T-1, rollout_steps)):
            h = self.transformer(h)  # [B, K, slot_dim] — attention handles interactions
            preds.append(self.dec(h))
            h = self.enc(preds[-1])
        return {
            "pred_states": torch.stack(preds, 1),
            "target_states": gt_states[:, 1:rollout_steps+1],
            "graph_infos": [],
        }


# ============ Multi-physics compositional split ============

def split_multi_physics(dataset, seed=42, train_frac=0.6):
    """Split multi-physics dataset by (color, mass, charge) collision pairs."""
    rng = random.Random(seed)

    # Classify each video
    train_idx, unseen_idx = [], []

    # Simple split: use charge-based held-out
    # Hold out: all collisions between same-charge objects
    # This is a natural compositional split: same physics, new object combinations
    for i in range(len(dataset)):
        s = dataset[i]
        has_unseen = False
        for ev in s["events"]:
            if ev["type"] == "collision":
                oi, oj = ev["objects"]
                props = s["objects"]["properties"]
                if oi < len(props) and oj < len(props):
                    ci = props[oi].get("charge", "neutral")
                    cj = props[oj].get("charge", "neutral")
                    # Hold out: positive-positive and negative-negative collisions
                    if (ci == "pos" and cj == "pos") or (ci == "neg" and cj == "neg"):
                        has_unseen = True
                        break
        if has_unseen:
            unseen_idx.append(i)
        else:
            train_idx.append(i)

    rng.shuffle(train_idx)
    n_seen = max(1, len(train_idx) // 10)
    seen_idx = train_idx[-n_seen:]
    train_idx = train_idx[:-n_seen]

    return train_idx, seen_idx, unseen_idx


# ============ Training ============

def train_model(model, train_loader, num_epochs, lr, device, use_graph_loss=False, rollout_steps=8):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            gt = batch["gt_states"].to(device)
            out = model(gt, rollout_steps=rollout_steps)
            if use_graph_loss:
                col = batch["collision_adj"].to(device)
                losses = compute_loss(out, collision_adj=col)
            else:
                losses = simple_loss(out)
            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model


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
    exp_dir = Path("experiments/neurips_eval")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("neurips_eval", exp_dir / "eval.log")

    environments = {
        "SimplePhysics": {
            "dataset_cls": SyntheticPhysicsDataset,
            "collate": synthetic_collate_fn,
            "state_dim": 8,
            "split_fn": "compositional",
        },
        "MultiPhysics": {
            "dataset_cls": MultiPhysicsDataset,
            "collate": multi_physics_collate_fn,
            "state_dim": 10,
            "split_fn": "charge",
        },
    }

    methods = {
        "NoGraph":       {"cls": NoGraphModel, "graph_loss": False},
        "FullGraph":     {"cls": FullGraphModel, "graph_loss": False},
        "SingleModule":  {"cls": SingleModuleModel, "graph_loss": False},
        "SlotFormer":    {"cls": SlotFormerBaseline, "graph_loss": False},
        "CausalComp":    {"cls": GTCausalComp, "graph_loss": True},
    }

    all_results = {}

    for env_name, env_cfg in environments.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"ENVIRONMENT: {env_name}")
        logger.info(f"{'='*70}")

        env_results = {m: {"seen": [], "unseen": [], "gap": []} for m in methods}

        for seed in seeds:
            logger.info(f"\n--- Seed {seed} ---")
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Create dataset
            dataset = env_cfg["dataset_cls"](
                num_videos=args.num_videos, num_frames=16, resolution=64, seed=seed
            )

            # Split
            if env_cfg["split_fn"] == "compositional":
                split_info = create_compositional_split(seed=seed)
                train_idx, seen_idx, unseen_idx = [], [], []
                for i in range(len(dataset)):
                    s = dataset[i]
                    label = classify_video(s["objects"]["properties"], s["events"], split_info)
                    if label == "train":
                        train_idx.append(i)
                    else:
                        unseen_idx.append(i)
                rng = random.Random(seed); rng.shuffle(train_idx)
                n_seen = max(1, len(train_idx) // 10)
                seen_idx = train_idx[-n_seen:]
                train_idx = train_idx[:-n_seen]
            else:
                train_idx, seen_idx, unseen_idx = split_multi_physics(dataset, seed)

            logger.info(f"  Train={len(train_idx)} Seen={len(seen_idx)} Unseen={len(unseen_idx)}")

            collate = env_cfg["collate"]
            train_ld = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=collate, drop_last=True)
            seen_ld = DataLoader(Subset(dataset, seen_idx), batch_size=args.batch_size,
                                 shuffle=False, num_workers=2, collate_fn=collate)
            unseen_ld = DataLoader(Subset(dataset, unseen_idx), batch_size=args.batch_size,
                                   shuffle=False, num_workers=2, collate_fn=collate)

            sd = env_cfg["state_dim"]

            for mname, mcfg in methods.items():
                logger.info(f"  Training {mname}...")
                torch.manual_seed(seed)

                if mname == "CausalComp":
                    model = GTCausalComp(state_dim=sd, slot_dim=128, num_interaction_types=8).to(device)
                elif mname == "SlotFormer":
                    model = SlotFormerBaseline(state_dim=sd, slot_dim=128).to(device)
                elif mname == "SingleModule":
                    # Need to handle different state_dim
                    model = SingleModuleModel(state_dim=sd, slot_dim=128).to(device)
                else:
                    model = mcfg["cls"](state_dim=sd, slot_dim=128).to(device)

                train_model(model, train_ld, args.num_epochs, args.lr, device,
                           use_graph_loss=mcfg["graph_loss"], rollout_steps=args.rollout_steps)

                s_mse = eval_mse(model, seen_ld, device, args.rollout_steps)
                u_mse = eval_mse(model, unseen_ld, device, args.rollout_steps)
                gap = 100 * (u_mse - s_mse) / max(s_mse, 1e-8)

                env_results[mname]["seen"].append(s_mse)
                env_results[mname]["unseen"].append(u_mse)
                env_results[mname]["gap"].append(gap)
                logger.info(f"    {mname}: seen={s_mse:.6f} unseen={u_mse:.6f} gap={gap:+.1f}%")

        all_results[env_name] = env_results

        # Print environment summary
        logger.info(f"\n{'='*80}")
        logger.info(f"{env_name} RESULTS (mean ± std over {len(seeds)} seeds)")
        logger.info(f"{'='*80}")
        logger.info(f"{'Method':<16} {'Seen MSE':>14} {'Unseen MSE':>14} {'Comp Gap':>14}")
        logger.info(f"{'-'*80}")
        for mname in methods:
            r = env_results[mname]
            s_m, s_s = np.mean(r["seen"]), np.std(r["seen"])
            u_m, u_s = np.mean(r["unseen"]), np.std(r["unseen"])
            g_m, g_s = np.mean(r["gap"]), np.std(r["gap"])
            logger.info(f"{mname:<16} {s_m:.4f}±{s_s:.4f} {u_m:.4f}±{u_s:.4f} {g_m:+.1f}±{g_s:.1f}%")
        logger.info(f"{'='*80}")

    # Save all results
    torch.save(all_results, exp_dir / "neurips_results.pt")
    logger.info(f"\nAll results saved to {exp_dir}/neurips_results.pt")


if __name__ == "__main__":
    main()
