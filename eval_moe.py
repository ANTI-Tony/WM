"""
MoE baseline evaluation on SimplePhysics + MultiPhysics.

Two MoE variants:
1. MoE (per-object routing): router based on single object features
2. MoE-Pair (pairwise routing): router based on pair features (closest to CausalComp)

Key comparison:
- CausalComp has explicit edge prediction (sparse graph) + typed modules
- MoE-Pair has typed modules but NO sparse graph (all pairs interact)
- If MoE-Pair gap ≈ CausalComp gap → graph structure doesn't matter, only typing
- If MoE-Pair gap > CausalComp gap → graph structure matters beyond typing

Usage:
    python eval_moe.py --synthetic
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
from models.moe_baseline import MoEDynamics, MoEPairwise
from train_gt import GTCausalComp, compute_loss
from eval_full import SingleModuleModel, eval_mse, simple_loss
from eval_neurips import train_model, split_multi_physics
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--num_videos", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seeds", type=str, default="42,123,777")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path("experiments/moe")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("moe", exp_dir / "eval.log")

    envs = {
        "SimplePhysics": {
            "cls": SyntheticPhysicsDataset,
            "collate": synthetic_collate_fn,
            "state_dim": 8,
            "split": "comp",
        },
        "MultiPhysics": {
            "cls": MultiPhysicsDataset,
            "collate": multi_physics_collate_fn,
            "state_dim": 10,
            "split": "charge",
        },
    }

    methods = {
        "SingleModule": lambda sd: SingleModuleModel(state_dim=sd, slot_dim=128),
        "MoE": lambda sd: MoEDynamics(state_dim=sd, slot_dim=128, num_experts=8),
        "MoE-Pair": lambda sd: MoEPairwise(state_dim=sd, slot_dim=128, num_experts=8),
        "CausalComp": lambda sd: GTCausalComp(state_dim=sd, slot_dim=128, num_interaction_types=8),
    }

    all_results = {}

    for env_name, ecfg in envs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"ENVIRONMENT: {env_name}")
        logger.info(f"{'='*60}")

        env_results = {m: {"seen": [], "unseen": [], "gap": []} for m in methods}

        for seed in seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

            ds = ecfg["cls"](num_videos=args.num_videos, num_frames=16, resolution=64, seed=seed)

            if ecfg["split"] == "comp":
                split_info = create_compositional_split(seed=seed)
                tr, un = [], []
                for i in range(len(ds)):
                    s = ds[i]
                    if classify_video(s["objects"]["properties"], s["events"], split_info) == "train":
                        tr.append(i)
                    else:
                        un.append(i)
                rng = random.Random(seed); rng.shuffle(tr)
                n = max(1, len(tr)//10); se = tr[-n:]; tr = tr[:-n]
            else:
                tr, se, un = split_multi_physics(ds, seed)

            logger.info(f"Seed {seed}: Train={len(tr)} Seen={len(se)} Unseen={len(un)}")

            tl = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True,
                           num_workers=0, collate_fn=ecfg["collate"], drop_last=True)
            sl = DataLoader(Subset(ds, se), batch_size=min(args.batch_size, len(se)),
                           shuffle=False, num_workers=0, collate_fn=ecfg["collate"])
            ul = DataLoader(Subset(ds, un), batch_size=min(args.batch_size, len(un)),
                           shuffle=False, num_workers=0, collate_fn=ecfg["collate"])

            sd = ecfg["state_dim"]
            for mname, make_model in methods.items():
                torch.manual_seed(seed)
                model = make_model(sd).to(device)
                params = sum(p.numel() for p in model.parameters())
                gl = (mname == "CausalComp")
                logger.info(f"  Training {mname} ({params:,} params)...")
                train_model(model, tl, args.num_epochs, args.lr, device,
                           use_graph_loss=gl, rollout_steps=8)
                s = eval_mse(model, sl, device, 8)
                u = eval_mse(model, ul, device, 8)
                g = 100*(u-s)/max(s, 1e-8)
                env_results[mname]["seen"].append(s)
                env_results[mname]["unseen"].append(u)
                env_results[mname]["gap"].append(g)
                logger.info(f"    {mname}: seen={s:.6f} unseen={u:.6f} gap={g:+.1f}%")

        all_results[env_name] = env_results

        logger.info(f"\n{env_name} RESULTS (mean ± std):")
        logger.info(f"{'Method':<16} {'Seen':>14} {'Unseen':>14} {'Gap':>14}")
        logger.info("-" * 60)
        for m in methods:
            r = env_results[m]
            if r["seen"]:
                logger.info(f"{m:<16} {np.mean(r['seen']):.4f}±{np.std(r['seen']):.4f} "
                           f"{np.mean(r['unseen']):.4f}±{np.std(r['unseen']):.4f} "
                           f"{np.mean(r['gap']):+.1f}±{np.std(r['gap']):.1f}%")

    torch.save(all_results, exp_dir / "moe_results.pt")
    logger.info(f"\nSaved to {exp_dir}/moe_results.pt")


if __name__ == "__main__":
    main()
