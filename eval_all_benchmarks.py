"""
Full benchmark evaluation: 4 environments × 5 methods × 3 seeds.

Environments:
  1. SimplePhysics (ours): elastic collisions, 18 object types
  2. MultiPhysics (ours): collisions + gravity + charge, 54 object types
  3. NRI Springs (Kipf 2018): spring-connected particles, standard benchmark
  4. N-Body Charged (Battaglia 2016): charged particle dynamics

Methods:
  NoGraph, FullGraph, SingleModule, SlotFormer, CausalComp

Usage:
    python eval_all_benchmarks.py --num_epochs 150 --synthetic
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
from data.nri_springs import NRISpringsDataset, springs_collate_fn, NBodyChargedDataset, nbody_collate_fn
from data.compositional_split import create_compositional_split, classify_video
from train_gt import GTCausalComp, compute_loss
from eval_full import NoGraphModel, FullGraphModel, SingleModuleModel, eval_mse, simple_loss
from eval_neurips import SlotFormerBaseline, split_multi_physics, train_model
from utils.logger import setup_logger


def split_springs(dataset, seed=42):
    """Split NRI Springs: hold out videos where specific spring-type combos appear."""
    rng = random.Random(seed)
    train_idx, unseen_idx = [], []
    for i in range(len(dataset)):
        s = dataset.data[i]
        edges = s["edges"]
        # Hold out: videos with spring type 3 (strongest spring)
        if 3 in edges:
            unseen_idx.append(i)
        else:
            train_idx.append(i)
    rng.shuffle(train_idx)
    n_seen = max(1, len(train_idx) // 10)
    seen_idx = train_idx[-n_seen:]
    train_idx = train_idx[:-n_seen]
    return train_idx, seen_idx, unseen_idx


def split_nbody(dataset, seed=42):
    """Split N-Body: hold out videos where all particles have same charge sign."""
    rng = random.Random(seed)
    train_idx, unseen_idx = [], []
    for i in range(len(dataset)):
        charges = dataset.data[i]["charges"]
        if all(c > 0 for c in charges) or all(c < 0 for c in charges):
            unseen_idx.append(i)
        else:
            train_idx.append(i)
    rng.shuffle(train_idx)
    n_seen = max(1, len(train_idx) // 10)
    seen_idx = train_idx[-n_seen:]
    train_idx = train_idx[:-n_seen]
    return train_idx, seen_idx, unseen_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--num_videos", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--seeds", type=str, default="42,123,777")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path("experiments/all_benchmarks")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("all_bench", exp_dir / "eval.log")

    # Environment configs
    envs = {
        "SimplePhysics": {"state_dim": 8, "num_particles": 6},
        "MultiPhysics":  {"state_dim": 10, "num_particles": 5},
        "NRI-Springs":   {"state_dim": 4, "num_particles": 5},
        "N-Body":        {"state_dim": 5, "num_particles": 5},
    }

    method_names = ["NoGraph", "FullGraph", "SingleModule", "SlotFormer", "CausalComp"]
    all_results = {}

    for env_name, ecfg in envs.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"ENVIRONMENT: {env_name}")
        logger.info(f"{'='*70}")

        sd = ecfg["state_dim"]
        env_results = {m: {"seen": [], "unseen": [], "gap": []} for m in method_names}

        for seed in seeds:
            logger.info(f"\n--- Seed {seed} ---")
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

            # Create dataset + split
            if env_name == "SimplePhysics":
                dataset = SyntheticPhysicsDataset(num_videos=args.num_videos, num_frames=16, resolution=64, seed=seed)
                collate = synthetic_collate_fn
                split_info = create_compositional_split(seed=seed)
                tr, se, un = [], [], []
                for i in range(len(dataset)):
                    s = dataset[i]
                    if classify_video(s["objects"]["properties"], s["events"], split_info) == "train":
                        tr.append(i)
                    else:
                        un.append(i)
                rng = random.Random(seed); rng.shuffle(tr)
                ns = max(1, len(tr)//10); se = tr[-ns:]; tr = tr[:-ns]
            elif env_name == "MultiPhysics":
                dataset = MultiPhysicsDataset(num_videos=args.num_videos, num_frames=16, resolution=64, seed=seed)
                collate = multi_physics_collate_fn
                tr, se, un = split_multi_physics(dataset, seed)
            elif env_name == "NRI-Springs":
                dataset = NRISpringsDataset(num_videos=args.num_videos, num_frames=49, num_particles=5, seed=seed)
                collate = springs_collate_fn
                tr, se, un = split_springs(dataset, seed)
            elif env_name == "N-Body":
                dataset = NBodyChargedDataset(num_videos=args.num_videos, num_frames=16, num_particles=5, seed=seed)
                collate = nbody_collate_fn
                tr, se, un = split_nbody(dataset, seed)

            logger.info(f"  Train={len(tr)} Seen={len(se)} Unseen={len(un)}")

            if len(tr) < args.batch_size or len(un) < 1:
                logger.info(f"  SKIP: insufficient data")
                continue

            train_ld = DataLoader(Subset(dataset, tr), batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate, drop_last=True)
            seen_ld = DataLoader(Subset(dataset, se), batch_size=min(args.batch_size, len(se)),
                                 shuffle=False, num_workers=0, collate_fn=collate)
            unseen_ld = DataLoader(Subset(dataset, un), batch_size=min(args.batch_size, len(un)),
                                   shuffle=False, num_workers=0, collate_fn=collate)

            for mname in method_names:
                torch.manual_seed(seed)
                if mname == "CausalComp":
                    model = GTCausalComp(state_dim=sd, slot_dim=128, num_interaction_types=8).to(device)
                    use_gl = True
                elif mname == "SlotFormer":
                    model = SlotFormerBaseline(state_dim=sd, slot_dim=128).to(device)
                    use_gl = False
                elif mname == "SingleModule":
                    model = SingleModuleModel(state_dim=sd, slot_dim=128).to(device)
                    use_gl = False
                elif mname == "FullGraph":
                    model = FullGraphModel(state_dim=sd, slot_dim=128).to(device)
                    use_gl = False
                else:
                    model = NoGraphModel(state_dim=sd, slot_dim=128).to(device)
                    use_gl = False

                logger.info(f"  Training {mname}...")
                train_model(model, train_ld, args.num_epochs, args.lr, device,
                           use_graph_loss=use_gl, rollout_steps=args.rollout_steps)

                s_mse = eval_mse(model, seen_ld, device, args.rollout_steps)
                u_mse = eval_mse(model, unseen_ld, device, args.rollout_steps)
                gap = 100*(u_mse - s_mse)/max(s_mse, 1e-8)

                env_results[mname]["seen"].append(s_mse)
                env_results[mname]["unseen"].append(u_mse)
                env_results[mname]["gap"].append(gap)
                logger.info(f"    {mname}: seen={s_mse:.6f} unseen={u_mse:.6f} gap={gap:+.1f}%")

        all_results[env_name] = env_results

        logger.info(f"\n{'='*80}")
        logger.info(f"{env_name} RESULTS (mean ± std over {len(seeds)} seeds)")
        logger.info(f"{'='*80}")
        logger.info(f"{'Method':<16} {'Seen MSE':>14} {'Unseen MSE':>14} {'Comp Gap':>14}")
        logger.info(f"{'-'*80}")
        for m in method_names:
            r = env_results[m]
            if r["seen"]:
                sm, ss = np.mean(r["seen"]), np.std(r["seen"])
                um, us = np.mean(r["unseen"]), np.std(r["unseen"])
                gm, gs = np.mean(r["gap"]), np.std(r["gap"])
                logger.info(f"{m:<16} {sm:.4f}±{ss:.4f} {um:.4f}±{us:.4f} {gm:+.1f}±{gs:.1f}%")
        logger.info(f"{'='*80}")

    torch.save(all_results, exp_dir / "all_results.pt")
    logger.info(f"\nSaved to {exp_dir}/all_results.pt")


if __name__ == "__main__":
    main()
