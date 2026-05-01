"""
Mechanistic evidence experiments:

Exp 7: M overfitting curve (M=2,4,8,16,32) — does gap U-shape?
Exp 8: Representation transferability — do typed modules activate
       similarly on seen vs unseen pairs? (CKA / cosine similarity)

These two experiments transform the paper from "observation" to
"mechanistic understanding."

Usage:
    python eval_mechanistic.py --synthetic
"""

import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.synthetic_dataset import SyntheticPhysicsDataset, synthetic_collate_fn
from data.compositional_split import create_compositional_split, classify_video
from train_gt import GTCausalComp, compute_loss
from eval_neurips import train_model
from eval_full import eval_mse
from utils.logger import setup_logger


# ========== Exp 7: M overfitting curve ==========

def exp7_m_curve(train_loader, seen_loader, unseen_loader, device, logger, num_epochs=150):
    """Test M ∈ {2, 4, 8, 16, 32} to find U-shape in comp gap."""
    logger.info("=" * 60)
    logger.info("EXP 7: Number of interaction types M")
    logger.info("=" * 60)

    m_values = [1, 2, 4, 8, 16, 32]
    results = []

    for M in m_values:
        torch.manual_seed(42)
        model = GTCausalComp(state_dim=8, slot_dim=128, num_interaction_types=M).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  M={M}: {n_params:,} params, training...")

        train_model(model, train_loader, num_epochs, 3e-4, device,
                   use_graph_loss=True, rollout_steps=8)

        s = eval_mse(model, seen_loader, device, 8)
        u = eval_mse(model, unseen_loader, device, 8)
        g = 100 * (u - s) / max(s, 1e-8)
        results.append({"M": M, "seen": s, "unseen": u, "gap": g, "params": n_params})
        logger.info(f"    M={M}: seen={s:.6f} unseen={u:.6f} gap={g:+.1f}% params={n_params:,}")

    logger.info("\n  Summary:")
    logger.info(f"  {'M':>4} {'Seen':>10} {'Unseen':>10} {'Gap':>8} {'Params':>10}")
    for r in results:
        logger.info(f"  {r['M']:>4} {r['seen']:.6f} {r['unseen']:.6f} {r['gap']:+.1f}% {r['params']:>10,}")

    return results


# ========== Exp 8: Representation transferability ==========

def exp8_transferability(model, seen_loader, unseen_loader, device, logger):
    """Analyze whether typed modules activate similarly on seen vs unseen pairs.

    For each interaction type τ:
    1. Collect f_inter^τ activations on SEEN collision pairs
    2. Collect f_inter^τ activations on UNSEEN collision pairs
    3. Compute cosine similarity between mean activations

    If typed modules learn mechanisms (not pairs):
        → seen/unseen activations for same type should be SIMILAR (high cosine)
    If SingleModule memorizes pairs:
        → seen/unseen activations should DIFFER (low cosine)
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXP 8: Representation transferability analysis")
    logger.info("=" * 60)

    model.eval()

    def collect_activations(loader, label):
        """Collect per-type interaction module activations."""
        # Hook into each f_inter module
        type_activations = defaultdict(list)  # type_idx → list of activation tensors

        def make_hook(type_idx):
            def hook_fn(module, input, output):
                type_activations[type_idx].append(output.detach().cpu())
            return hook_fn

        hooks = []
        for tau in range(len(model.dynamics.f_inter)):
            h = model.dynamics.f_inter[tau].net[-1].register_forward_hook(make_hook(tau))
            hooks.append(h)

        with torch.no_grad():
            for batch in loader:
                gt = batch["gt_states"].to(device)
                _ = model(gt, rollout_steps=1)

        for h in hooks:
            h.remove()

        # Compute mean activation per type
        mean_acts = {}
        for tau, acts in type_activations.items():
            if acts:
                all_acts = torch.cat(acts, dim=0)
                mean_acts[tau] = all_acts.mean(dim=0)  # [D]
                logger.info(f"    {label} type {tau}: {len(acts)} batches, "
                           f"mean norm={mean_acts[tau].norm():.4f}")

        return mean_acts

    # Collect on seen and unseen
    logger.info("  Collecting activations on SEEN data...")
    seen_acts = collect_activations(seen_loader, "seen")

    logger.info("  Collecting activations on UNSEEN data...")
    unseen_acts = collect_activations(unseen_loader, "unseen")

    # Compute cosine similarity: same type, seen vs unseen
    logger.info("\n  Cosine similarity (same type, seen vs unseen):")
    logger.info("  Higher = module activates similarly regardless of object types")
    cosines_same_type = []
    for tau in sorted(set(seen_acts.keys()) & set(unseen_acts.keys())):
        cos = F.cosine_similarity(
            seen_acts[tau].unsqueeze(0),
            unseen_acts[tau].unsqueeze(0)
        ).item()
        cosines_same_type.append(cos)
        logger.info(f"    Type {tau}: cosine = {cos:.4f}")

    if cosines_same_type:
        logger.info(f"  Mean same-type cosine: {np.mean(cosines_same_type):.4f}")

    # Cross-type similarity (should be lower)
    logger.info("\n  Cosine similarity (different types, within seen):")
    cosines_cross_type = []
    types = sorted(seen_acts.keys())
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            cos = F.cosine_similarity(
                seen_acts[types[i]].unsqueeze(0),
                seen_acts[types[j]].unsqueeze(0)
            ).item()
            cosines_cross_type.append(cos)

    if cosines_cross_type:
        logger.info(f"  Mean cross-type cosine: {np.mean(cosines_cross_type):.4f}")

    logger.info(f"\n  KEY RESULT:")
    if cosines_same_type and cosines_cross_type:
        same = np.mean(cosines_same_type)
        cross = np.mean(cosines_cross_type)
        logger.info(f"  Same-type (seen vs unseen): {same:.4f}")
        logger.info(f"  Cross-type (within seen):   {cross:.4f}")
        if same > cross:
            logger.info(f"  → Typed modules generalize: same-type activations are MORE similar")
            logger.info(f"    across seen/unseen than different types within seen ({same:.3f} > {cross:.3f})")
        else:
            logger.info(f"  → WARNING: cross-type similarity is higher, modules may not be specializing")

    return {"same_type_cosines": cosines_same_type, "cross_type_cosines": cosines_cross_type}


def exp8_singlemodule_comparison(seen_loader, unseen_loader, device, logger):
    """Same analysis for SingleModule — expect LOWER seen/unseen similarity."""
    from eval_full import SingleModuleModel

    logger.info("\n  --- SingleModule comparison ---")
    torch.manual_seed(42)
    sm = SingleModuleModel(state_dim=8, slot_dim=128).to(device)
    tl_for_train = seen_loader  # reuse seen as proxy
    train_model(sm, seen_loader, 100, 3e-4, device, use_graph_loss=False, rollout_steps=8)

    sm.eval()

    # Collect activations from the single interaction module
    seen_acts_list, unseen_acts_list = [], []

    def hook_fn_seen(module, input, output):
        seen_acts_list.append(output.detach().cpu())

    def hook_fn_unseen(module, input, output):
        unseen_acts_list.append(output.detach().cpu())

    # Seen
    h = sm.f_inter[-1].register_forward_hook(hook_fn_seen)
    with torch.no_grad():
        for batch in seen_loader:
            _ = sm(batch["gt_states"].to(device), rollout_steps=1)
    h.remove()

    # Unseen
    h = sm.f_inter[-1].register_forward_hook(hook_fn_unseen)
    with torch.no_grad():
        for batch in unseen_loader:
            _ = sm(batch["gt_states"].to(device), rollout_steps=1)
    h.remove()

    if seen_acts_list and unseen_acts_list:
        seen_mean = torch.cat(seen_acts_list).mean(0)
        unseen_mean = torch.cat(unseen_acts_list).mean(0)
        cos = F.cosine_similarity(seen_mean.unsqueeze(0), unseen_mean.unsqueeze(0)).item()
        logger.info(f"  SingleModule seen/unseen cosine: {cos:.4f}")
        logger.info(f"  (Expected: LOWER than CausalComp's same-type cosine)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--num_videos", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path("experiments/mechanistic")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("mech", exp_dir / "eval.log")

    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Data
    ds = SyntheticPhysicsDataset(num_videos=args.num_videos, num_frames=16, resolution=64, seed=seed)
    split_info = create_compositional_split(seed=seed)
    tr, un = [], []
    for i in range(len(ds)):
        s = ds[i]
        if classify_video(s["objects"]["properties"], s["events"], split_info) == "train":
            tr.append(i)
        else:
            un.append(i)
    rng = random.Random(seed); rng.shuffle(tr)
    n = max(1, len(tr) // 10); se = tr[-n:]; tr = tr[:-n]

    tl = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True,
                    num_workers=0, collate_fn=synthetic_collate_fn, drop_last=True)
    sl = DataLoader(Subset(ds, se), batch_size=min(args.batch_size, len(se)),
                    shuffle=False, num_workers=0, collate_fn=synthetic_collate_fn)
    ul = DataLoader(Subset(ds, un), batch_size=min(args.batch_size, len(un)),
                    shuffle=False, num_workers=0, collate_fn=synthetic_collate_fn)

    logger.info(f"Train={len(tr)} Seen={len(se)} Unseen={len(un)}")

    # Exp 7: M curve
    m_results = exp7_m_curve(tl, sl, ul, device, logger, num_epochs=args.num_epochs)

    # Exp 8: Transferability — use the M=8 model from exp7
    # Train a fresh M=8 model for analysis
    logger.info("\n  Training CausalComp (M=8) for transferability analysis...")
    torch.manual_seed(seed)
    model = GTCausalComp(state_dim=8, slot_dim=128, num_interaction_types=8).to(device)
    train_model(model, tl, args.num_epochs, 3e-4, device, use_graph_loss=True, rollout_steps=8)

    transfer_results = exp8_transferability(model, sl, ul, device, logger)
    exp8_singlemodule_comparison(sl, ul, device, logger)

    # Save
    torch.save({"m_curve": m_results, "transfer": transfer_results},
               exp_dir / "mechanistic_results.pt")
    logger.info(f"\nSaved to {exp_dir}/mechanistic_results.pt")


if __name__ == "__main__":
    main()
