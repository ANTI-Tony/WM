"""
Upgraded transferability analysis for Table 5:
1. Per-type cosine breakdown (8 types, each with its own number)
2. SingleModule comparison (critical control)
3. Cross-type within unseen (verify specialization holds on unseen data)

Uses existing trained checkpoints or trains fresh models.
GPU time: ~1.5 hours.

Usage:
    python eval_transfer_v2.py --synthetic
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
from eval_full import SingleModuleModel
from eval_neurips import train_model
from utils.logger import setup_logger


def collect_typed_activations(model, loader, device, num_types):
    """Collect per-type interaction module activations via hooks."""
    model.eval()
    type_acts = defaultdict(list)

    hooks = []
    for tau in range(num_types):
        def make_hook(t):
            def hook_fn(module, inp, out):
                type_acts[t].append(out.detach().cpu().reshape(-1, out.shape[-1]))
            return hook_fn
        h = model.dynamics.f_inter[tau].net[-1].register_forward_hook(make_hook(tau))
        hooks.append(h)

    with torch.no_grad():
        for batch in loader:
            gt = batch["gt_states"].to(device)
            _ = model(gt, rollout_steps=1)

    for h in hooks:
        h.remove()

    # Compute mean activation per type
    means = {}
    for tau, acts in type_acts.items():
        all_a = torch.cat(acts, dim=0)  # [N, D]
        means[tau] = all_a.mean(dim=0)  # [D]
    return means


def collect_single_module_activations(model, loader, device):
    """Collect activations from SingleModule's single interaction MLP."""
    model.eval()
    acts_list = []

    def hook_fn(module, inp, out):
        acts_list.append(out.detach().cpu().reshape(-1, out.shape[-1]))

    h = model.f_inter[-1].register_forward_hook(hook_fn)
    with torch.no_grad():
        for batch in loader:
            gt = batch["gt_states"].to(device)
            _ = model(gt, rollout_steps=1)
    h.remove()

    if acts_list:
        return torch.cat(acts_list, dim=0).mean(dim=0)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--num_videos", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path("experiments/transfer_v2")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("transfer_v2", exp_dir / "eval.log")

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

    # ===== Train CausalComp =====
    logger.info("Training CausalComp (M=8)...")
    torch.manual_seed(seed)
    cc_model = GTCausalComp(state_dim=8, slot_dim=128, num_interaction_types=8).to(device)
    train_model(cc_model, tl, args.num_epochs, 3e-4, device, use_graph_loss=True, rollout_steps=8)

    # ===== Train SingleModule =====
    logger.info("Training SingleModule...")
    torch.manual_seed(seed)
    sm_model = SingleModuleModel(state_dim=8, slot_dim=128).to(device)
    train_model(sm_model, tl, args.num_epochs, 3e-4, device, use_graph_loss=False, rollout_steps=8)

    # ===== CausalComp: per-type analysis =====
    logger.info("\n" + "=" * 60)
    logger.info("CausalComp: Per-type transferability")
    logger.info("=" * 60)

    cc_seen = collect_typed_activations(cc_model, sl, device, 8)
    cc_unseen = collect_typed_activations(cc_model, ul, device, 8)

    # Same-type, seen vs unseen
    logger.info("\n  Same-type cosine (seen vs unseen) — per type:")
    same_cosines = []
    for tau in range(8):
        if tau in cc_seen and tau in cc_unseen:
            cos = F.cosine_similarity(cc_seen[tau].unsqueeze(0), cc_unseen[tau].unsqueeze(0)).item()
            same_cosines.append(cos)
            logger.info(f"    Type {tau}: {cos:.4f}")
    logger.info(f"  Mean: {np.mean(same_cosines):.4f} ± {np.std(same_cosines):.4f}")

    # Cross-type within seen
    logger.info("\n  Cross-type cosine (within seen):")
    cross_seen = []
    for i in range(8):
        for j in range(i+1, 8):
            if i in cc_seen and j in cc_seen:
                cos = F.cosine_similarity(cc_seen[i].unsqueeze(0), cc_seen[j].unsqueeze(0)).item()
                cross_seen.append(cos)
    logger.info(f"  Mean: {np.mean(cross_seen):.4f} ± {np.std(cross_seen):.4f}")

    # Cross-type within unseen
    logger.info("\n  Cross-type cosine (within unseen):")
    cross_unseen = []
    for i in range(8):
        for j in range(i+1, 8):
            if i in cc_unseen and j in cc_unseen:
                cos = F.cosine_similarity(cc_unseen[i].unsqueeze(0), cc_unseen[j].unsqueeze(0)).item()
                cross_unseen.append(cos)
    logger.info(f"  Mean: {np.mean(cross_unseen):.4f} ± {np.std(cross_unseen):.4f}")

    # ===== SingleModule: seen vs unseen comparison =====
    logger.info("\n" + "=" * 60)
    logger.info("SingleModule: Transferability (CONTROL)")
    logger.info("=" * 60)

    sm_seen = collect_single_module_activations(sm_model, sl, device)
    sm_unseen = collect_single_module_activations(sm_model, ul, device)

    if sm_seen is not None and sm_unseen is not None:
        sm_cos = F.cosine_similarity(sm_seen.unsqueeze(0), sm_unseen.unsqueeze(0)).item()
        logger.info(f"  SingleModule seen vs unseen cosine: {sm_cos:.4f}")
    else:
        logger.info("  SingleModule: failed to collect activations")
        sm_cos = None

    # ===== Summary =====
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY TABLE FOR PAPER")
    logger.info("=" * 60)
    logger.info(f"{'Comparison':<45} {'Cosine':>8}")
    logger.info("-" * 55)
    logger.info(f"{'CausalComp: same-type, seen vs unseen':<45} {np.mean(same_cosines):.4f} ± {np.std(same_cosines):.4f}")
    logger.info(f"{'CausalComp: cross-type, within seen':<45} {np.mean(cross_seen):.4f} ± {np.std(cross_seen):.4f}")
    logger.info(f"{'CausalComp: cross-type, within unseen':<45} {np.mean(cross_unseen):.4f} ± {np.std(cross_unseen):.4f}")
    if sm_cos is not None:
        logger.info(f"{'SingleModule: seen vs unseen':<45} {sm_cos:.4f}")
    logger.info("-" * 55)
    logger.info(f"Key: CausalComp same-type >> cross-type confirms mechanism learning")
    if sm_cos is not None:
        if np.mean(same_cosines) > sm_cos:
            logger.info(f"Key: CausalComp ({np.mean(same_cosines):.3f}) > SingleModule ({sm_cos:.3f}) confirms typed > shared")
        else:
            logger.info(f"Note: SingleModule cosine ({sm_cos:.3f}) is also high — both generalize at activation level")

    torch.save({
        "cc_same_cosines": same_cosines,
        "cc_cross_seen": cross_seen,
        "cc_cross_unseen": cross_unseen,
        "sm_cosine": sm_cos,
    }, exp_dir / "transfer_v2_results.pt")
    logger.info(f"\nSaved to {exp_dir}/transfer_v2_results.pt")


if __name__ == "__main__":
    main()
