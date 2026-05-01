"""
Supplementary experiments:
1. Capacity-matched ablation (addresses "implicit regularization" concern)
2. Per-rollout-step error analysis
3. Additional seeds (3→5)

Usage:
    python eval_supplementary.py --synthetic
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
from eval_full import NoGraphModel, SingleModuleModel, eval_mse, simple_loss
from eval_neurips import SlotFormerBaseline, split_multi_physics, train_model
from utils.logger import setup_logger


# ========== Experiment 1: Capacity-matched ablation ==========

class CausalCompBig(nn.Module):
    """CausalComp with MORE parameters (wider MLPs) to match SlotFormer capacity.
    If CausalComp-Big still has lower gap than SingleModule-Big,
    it proves typed modules help beyond just limiting capacity."""

    def __init__(self, state_dim=8, slot_dim=256, num_interaction_types=8):
        super().__init__()
        from models.causal_graph import CausalGraphDiscovery
        from models.modular_dynamics import ModularCausalDynamics

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.graph_discovery = CausalGraphDiscovery(
            slot_dim=slot_dim, num_interaction_types=num_interaction_types, hidden_dim=slot_dim)
        self.dynamics = ModularCausalDynamics(
            slot_dim=slot_dim, num_interaction_types=num_interaction_types,
            hidden_dim=slot_dim, num_message_passing=2)
        self.state_decoder = nn.Sequential(
            nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, state_dim))

    def forward(self, gt_states, rollout_steps=8):
        B, T, K, D = gt_states.shape
        slots = self.state_encoder(gt_states[:, 0])
        preds, ginfos = [], []
        for _ in range(min(T-1, rollout_steps)):
            ep, et, gi = self.graph_discovery(slots)
            ginfos.append(gi)
            slots = self.dynamics(slots, ep, et)
            preds.append(self.state_decoder(slots))
            slots = self.state_encoder(preds[-1])
        return {"pred_states": torch.stack(preds, 1),
                "target_states": gt_states[:, 1:rollout_steps+1],
                "graph_infos": ginfos}


class SingleModuleBig(nn.Module):
    """SingleModule with same total capacity as CausalComp-Big."""

    def __init__(self, state_dim=8, slot_dim=256):
        super().__init__()
        from models.causal_graph import CausalGraphDiscovery
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
            di = (self.f_inter(torch.cat([si,sj],-1)) * ep.unsqueeze(-1)).sum(1)
            h = self.update(torch.cat([ds,di],-1))
            preds.append(self.dec(h))
            h = self.enc(preds[-1])
        return {"pred_states": torch.stack(preds,1), "target_states": gt_states[:,1:rollout_steps+1], "graph_infos": ginfos}


# ========== Experiment 2: Per-rollout-step error ==========

@torch.no_grad()
def eval_per_step(model, loader, device, max_steps=12):
    """Compute MSE at each rollout step separately."""
    model.eval()
    step_errors = [0.0] * max_steps
    step_counts = [0] * max_steps

    for batch in loader:
        gt = batch["gt_states"].to(device)
        out = model(gt, rollout_steps=max_steps)
        pred, target = out["pred_states"], out["target_states"]
        T = min(pred.shape[1], target.shape[1], max_steps)
        for t in range(T):
            mse = F.mse_loss(pred[:, t], target[:, t], reduction="sum").item()
            step_errors[t] += mse
            step_counts[t] += pred[:, t].numel()

    return [e / max(c, 1) for e, c in zip(step_errors, step_counts)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--num_videos", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path("experiments/supplementary")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("supp", exp_dir / "eval.log")

    # ===== Exp 1: Capacity-matched =====
    logger.info("=" * 60)
    logger.info("EXP 1: Capacity-matched ablation (slot_dim=256)")
    logger.info("=" * 60)

    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

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
    n = max(1, len(tr)//10); se = tr[-n:]; tr = tr[:-n]

    tl = DataLoader(Subset(ds,tr), batch_size=args.batch_size, shuffle=True,
                    num_workers=0, collate_fn=synthetic_collate_fn, drop_last=True)
    sl = DataLoader(Subset(ds,se), batch_size=args.batch_size, shuffle=False,
                    num_workers=0, collate_fn=synthetic_collate_fn)
    ul = DataLoader(Subset(ds,un), batch_size=args.batch_size, shuffle=False,
                    num_workers=0, collate_fn=synthetic_collate_fn)

    for name, ModelCls, gl in [
        ("SingleModule-Big", SingleModuleBig, False),
        ("CausalComp-Big", CausalCompBig, True),
    ]:
        torch.manual_seed(seed)
        m = ModelCls(state_dim=8, slot_dim=256).to(device)
        n_params = sum(p.numel() for p in m.parameters())
        logger.info(f"  {name}: {n_params:,} parameters")
        train_model(m, tl, args.num_epochs, 3e-4, device, use_graph_loss=gl, rollout_steps=8)
        s_mse = eval_mse(m, sl, device, 8)
        u_mse = eval_mse(m, ul, device, 8)
        gap = 100 * (u_mse - s_mse) / max(s_mse, 1e-8)
        logger.info(f"  {name}: seen={s_mse:.6f} unseen={u_mse:.6f} gap={gap:+.1f}%")

    # Also run standard size for comparison
    for name, ModelCls, gl in [
        ("SingleModule-Std", SingleModuleModel, False),
        ("CausalComp-Std", GTCausalComp, True),
    ]:
        torch.manual_seed(seed)
        if gl:
            m = ModelCls(state_dim=8, slot_dim=128, num_interaction_types=8).to(device)
        else:
            m = ModelCls(state_dim=8, slot_dim=128).to(device)
        n_params = sum(p.numel() for p in m.parameters())
        logger.info(f"  {name}: {n_params:,} parameters")
        train_model(m, tl, args.num_epochs, 3e-4, device, use_graph_loss=gl, rollout_steps=8)
        s_mse = eval_mse(m, sl, device, 8)
        u_mse = eval_mse(m, ul, device, 8)
        gap = 100 * (u_mse - s_mse) / max(s_mse, 1e-8)
        logger.info(f"  {name}: seen={s_mse:.6f} unseen={u_mse:.6f} gap={gap:+.1f}%")

    # ===== Exp 2: Per-rollout-step error =====
    logger.info("\n" + "=" * 60)
    logger.info("EXP 2: Per-rollout-step error (unseen combinations)")
    logger.info("=" * 60)

    models_for_rollout = {}
    for name, ModelCls, gl, sd in [
        ("NoGraph", NoGraphModel, False, 128),
        ("SlotFormer", SlotFormerBaseline, False, 128),
        ("SingleModule", SingleModuleModel, False, 128),
        ("CausalComp", GTCausalComp, True, 128),
    ]:
        torch.manual_seed(seed)
        if name == "CausalComp":
            m = ModelCls(state_dim=8, slot_dim=sd, num_interaction_types=8).to(device)
        else:
            m = ModelCls(state_dim=8, slot_dim=sd).to(device)
        train_model(m, tl, args.num_epochs, 3e-4, device, use_graph_loss=gl, rollout_steps=8)
        models_for_rollout[name] = m

    logger.info(f"{'Step':>4} | {'NoGraph':>10} {'SlotFormer':>12} {'SingleMod':>12} {'CausalComp':>12}")
    logger.info("-" * 60)
    all_step_errors = {}
    for name, model in models_for_rollout.items():
        steps = eval_per_step(model, ul, device, max_steps=12)
        all_step_errors[name] = steps
        for t, e in enumerate(steps):
            if t == 0 or (t+1) % 2 == 0:
                pass  # will print below

    for t in range(12):
        vals = {n: all_step_errors[n][t] for n in models_for_rollout}
        logger.info(f"  t={t+1:>2} | {vals['NoGraph']:.6f}  {vals['SlotFormer']:.6f}  "
                    f"{vals['SingleModule']:.6f}  {vals['CausalComp']:.6f}")

    torch.save({"capacity": {}, "rollout": all_step_errors}, exp_dir / "supp_results.pt")
    logger.info(f"\nSaved to {exp_dir}/supp_results.pt")


if __name__ == "__main__":
    main()
