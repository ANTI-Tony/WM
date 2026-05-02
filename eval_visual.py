"""
Visual input experiments: end-to-end from rendered frames.

Env 1: SimplePhysics-Visual (our synthetic env, pixel input)
Env 2: BouncingBalls-Visual (classic benchmark, pixel input)

Both use CNN SlotEncoder → CausalComp pipeline (no GT states).
Tests whether compositional gap trend holds with learned representations.

Usage:
    python eval_visual.py --synthetic --num_epochs 200 --num_videos 3000
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
from models.causalcomp import CausalComp
from utils.logger import setup_logger


# ========== Bouncing Balls Dataset (Visual) ==========

class BouncingBallsVisual(torch.utils.data.Dataset):
    """Classic bouncing balls benchmark — from pixels.
    3-5 colored balls bouncing in a box. Standard in object-centric literature.
    Used by: NRI, SlotFormer, SAVi, STEVE, etc.
    """

    def __init__(self, num_videos=3000, num_frames=16, resolution=64,
                 num_balls_range=(3, 5), seed=42):
        self.num_videos = num_videos
        self.num_frames = num_frames
        self.resolution = resolution
        self.num_balls_range = num_balls_range

        rng = random.Random(seed)
        self.seeds = [rng.randint(0, 2**31) for _ in range(num_videos)]

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        rng = random.Random(self.seeds[idx])
        R = self.resolution
        n_balls = rng.randint(*self.num_balls_range)

        colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1)]
        color_names = ["red", "green", "blue", "yellow", "cyan"]
        radii = [4, 6, 8]
        radius_names = ["small", "medium", "large"]

        balls = []
        for i in range(n_balls):
            r = rng.choice(radii)
            balls.append({
                "x": rng.uniform(r+1, R-r-1),
                "y": rng.uniform(r+1, R-r-1),
                "vx": rng.uniform(-2.5, 2.5),
                "vy": rng.uniform(-2.5, 2.5),
                "radius": r,
                "color": colors[i % len(colors)],
                "color_name": color_names[i % len(color_names)],
                "radius_name": radius_names[radii.index(r)],
            })

        frames = []
        events = []
        max_obj = self.num_balls_range[1]
        collision_adj = torch.zeros(self.num_frames, max_obj, max_obj)

        for t in range(self.num_frames):
            # Render
            frame = torch.zeros(3, R, R)
            yy, xx = torch.meshgrid(
                torch.arange(R, dtype=torch.float32),
                torch.arange(R, dtype=torch.float32), indexing="ij")
            for b in balls:
                dist = torch.sqrt((xx - b["x"])**2 + (yy - b["y"])**2)
                mask = (dist < b["radius"]).float()
                for c in range(3):
                    frame[c] = torch.maximum(frame[c], mask * b["color"][c])
            frames.append(frame)

            # Physics
            for b in balls:
                b["x"] += b["vx"]
                b["y"] += b["vy"]

            # Walls
            for b in balls:
                r = b["radius"]
                if b["x"] - r < 0: b["x"] = r; b["vx"] = abs(b["vx"])
                if b["x"] + r > R: b["x"] = R - r; b["vx"] = -abs(b["vx"])
                if b["y"] - r < 0: b["y"] = r; b["vy"] = abs(b["vy"])
                if b["y"] + r > R: b["y"] = R - r; b["vy"] = -abs(b["vy"])

            # Collisions
            import math
            for i in range(n_balls):
                for j in range(i+1, n_balls):
                    bi, bj = balls[i], balls[j]
                    dx = bi["x"] - bj["x"]
                    dy = bi["y"] - bj["y"]
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < bi["radius"] + bj["radius"] and dist > 0:
                        events.append({"type": "collision", "frame": t, "objects": [i, j]})
                        if i < max_obj and j < max_obj:
                            collision_adj[t, i, j] = 1.0
                            collision_adj[t, j, i] = 1.0
                        nx, ny = dx/dist, dy/dist
                        dvn = (bi["vx"]-bj["vx"])*nx + (bi["vy"]-bj["vy"])*ny
                        if dvn < 0:
                            bi["vx"] -= dvn*nx; bi["vy"] -= dvn*ny
                            bj["vx"] += dvn*nx; bj["vy"] += dvn*ny
                        overlap = bi["radius"]+bj["radius"]-dist
                        bi["x"] += overlap*0.5*nx; bi["y"] += overlap*0.5*ny
                        bj["x"] -= overlap*0.5*nx; bj["y"] -= overlap*0.5*ny

        video = torch.stack(frames)
        props = [{"color": b["color_name"], "shape": "ball", "material": b["radius_name"]}
                 for b in balls]

        return {
            "video": video,
            "video_id": f"bounce_{idx:05d}",
            "objects": {"num_objects": n_balls, "properties": props},
            "events": events,
            "collision_adj": collision_adj,
            "positions": torch.zeros(self.num_frames, max_obj, 2),  # not used for visual
            "num_objects": n_balls,
        }


def bounce_collate_fn(batch):
    videos = torch.stack([b["video"] for b in batch])
    result = {
        "video": videos,
        "video_ids": [b["video_id"] for b in batch],
        "objects": [b["objects"] for b in batch],
        "events": [b["events"] for b in batch],
        "collision_adj": torch.stack([b["collision_adj"] for b in batch]),
        "num_objects": [b["num_objects"] for b in batch],
    }
    return result


# ========== Visual CausalComp variants ==========

class VisualCausalComp(nn.Module):
    """End-to-end: frames → SlotEncoder → GraphDiscovery → Dynamics → Decoder."""

    def __init__(self, resolution=64, num_slots=6, slot_dim=64,
                 num_interaction_types=8):
        super().__init__()
        from models.slot_attention import SlotEncoder
        from models.causal_graph import CausalGraphDiscovery
        from models.modular_dynamics import ModularCausalDynamics
        from models.decoder import SpatialBroadcastDecoder

        self.encoder = SlotEncoder(resolution=resolution, num_slots=num_slots,
                                    slot_dim=slot_dim, encoder_channels=64)
        self.graph = CausalGraphDiscovery(slot_dim=slot_dim,
                                           num_interaction_types=num_interaction_types,
                                           hidden_dim=slot_dim)
        self.dynamics = ModularCausalDynamics(slot_dim=slot_dim,
                                              num_interaction_types=num_interaction_types,
                                              hidden_dim=slot_dim)
        self.decoder = SpatialBroadcastDecoder(slot_dim=slot_dim, resolution=resolution)

    def forward(self, video, rollout_steps=5):
        B, T, C, H, W = video.shape
        all_slots, all_attn = self.encoder(video)  # [B, T, K, D]

        # Autoregressive from frame 0
        current = all_slots[:, 0]
        pred_frames = []
        graph_infos = []
        recon_t0, masks_t0, _ = self.decoder(current)

        for t in range(min(T-1, rollout_steps)):
            ep, et, gi = self.graph(current)
            graph_infos.append(gi)
            current = self.dynamics(current, ep, et)
            frame, _, _ = self.decoder(current)
            pred_frames.append(frame)

        pred_frames = torch.stack(pred_frames, 1)
        return {
            "recon_t0": recon_t0,
            "pred_frames": pred_frames,
            "target_frames": video[:, 1:rollout_steps+1],
            "frame_t0": video[:, 0],
            "graph_infos": graph_infos,
            "all_slots": all_slots,
            "attn_maps": all_attn,
            "masks_t0": masks_t0,
        }


class VisualNoGraph(nn.Module):
    """Visual baseline: SlotEncoder → per-slot MLP dynamics → Decoder."""

    def __init__(self, resolution=64, num_slots=6, slot_dim=64):
        super().__init__()
        from models.slot_attention import SlotEncoder
        from models.decoder import SpatialBroadcastDecoder

        self.encoder = SlotEncoder(resolution=resolution, num_slots=num_slots,
                                    slot_dim=slot_dim, encoder_channels=64)
        self.dyn = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(),
                                  nn.Linear(slot_dim, slot_dim), nn.ReLU(),
                                  nn.Linear(slot_dim, slot_dim))
        self.decoder = SpatialBroadcastDecoder(slot_dim=slot_dim, resolution=resolution)

    def forward(self, video, rollout_steps=5):
        B, T, C, H, W = video.shape
        all_slots, all_attn = self.encoder(video)
        current = all_slots[:, 0]
        pred_frames = []
        recon_t0, masks_t0, _ = self.decoder(current)

        for t in range(min(T-1, rollout_steps)):
            current = self.dyn(current)
            frame, _, _ = self.decoder(current)
            pred_frames.append(frame)

        pred_frames = torch.stack(pred_frames, 1)
        return {
            "recon_t0": recon_t0,
            "pred_frames": pred_frames,
            "target_frames": video[:, 1:rollout_steps+1],
            "frame_t0": video[:, 0],
            "graph_infos": [],
            "all_slots": all_slots,
            "attn_maps": all_attn,
            "masks_t0": masks_t0,
        }


class VisualSingleModule(nn.Module):
    """Visual: SlotEncoder → graph (1 module) → Dynamics → Decoder."""

    def __init__(self, resolution=64, num_slots=6, slot_dim=64):
        super().__init__()
        from models.slot_attention import SlotEncoder
        from models.causal_graph import CausalGraphDiscovery
        from models.decoder import SpatialBroadcastDecoder

        self.encoder = SlotEncoder(resolution=resolution, num_slots=num_slots,
                                    slot_dim=slot_dim, encoder_channels=64)
        self.graph = CausalGraphDiscovery(slot_dim=slot_dim, num_interaction_types=1,
                                           hidden_dim=slot_dim)
        self.f_self = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim),
                                     nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.f_inter = nn.Sequential(nn.Linear(slot_dim*2, slot_dim), nn.ReLU(),
                                      nn.Linear(slot_dim, slot_dim))
        self.update = nn.Sequential(nn.LayerNorm(slot_dim*2), nn.Linear(slot_dim*2, slot_dim),
                                     nn.ReLU(), nn.Linear(slot_dim, slot_dim))
        self.decoder = SpatialBroadcastDecoder(slot_dim=slot_dim, resolution=resolution)

    def forward(self, video, rollout_steps=5):
        B, T, C, H, W = video.shape
        all_slots, all_attn = self.encoder(video)
        current = all_slots[:, 0]
        pred_frames = []
        graph_infos = []
        recon_t0, _, _ = self.decoder(current)

        K = current.shape[1]
        for t in range(min(T-1, rollout_steps)):
            ep, et, gi = self.graph(current)
            graph_infos.append(gi)
            ds = self.f_self(current)
            si = current.unsqueeze(2).expand(B,K,K,-1)
            sj = current.unsqueeze(1).expand(B,K,K,-1)
            di = (self.f_inter(torch.cat([si,sj],-1)) * ep.unsqueeze(-1)).sum(1)
            current = self.update(torch.cat([ds,di],-1))
            frame, _, _ = self.decoder(current)
            pred_frames.append(frame)

        pred_frames = torch.stack(pred_frames, 1)
        return {
            "recon_t0": recon_t0,
            "pred_frames": pred_frames,
            "target_frames": video[:, 1:rollout_steps+1],
            "frame_t0": video[:, 0],
            "graph_infos": graph_infos,
            "all_slots": all_slots,
            "attn_maps": all_attn,
        }


# ========== Training and Evaluation ==========

def visual_loss(output):
    losses = {}
    losses["recon"] = F.mse_loss(output["recon_t0"], output["frame_t0"])
    pred = output["pred_frames"]
    target = output["target_frames"]
    T = min(pred.shape[1], target.shape[1])
    if T > 0:
        fl = [(1+t) * F.mse_loss(pred[:,t], target[:,t]) for t in range(T)]
        losses["dynamics"] = torch.stack(fl).mean()
    else:
        losses["dynamics"] = torch.tensor(0.0, device=pred.device)
    losses["total"] = losses["recon"] + 2.0 * losses["dynamics"]
    return losses


@torch.no_grad()
def eval_visual_mse(model, loader, device, rollout_steps=5):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        video = batch["video"].to(device)
        out = model(video, rollout_steps=rollout_steps)
        p, t = out["pred_frames"], out["target_frames"]
        T = min(p.shape[1], t.shape[1])
        total += F.mse_loss(p[:,:T], t[:,:T], reduction="sum").item()
        count += p[:,:T].numel()
    return total / max(count, 1)


def train_visual(model, loader, num_epochs, lr, device, rollout_steps=5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    for epoch in range(num_epochs):
        model.train()
        for batch in loader:
            video = batch["video"].to(device)
            out = model(video, rollout_steps=rollout_steps)
            losses = visual_loss(out)
            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_videos", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rollout_steps", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42,123,777")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path("experiments/visual")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("visual", exp_dir / "eval.log")

    envs = {
        "SimplePhysics-Visual": {
            "cls": SyntheticPhysicsDataset,
            "collate": synthetic_collate_fn,
            "split": "compositional",
        },
        "BouncingBalls-Visual": {
            "cls": BouncingBallsVisual,
            "collate": bounce_collate_fn,
            "split": "compositional",
        },
    }

    methods = {
        "NoGraph-V": lambda: VisualNoGraph(resolution=64, num_slots=6, slot_dim=64),
        "SingleModule-V": lambda: VisualSingleModule(resolution=64, num_slots=6, slot_dim=64),
        "CausalComp-V": lambda: VisualCausalComp(resolution=64, num_slots=6, slot_dim=64, num_interaction_types=8),
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
            logger.info(f"Seed {seed}: Train={len(tr)} Seen={len(se)} Unseen={len(un)}")

            tl = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True,
                           num_workers=0, collate_fn=ecfg["collate"], drop_last=True)
            sl = DataLoader(Subset(ds, se), batch_size=args.batch_size, shuffle=False,
                           num_workers=0, collate_fn=ecfg["collate"])
            ul = DataLoader(Subset(ds, un), batch_size=args.batch_size, shuffle=False,
                           num_workers=0, collate_fn=ecfg["collate"])

            for mname, make_model in methods.items():
                torch.manual_seed(seed)
                model = make_model().to(device)
                params = sum(p.numel() for p in model.parameters())
                logger.info(f"  Training {mname} ({params:,} params)...")
                train_visual(model, tl, args.num_epochs, args.lr, device, args.rollout_steps)
                s = eval_visual_mse(model, sl, device, args.rollout_steps)
                u = eval_visual_mse(model, ul, device, args.rollout_steps)
                g = 100*(u-s)/max(s, 1e-8)
                env_results[mname]["seen"].append(s)
                env_results[mname]["unseen"].append(u)
                env_results[mname]["gap"].append(g)
                logger.info(f"    {mname}: seen={s:.6f} unseen={u:.6f} gap={g:+.1f}%")

        all_results[env_name] = env_results

        logger.info(f"\n{env_name} RESULTS (mean ± std):")
        for m in methods:
            r = env_results[m]
            if r["seen"]:
                logger.info(f"  {m}: seen={np.mean(r['seen']):.4f}±{np.std(r['seen']):.4f} "
                           f"unseen={np.mean(r['unseen']):.4f}±{np.std(r['unseen']):.4f} "
                           f"gap={np.mean(r['gap']):+.1f}±{np.std(r['gap']):.1f}%")

    torch.save(all_results, exp_dir / "visual_results.pt")
    logger.info(f"\nSaved to {exp_dir}/visual_results.pt")


if __name__ == "__main__":
    main()
