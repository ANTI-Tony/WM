"""
Visualize model outputs for debugging.
Generates slot decomposition, causal graph, and prediction quality images.

Usage:
    python scripts/visualize_debug.py --checkpoint experiments/v2_fix/checkpoints/best.pt
    python scripts/visualize_debug.py  # no checkpoint, just test with random weights
"""

import argparse
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import Config
from data.clevrer_dataset import CLEVRERDataset, clevrer_collate_fn
from models.causalcomp import CausalComp
from utils.visualize import visualize_slots, visualize_graph, visualize_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data/clevrer")
    parser.add_argument("--output_dir", type=str, default="./figures/debug")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = ckpt.get("config", Config())
        model = CausalComp(
            resolution=config.data.resolution,
            num_slots=config.slot.num_slots,
            slot_dim=config.slot.slot_dim,
            num_interaction_types=config.causal.num_interaction_types,
            encoder_channels=config.slot.encoder_channels,
            dynamics_hidden=config.dynamics.hidden_dim,
            num_message_passing=config.dynamics.num_message_passing,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')}")
    else:
        print("No checkpoint found, using random weights (for testing pipeline)")
        config = Config()
        config.data.resolution = args.resolution
        model = CausalComp(
            resolution=args.resolution, num_slots=8, slot_dim=128,
            num_interaction_types=4, encoder_channels=64,
        ).to(device)

    model.eval()

    # Load data
    dataset = CLEVRERDataset(
        data_dir=args.data_dir,
        split="validation",
        num_frames=16,
        frame_skip=4,
        resolution=config.data.resolution,
    )

    if len(dataset) == 0:
        print("No data found! Creating dummy data for pipeline test.")
        dummy_video = torch.randn(args.num_samples, 16, 3,
                                  config.data.resolution, config.data.resolution)
        batch = {"video": dummy_video}
    else:
        # Sample a few videos
        indices = list(range(min(args.num_samples, len(dataset))))
        samples = [dataset[i] for i in indices]
        batch = clevrer_collate_fn(samples)

    video = batch["video"].to(device)  # [B, T, 3, H, W]
    B, T, C, H, W = video.shape
    print(f"Input: {B} videos, {T} frames, {H}x{W}")

    with torch.no_grad():
        output = model(video, rollout_steps=5)

    # === 1. Slot decomposition ===
    print("Generating slot decomposition...")
    visualize_slots(
        images=video[:, 0],                    # first frame
        recon=output["recon_frames"][:, 0],     # first frame recon
        masks=output["masks_t0"],
        save_path=out_dir / "slots.png",
    )
    print(f"  Saved: {out_dir / 'slots.png'}")

    # === 2. Causal graph ===
    print("Generating causal graphs...")
    for b in range(min(B, 2)):
        if output["graph_infos"]:
            gi = output["graph_infos"][0]  # first timestep
            ep = gi["edge_probs"][b].cpu()
            tl = gi["type_logits"][b].cpu()
            et = torch.nn.functional.softmax(tl, dim=-1)

            # Print edge statistics
            print(f"  Sample {b}: edge_probs stats:")
            print(f"    mean={ep.mean():.4f}, max={ep.max():.4f}, "
                  f"min={ep[ep>0].min():.4f}, >0.3: {(ep>0.3).sum().item()} edges")

            # Get object names from annotations if available
            obj_names = None
            if "objects" in batch and batch["objects"][b]["num_objects"] > 0:
                props = batch["objects"][b]["properties"]
                obj_names = [f"{p['color']} {p['shape']}" for p in props]

            visualize_graph(
                edge_probs=ep,
                edge_types=et,
                save_path=out_dir / f"graph_sample{b}.png",
                object_names=obj_names,
                threshold=0.2,
            )
            print(f"  Saved: {out_dir / f'graph_sample{b}.png'}")

    # === 3. Prediction quality ===
    print("Generating trajectory predictions...")
    context = video[:, :6]  # first 6 frames as context
    pred_slots, pred_frames = model.predict_trajectory(context, future_steps=8)

    for b in range(min(B, 2)):
        visualize_trajectory(
            gt_frames=video[b, 6:14],   # ground truth future
            pred_frames=pred_frames[b],  # predicted future
            save_path=out_dir / f"trajectory_sample{b}.png",
        )
        print(f"  Saved: {out_dir / f'trajectory_sample{b}.png'}")

    # === 4. Summary statistics ===
    print("\n=== Summary ===")
    losses = model.compute_loss(output, config.train)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")

    # Edge statistics
    if output["graph_infos"]:
        all_probs = output["graph_infos"][0]["edge_probs"]
        print(f"\n  Graph edge_probs:")
        print(f"    Shape: {all_probs.shape}")
        print(f"    Mean: {all_probs.mean():.4f}")
        print(f"    Edges > 0.5: {(all_probs > 0.5).float().sum(dim=(-1,-2)).mean():.1f}")
        print(f"    Edges > 0.3: {(all_probs > 0.3).float().sum(dim=(-1,-2)).mean():.1f}")
        print(f"    Edges > 0.1: {(all_probs > 0.1).float().sum(dim=(-1,-2)).mean():.1f}")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
