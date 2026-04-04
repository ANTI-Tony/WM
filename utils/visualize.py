"""Visualization utilities for CausalComp."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


INTERACTION_COLORS = {
    0: "#e74c3c",  # collision - red
    1: "#2ecc71",  # contact - green
    2: "#3498db",  # approach - blue
    3: "#95a5a6",  # none/other - gray
}

INTERACTION_NAMES = {
    0: "collision",
    1: "contact",
    2: "approach",
    3: "none",
}


def visualize_slots(images, recon, masks, save_path: Path, max_slots: int = 8):
    """Visualize slot decomposition.

    Args:
        images: [B, 3, H, W] original images
        recon: [B, 3, H, W] reconstructed images
        masks: [B, K, 1, H, W] slot attention masks
        save_path: where to save the figure
    """
    B = min(images.shape[0], 4)
    K = min(masks.shape[1], max_slots)

    fig, axes = plt.subplots(B, K + 2, figsize=(2 * (K + 2), 2 * B))
    if B == 1:
        axes = axes[np.newaxis]

    for b in range(B):
        # Original
        img = images[b].cpu().permute(1, 2, 0).numpy()
        axes[b, 0].imshow(img.clip(0, 1))
        axes[b, 0].set_title("Original")
        axes[b, 0].axis("off")

        # Reconstruction
        rec = recon[b].cpu().permute(1, 2, 0).numpy()
        axes[b, 1].imshow(rec.clip(0, 1))
        axes[b, 1].set_title("Recon")
        axes[b, 1].axis("off")

        # Per-slot masks
        for k in range(K):
            mask = masks[b, k, 0].cpu().numpy()
            axes[b, k + 2].imshow(mask, cmap="viridis", vmin=0, vmax=1)
            axes[b, k + 2].set_title(f"Slot {k}")
            axes[b, k + 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_graph(edge_probs, edge_types, save_path: Path,
                    object_names=None, threshold: float = 0.3):
    """Visualize discovered causal interaction graph.

    Args:
        edge_probs: [K, K] edge existence probabilities
        edge_types: [K, K, M] type assignments
        save_path: where to save
        object_names: optional list of object labels
        threshold: minimum edge probability to display
    """
    K = edge_probs.shape[0]
    if object_names is None:
        object_names = [f"Obj {i}" for i in range(K)]

    # Filter active objects (those with any significant edge)
    active = set()
    for i in range(K):
        for j in range(K):
            if edge_probs[i, j] > threshold:
                active.add(i)
                active.add(j)

    if not active:
        active = set(range(min(K, 4)))

    active = sorted(active)
    n_active = len(active)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Place objects in a circle
    angles = np.linspace(0, 2 * np.pi, n_active, endpoint=False)
    positions = {
        idx: (np.cos(a) * 0.35 + 0.5, np.sin(a) * 0.35 + 0.5)
        for idx, a in zip(active, angles)
    }

    # Draw edges
    for i in active:
        for j in active:
            if i == j:
                continue
            prob = edge_probs[i, j].item() if torch.is_tensor(edge_probs) else edge_probs[i, j]
            if prob < threshold:
                continue

            # Get dominant type
            types = edge_types[i, j]
            if torch.is_tensor(types):
                types = types.numpy()
            dom_type = int(np.argmax(types))

            x1, y1 = positions[i]
            x2, y2 = positions[j]

            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color=INTERACTION_COLORS.get(dom_type, "#333"),
                    lw=2 * prob,
                    alpha=float(prob),
                ),
            )

    # Draw nodes
    for idx in active:
        x, y = positions[idx]
        circle = plt.Circle((x, y), 0.04, color="#2c3e50", zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, object_names[idx], ha="center", va="center",
                fontsize=8, color="white", zorder=6)

    # Legend
    patches = [
        mpatches.Patch(color=c, label=INTERACTION_NAMES.get(t, f"type_{t}"))
        for t, c in INTERACTION_COLORS.items()
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Discovered Causal Interaction Graph")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_trajectory(gt_frames, pred_frames, save_path: Path):
    """Visualize ground truth vs predicted trajectory.

    Args:
        gt_frames: [T, 3, H, W]
        pred_frames: [T, 3, H, W]
        save_path: where to save
    """
    T = min(gt_frames.shape[0], pred_frames.shape[0], 10)

    fig, axes = plt.subplots(2, T, figsize=(2 * T, 4))

    for t in range(T):
        gt = gt_frames[t].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        pred = pred_frames[t].cpu().permute(1, 2, 0).numpy().clip(0, 1)

        axes[0, t].imshow(gt)
        axes[0, t].set_title(f"GT t={t}")
        axes[0, t].axis("off")

        axes[1, t].imshow(pred)
        axes[1, t].set_title(f"Pred t={t}")
        axes[1, t].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
