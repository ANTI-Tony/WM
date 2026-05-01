"""
Draw a clean, publication-quality architecture diagram for CausalComp.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path


def draw_architecture(save_path):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.0, 5.0)
    ax.axis("off")

    # ===== Module boxes =====
    box_style = "round,pad=0.15"
    modules = [
        # (x, y, w, h, label, sublabel, facecolor, edgecolor)
        (0.0, 1.5, 2.2, 2.0, "State\nEncoder", "$f_{\\rm enc}$", "#dbeafe", "#3b82f6"),
        (3.0, 1.5, 2.5, 2.0, "Causal Graph\nDiscovery", "$g_{\\rm graph}$", "#fce7f3", "#ec4899"),
        (6.3, 1.5, 2.8, 2.0, "Modular Causal\nDynamics", "$f_{\\rm dyn}$", "#dcfce7", "#22c55e"),
        (9.9, 1.5, 2.2, 2.0, "State\nDecoder", "$f_{\\rm dec}$", "#fef3c7", "#f59e0b"),
    ]

    for x, y, w, h, label, sublabel, fc, ec in modules:
        rect = FancyBboxPatch((x, y), w, h, boxstyle=box_style,
                               facecolor=fc, edgecolor=ec, linewidth=2.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.2, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#1e293b", zorder=3)
        ax.text(x + w/2, y + h/2 - 0.45, sublabel, ha="center", va="center",
                fontsize=10, color="#64748b", zorder=3, style="italic")

    # ===== Main flow arrows =====
    arrow_kw = dict(arrowstyle="-|>", color="#334155", lw=2.5,
                    mutation_scale=18, zorder=4)
    # Encoder → Graph
    ax.annotate("", xy=(3.0, 2.5), xytext=(2.2, 2.5), arrowprops=arrow_kw)
    ax.text(2.6, 2.85, "$\\mathbf{h}_t^i$", ha="center", fontsize=9, color="#475569")

    # Graph → Dynamics
    ax.annotate("", xy=(6.3, 2.5), xytext=(5.5, 2.5), arrowprops=arrow_kw)

    # Dynamics → Decoder
    ax.annotate("", xy=(9.9, 2.5), xytext=(9.1, 2.5), arrowprops=arrow_kw)
    ax.text(9.5, 2.85, "$\\mathbf{h}_{t+1}^i$", ha="center", fontsize=9, color="#475569")

    # ===== Input / Output =====
    # Input
    ax.text(1.1, 4.3, "$\\mathbf{s}_t^{1..K}$", ha="center", fontsize=12,
            fontweight="bold", color="#1e293b")
    ax.text(1.1, 3.9, "Object States", ha="center", fontsize=9, color="#64748b")
    ax.annotate("", xy=(1.1, 3.5), xytext=(1.1, 3.7),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=2, mutation_scale=15))

    # Output
    ax.text(11.0, 4.3, "$\\hat{\\mathbf{s}}_{t+1}^{1..K}$", ha="center", fontsize=12,
            fontweight="bold", color="#1e293b")
    ax.text(11.0, 3.9, "Predicted States", ha="center", fontsize=9, color="#64748b")
    ax.annotate("", xy=(11.0, 3.7), xytext=(11.0, 3.5),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=2, mutation_scale=15))

    # ===== Graph Discovery details =====
    # Edge probs + types arrow (curved, from graph to dynamics)
    ax.annotate("", xy=(7.2, 1.5), xytext=(4.8, 1.5),
                arrowprops=dict(arrowstyle="-|>", color="#ec4899", lw=2,
                                connectionstyle="arc3,rad=-0.4", linestyle="--",
                                mutation_scale=14, zorder=3))
    ax.text(6.0, 0.45, "edge probs $e_{ij}$\n+ types $\\mathbf{w}_{ij}$",
            ha="center", fontsize=9, color="#be185d", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fce7f3", edgecolor="none", alpha=0.8))

    # ===== Typed interaction modules (inside dynamics box) =====
    type_colors = ["#ef4444", "#3b82f6", "#22c55e", "#a855f7"]
    type_labels = ["$f^1$", "$f^2$", "$\\cdots$", "$f^M$"]
    for i, (lbl, c) in enumerate(zip(type_labels, type_colors)):
        bx = 6.6 + i * 0.6
        by = 1.7
        rect = FancyBboxPatch((bx, by), 0.45, 0.45, boxstyle="round,pad=0.08",
                               facecolor=c, edgecolor="white", linewidth=1.5, alpha=0.25, zorder=3)
        ax.add_patch(rect)
        ax.text(bx + 0.225, by + 0.225, lbl, ha="center", va="center",
                fontsize=8, fontweight="bold", color=c, zorder=4)

    ax.text(7.7, 1.35, "typed interaction modules", ha="center", fontsize=7,
            color="#16a34a", style="italic")

    # ===== Autoregressive rollout arrow =====
    ax.annotate("", xy=(0.3, 1.5), xytext=(11.5, 1.0),
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2,
                                connectionstyle="arc3,rad=0.15", linestyle=":",
                                mutation_scale=14, zorder=1))
    ax.text(5.5, -0.3, "autoregressive rollout (predicted $\\hat{\\mathbf{s}}_{t+1}$ feeds back as input)",
            ha="center", fontsize=9, color="#7c3aed", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ede9fe", edgecolor="none", alpha=0.8))

    # ===== Graph structure illustration (mini graph between Graph and Dynamics) =====
    # Small circles representing objects with edges
    cx, cy = 5.75, 3.2
    obj_positions = [(cx-0.3, cy+0.3), (cx+0.3, cy+0.3), (cx, cy-0.2)]
    obj_colors = ["#ef4444", "#3b82f6", "#22c55e"]

    # Draw edges
    for i in range(3):
        for j in range(i+1, 3):
            x1, y1 = obj_positions[i]
            x2, y2 = obj_positions[j]
            alpha = 0.8 if (i == 0 and j == 1) else 0.15
            lw = 2.0 if alpha > 0.5 else 0.8
            ax.plot([x1, x2], [y1, y2], color="#94a3b8", linewidth=lw, alpha=alpha, zorder=2)

    # Draw nodes
    for (x, y), c in zip(obj_positions, obj_colors):
        circle = plt.Circle((x, y), 0.12, color=c, alpha=0.9, zorder=5,
                             edgecolor="white", linewidth=1.5)
        ax.add_patch(circle)

    ax.text(cx, cy + 0.7, "discovered\ngraph", ha="center", fontsize=7,
            color="#64748b", style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


if __name__ == "__main__":
    out = Path("paper/figures")
    out.mkdir(parents=True, exist_ok=True)
    draw_architecture(out / "architecture.pdf")
