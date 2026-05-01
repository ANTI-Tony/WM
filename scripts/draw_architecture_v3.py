"""
Clean, readable architecture diagram. Less is more.
Remove clutter, increase font sizes, clear flow.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
})


def draw():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Colors
    BG1 = "#eef4ff"   # encoding
    BG2 = "#fffbe6"   # main
    BG3 = "#eefbf0"   # prediction
    BD1 = "#7faed4"
    BD2 = "#d4a94e"
    BD3 = "#6dba7d"
    BOX_ENC = "#c5ddf5"
    BOX_GRH = "#f5d0e0"
    BOX_DYN = "#c8f0d0"
    BOX_DEC = "#fce8b8"
    ARROW = "#444444"

    # ===== Background panels =====
    ax.add_patch(FancyBboxPatch((0.15, 0.6), 2.85, 4.6, boxstyle="round,pad=0.2",
                                 facecolor=BG1, edgecolor=BD1, lw=1.2))
    ax.text(1.57, 5.0, "Encoding", ha="center", fontsize=11, fontweight="bold", color="#2c5f8a")

    ax.add_patch(FancyBboxPatch((3.2, 0.6), 6.7, 4.6, boxstyle="round,pad=0.2",
                                 facecolor=BG2, edgecolor=BD2, lw=1.2))
    ax.text(6.55, 5.0, "Causal Discovery + Modular Dynamics", ha="center", fontsize=11,
            fontweight="bold", color="#8a6d2c")

    ax.add_patch(FancyBboxPatch((10.1, 0.6), 2.75, 4.6, boxstyle="round,pad=0.2",
                                 facecolor=BG3, edgecolor=BD3, lw=1.2))
    ax.text(11.47, 5.0, "Prediction", ha="center", fontsize=11, fontweight="bold", color="#2c7a3f")

    # ===== Scene (input) =====
    scene_bg = FancyBboxPatch((0.45, 3.0), 1.2, 1.2, boxstyle="round,pad=0.05",
                               facecolor="#1e293b", edgecolor="#555", lw=1.2)
    ax.add_patch(scene_bg)
    objs_in = [(0.7, 3.85, 0.09, "#EE6677"), (1.0, 3.75, 0.07, "#4477AA"),
               (0.85, 3.3, 0.11, "#228833"), (1.3, 3.5, 0.08, "#CCBB44")]
    for ox, oy, r, c in objs_in:
        ax.add_patch(Circle((ox, oy), r, fc=c, ec="white", lw=0.8, zorder=3))
    ax.text(1.05, 2.7, "$\\mathbf{s}_t^{1..K}$", ha="center", fontsize=12, color="#333")

    # ===== State Encoder =====
    ax.add_patch(FancyBboxPatch((0.4, 1.3), 1.3, 1.0, boxstyle="round,pad=0.12",
                                 facecolor=BOX_ENC, edgecolor="#5a9ac7", lw=1.8))
    ax.text(1.05, 1.95, "State Encoder", ha="center", fontsize=10, fontweight="bold", color="#2c5f8a")
    ax.text(1.05, 1.55, "$f_{\\mathrm{enc}}$", ha="center", fontsize=11, color="#3a7bb8", style="italic")

    # Arrow: scene → encoder
    ax.annotate("", xy=(1.05, 2.3), xytext=(1.05, 2.65),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.8, mutation_scale=16))

    # ===== Slots =====
    slot_x = 2.1
    ax.text(2.45, 2.6, "Slots $\\mathbf{h}_t^i$", ha="center", fontsize=9, color="#555", style="italic")
    slot_colors = ["#EE6677", "#4477AA", "#228833", "#CCBB44"]
    for i, c in enumerate(slot_colors):
        ax.add_patch(FancyBboxPatch((slot_x, 1.4 + i * 0.28), 0.7, 0.22, boxstyle="round,pad=0.03",
                                     facecolor=c, alpha=0.35, edgecolor=c, lw=1))

    # Arrow: encoder → slots
    ax.annotate("", xy=(2.1, 1.8), xytext=(1.7, 1.8),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.5, mutation_scale=14))

    # Arrow: slots → graph discovery
    ax.annotate("", xy=(3.6, 3.2), xytext=(2.85, 2.2),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.5, mutation_scale=14,
                                connectionstyle="arc3,rad=-0.15"))

    # ===== Causal Graph Discovery =====
    ax.add_patch(FancyBboxPatch((3.5, 2.6), 3.2, 2.2, boxstyle="round,pad=0.12",
                                 facecolor=BOX_GRH, edgecolor="#c44e78", lw=1.8))
    ax.text(5.1, 4.5, "Causal Graph Discovery", ha="center", fontsize=10,
            fontweight="bold", color="#8a2c4e")
    ax.text(5.1, 4.15, "$g_{\\mathrm{graph}}$", ha="center", fontsize=11,
            color="#c44e78", style="italic")

    # Edge equation
    ax.text(5.1, 3.7, "Edge: $e_{ij} = \\sigma(\\mathrm{MLP}([\\mathbf{h}^i; \\mathbf{h}^j]))$",
            ha="center", fontsize=8, color="#6b2040")
    ax.text(5.1, 3.35, "Type: $\\mathbf{w}_{ij} = \\mathrm{GumbelSoftmax}(\\cdot)$",
            ha="center", fontsize=8, color="#6b2040")

    # Mini graph
    gnodes = [(4.1, 3.0, "#EE6677"), (4.6, 3.0, "#4477AA"),
              (4.35, 2.75, "#228833")]
    gedges = [(0, 1, 0.9), (0, 2, 0.15), (1, 2, 0.6)]
    for i, j, w in gedges:
        ax.plot([gnodes[i][0], gnodes[j][0]], [gnodes[i][1], gnodes[j][1]],
                color="#c44e78", lw=w * 3, alpha=w, zorder=2)
    for gx, gy, gc in gnodes:
        ax.add_patch(Circle((gx, gy), 0.08, fc=gc, ec="white", lw=1, zorder=3))

    # Discovered graph label
    ax.text(5.8, 2.85, "→ $e_{ij}, \\mathbf{w}_{ij}$", fontsize=9, color="#8a2c4e",
            style="italic")

    # ===== Modular Causal Dynamics =====
    ax.add_patch(FancyBboxPatch((3.5, 0.8), 6.2, 1.5, boxstyle="round,pad=0.12",
                                 facecolor=BOX_DYN, edgecolor="#3a9e52", lw=1.8))
    ax.text(6.6, 2.05, "Modular Causal Dynamics  $f_{\\mathrm{dyn}}$", ha="center",
            fontsize=10, fontweight="bold", color="#1e6b30")

    # Self + Typed modules
    mods = [
        ("$f_{\\mathrm{self}}$", "self-dyn", "#6dba7d", 3.8),
        ("$f^1_{\\mathrm{inter}}$", "collision", "#EE6677", 5.2),
        ("$f^2_{\\mathrm{inter}}$", "gravity", "#4477AA", 6.15),
        ("$f^3_{\\mathrm{inter}}$", "charge", "#AA3377", 7.1),
        ("$\\cdots$", "", "#999", 7.85),
        ("$f^M_{\\mathrm{inter}}$", "", "#CCBB44", 8.4),
    ]
    for label, desc, c, mx in mods:
        ax.add_patch(FancyBboxPatch((mx, 1.05), 0.75, 0.65, boxstyle="round,pad=0.06",
                                     facecolor=c, alpha=0.2, edgecolor=c, lw=1.2))
        ax.text(mx + 0.375, 1.55, label, ha="center", fontsize=8, fontweight="bold", color=c)
        if desc:
            ax.text(mx + 0.375, 1.2, desc, ha="center", fontsize=6.5, color=c)

    # Plus and MLP_update
    ax.text(4.7, 1.37, "+", fontsize=14, fontweight="bold", color="#1e6b30", ha="center")

    # Arrow: graph → dynamics
    ax.annotate("", xy=(5.5, 2.3), xytext=(5.5, 2.55),
                arrowprops=dict(arrowstyle="-|>", color="#c44e78", lw=1.5, linestyle="--",
                                mutation_scale=13))

    # Arrow: dynamics → decoder
    ax.annotate("", xy=(10.3, 2.5), xytext=(9.7, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.8, mutation_scale=16,
                                connectionstyle="arc3,rad=-0.2"))

    # Output label from dynamics
    ax.text(9.5, 1.0, "$\\mathbf{h}_{t+1}^i$", fontsize=10, color="#1e6b30",
            style="italic", ha="center")

    # ===== State Decoder =====
    ax.add_patch(FancyBboxPatch((10.3, 1.8), 1.4, 1.0, boxstyle="round,pad=0.12",
                                 facecolor=BOX_DEC, edgecolor="#c49430", lw=1.8))
    ax.text(11.0, 2.5, "State Decoder", ha="center", fontsize=10, fontweight="bold", color="#8a6d2c")
    ax.text(11.0, 2.05, "$f_{\\mathrm{dec}}$", ha="center", fontsize=11, color="#c49430", style="italic")

    # ===== Predicted scene =====
    pred_bg = FancyBboxPatch((10.35, 3.0), 1.2, 1.2, boxstyle="round,pad=0.05",
                              facecolor="#1e293b", edgecolor="#555", lw=1.2, linestyle="--")
    ax.add_patch(pred_bg)
    pred_objs = [(10.6, 3.8, 0.09, "#EE6677"), (10.95, 3.65, 0.07, "#4477AA"),
                 (10.75, 3.25, 0.11, "#228833"), (11.25, 3.45, 0.08, "#CCBB44")]
    for ox, oy, r, c in pred_objs:
        ax.add_patch(Circle((ox, oy), r, fc=c, ec="white", lw=0.8, alpha=0.85, zorder=3))
    ax.text(10.95, 2.7, "$\\hat{\\mathbf{s}}_{t+1}^{1..K}$", ha="center", fontsize=12, color="#333")

    # Arrow: decoder → predicted
    ax.annotate("", xy=(11.0, 3.0), xytext=(11.0, 2.8),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.5, mutation_scale=14))

    # ===== Autoregressive rollout =====
    ax.annotate("", xy=(0.5, 0.8), xytext=(11.5, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2.2,
                                connectionstyle="arc3,rad=0.15", linestyle=":",
                                mutation_scale=15))
    ax.text(6.0, 0.1, "Autoregressive: $\\hat{\\mathbf{s}}_{t+1}$ feeds back as $\\mathbf{s}_{t+1}$",
            ha="center", fontsize=9.5, color="#7c3aed", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0ebff", edgecolor="#c4b5fd", lw=0.8))

    plt.savefig("paper/figures/architecture.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved paper/figures/architecture.pdf")


if __name__ == "__main__":
    draw()
