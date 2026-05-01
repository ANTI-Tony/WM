"""
Publication-quality architecture diagram — dense, detailed, multi-panel.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path


def draw():
    fig = plt.figure(figsize=(14, 7.5))

    # Main canvas
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Background panels
    # Phase 1: Encoding
    p1 = FancyBboxPatch((0.15, 2.8), 3.0, 4.3, boxstyle="round,pad=0.15",
                         facecolor="#f0f9ff", edgecolor="#93c5fd", linewidth=1.5, zorder=0)
    ax.add_patch(p1)
    ax.text(1.65, 6.85, "Phase 1: Object Encoding", ha="center", fontsize=10,
            fontweight="bold", color="#1e40af")

    # Phase 2: Causal Discovery + Dynamics
    p2 = FancyBboxPatch((3.4, 2.8), 7.5, 4.3, boxstyle="round,pad=0.15",
                         facecolor="#fefce8", edgecolor="#fbbf24", linewidth=1.5, zorder=0)
    ax.add_patch(p2)
    ax.text(7.15, 6.85, "Phase 2: Causal Interaction Discovery + Modular Dynamics",
            ha="center", fontsize=10, fontweight="bold", color="#92400e")

    # Phase 3: Decoding
    p3 = FancyBboxPatch((11.15, 2.8), 2.7, 4.3, boxstyle="round,pad=0.15",
                         facecolor="#f0fdf4", edgecolor="#86efac", linewidth=1.5, zorder=0)
    ax.add_patch(p3)
    ax.text(12.5, 6.85, "Phase 3: Prediction", ha="center", fontsize=10,
            fontweight="bold", color="#166534")

    # ===== PHASE 1: INPUT SCENE + ENCODER =====

    # Draw example scene (colored circles on dark bg)
    scene_cx, scene_cy = 1.0, 5.6
    scene_r = 0.55
    scene_bg = FancyBboxPatch((scene_cx-scene_r, scene_cy-scene_r), 2*scene_r, 2*scene_r,
                               boxstyle="round,pad=0.05", facecolor="#1e293b",
                               edgecolor="#475569", linewidth=1.5, zorder=2)
    ax.add_patch(scene_bg)
    # Mini circles
    objs = [(0.65, 5.85, 0.08, "#ef4444"), (0.85, 5.45, 0.06, "#3b82f6"),
            (1.2, 5.75, 0.10, "#22c55e"), (1.35, 5.4, 0.07, "#eab308")]
    for ox, oy, r, c in objs:
        ax.add_patch(Circle((ox, oy), r, color=c, zorder=3))
    ax.text(1.0, 4.85, "Input scene\n$t=0$", ha="center", fontsize=7, color="#475569")

    # State encoder box
    enc_x, enc_y = 0.4, 3.4
    enc = FancyBboxPatch((enc_x, enc_y), 1.2, 1.1, boxstyle="round,pad=0.1",
                          facecolor="#dbeafe", edgecolor="#3b82f6", linewidth=2, zorder=2)
    ax.add_patch(enc)
    ax.text(enc_x+0.6, enc_y+0.7, "State Encoder", ha="center", fontsize=8, fontweight="bold", color="#1e40af")
    ax.text(enc_x+0.6, enc_y+0.35, "$f_{\\rm enc}$: MLP", ha="center", fontsize=7, color="#3b82f6")
    ax.text(enc_x+0.6, enc_y+0.1, "$\\mathbf{s}_t^i \\to \\mathbf{h}_t^i$", ha="center", fontsize=7, color="#64748b")

    # Arrow: scene → encoder
    ax.annotate("", xy=(1.0, 4.5), xytext=(1.0, 4.8),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.5))

    # State vector detail
    ax.text(2.2, 5.6, "$\\mathbf{s}_t^i = [x, y,$\n$v_x, v_y,$\n$r, g, b, rad]$",
            ha="left", fontsize=7, color="#334155",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1", alpha=0.9))

    # Arrow: encoder output
    ax.annotate("", xy=(1.8, 3.95), xytext=(1.6, 3.95),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=2))

    # Slot representations
    ax.text(2.3, 4.2, "Object Slots", ha="center", fontsize=7, fontweight="bold", color="#475569")
    slot_colors = ["#ef4444", "#3b82f6", "#22c55e", "#eab308"]
    for i, c in enumerate(slot_colors):
        rect = FancyBboxPatch((1.85, 3.2 + i*0.22), 0.9, 0.18, boxstyle="round,pad=0.03",
                               facecolor=c, alpha=0.3, edgecolor=c, linewidth=1, zorder=2)
        ax.add_patch(rect)
        ax.text(2.3, 3.29 + i*0.22, f"$\\mathbf{{h}}^{i+1}_t$ (d=128)", ha="center",
                fontsize=5.5, color="#1e293b")

    # ===== PHASE 2a: CAUSAL GRAPH DISCOVERY =====

    graph_x, graph_y = 3.7, 4.4
    gbox = FancyBboxPatch((graph_x, graph_y), 2.8, 2.2, boxstyle="round,pad=0.1",
                           facecolor="#fce7f3", edgecolor="#ec4899", linewidth=2, zorder=2)
    ax.add_patch(gbox)
    ax.text(graph_x+1.4, graph_y+1.95, "Causal Graph Discovery", ha="center",
            fontsize=9, fontweight="bold", color="#be185d")
    ax.text(graph_x+1.4, graph_y+1.6, "$g_{\\rm graph}$", ha="center",
            fontsize=8, color="#ec4899", style="italic")

    # Edge prediction detail
    ax.text(graph_x+0.15, graph_y+1.2, "Edge existence:", ha="left", fontsize=6.5,
            fontweight="bold", color="#831843")
    ax.text(graph_x+0.15, graph_y+0.9, "$e_{ij} = \\sigma(\\mathrm{MLP}([\\mathbf{h}^i;\\mathbf{h}^j;|\\mathbf{h}^i-\\mathbf{h}^j|]))$",
            ha="left", fontsize=6, color="#9d174d")
    ax.text(graph_x+0.15, graph_y+0.55, "Type classification:", ha="left", fontsize=6.5,
            fontweight="bold", color="#831843")
    ax.text(graph_x+0.15, graph_y+0.25, "$\\mathbf{w}_{ij} = \\mathrm{GumbelSoftmax}(\\mathrm{MLP}(...), \\tau)$",
            ha="left", fontsize=6, color="#9d174d")

    # Mini graph illustration
    gx, gy = 5.7, 5.5
    nodes = [(gx-0.3, gy+0.25), (gx+0.3, gy+0.25), (gx-0.15, gy-0.2), (gx+0.15, gy-0.2)]
    ncolors = ["#ef4444", "#3b82f6", "#22c55e", "#eab308"]
    # Edges with varying thickness
    edges = [((0,1), 0.8, "#ef4444"), ((0,2), 0.2, "#94a3b8"), ((1,3), 0.6, "#3b82f6"),
             ((2,3), 0.1, "#94a3b8"), ((0,3), 0.7, "#22c55e")]
    for (i,j), w, c in edges:
        ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]],
                color=c, linewidth=w*3, alpha=w, zorder=2)
    for (nx, ny), nc in zip(nodes, ncolors):
        ax.add_patch(Circle((nx, ny), 0.09, color=nc, zorder=3, edgecolor="white", linewidth=1))

    # Arrow: slots → graph
    ax.annotate("", xy=(3.7, 5.2), xytext=(2.8, 4.0),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.8,
                                connectionstyle="arc3,rad=-0.2"))

    # ===== PHASE 2b: MODULAR CAUSAL DYNAMICS =====

    dyn_x, dyn_y = 3.7, 3.0
    dbox = FancyBboxPatch((dyn_x, dyn_y), 5.8, 1.2, boxstyle="round,pad=0.1",
                           facecolor="#dcfce7", edgecolor="#22c55e", linewidth=2, zorder=2)
    ax.add_patch(dbox)
    ax.text(dyn_x+2.9, dyn_y+0.95, "Modular Causal Dynamics $f_{\\rm dyn}$",
            ha="center", fontsize=9, fontweight="bold", color="#166534")

    # Self-dynamics
    self_x = dyn_x + 0.2
    self_box = FancyBboxPatch((self_x, dyn_y+0.15), 1.3, 0.55, boxstyle="round,pad=0.05",
                               facecolor="#bbf7d0", edgecolor="#4ade80", linewidth=1.2, zorder=3)
    ax.add_patch(self_box)
    ax.text(self_x+0.65, dyn_y+0.55, "$f_{\\rm self}$", ha="center", fontsize=8,
            fontweight="bold", color="#166534")
    ax.text(self_x+0.65, dyn_y+0.28, "self-dynamics\n(gravity, inertia)", ha="center",
            fontsize=5.5, color="#15803d")

    # Typed interaction modules
    type_info = [
        ("$f^1_{\\rm inter}$", "collision", "#ef4444"),
        ("$f^2_{\\rm inter}$", "gravity", "#3b82f6"),
        ("$f^3_{\\rm inter}$", "charge", "#a855f7"),
        ("$\\cdots$", "", "#94a3b8"),
        ("$f^M_{\\rm inter}$", "type M", "#f59e0b"),
    ]
    for i, (lbl, desc, c) in enumerate(type_info):
        tx = dyn_x + 1.8 + i * 0.75
        tbox = FancyBboxPatch((tx, dyn_y+0.15), 0.65, 0.55, boxstyle="round,pad=0.05",
                               facecolor=c, alpha=0.15, edgecolor=c, linewidth=1.2, zorder=3)
        ax.add_patch(tbox)
        ax.text(tx+0.325, dyn_y+0.5, lbl, ha="center", fontsize=7, fontweight="bold", color=c)
        if desc:
            ax.text(tx+0.325, dyn_y+0.25, desc, ha="center", fontsize=4.5, color=c)

    # Update equation
    ax.text(7.7, dyn_y+0.55, "$+$", ha="center", fontsize=12, fontweight="bold", color="#166534")
    upd_box = FancyBboxPatch((7.95, dyn_y+0.15), 1.4, 0.55, boxstyle="round,pad=0.05",
                              facecolor="#bbf7d0", edgecolor="#4ade80", linewidth=1.2, zorder=3)
    ax.add_patch(upd_box)
    ax.text(8.65, dyn_y+0.55, "MLP$_{\\rm update}$", ha="center", fontsize=7,
            fontweight="bold", color="#166534")
    ax.text(8.65, dyn_y+0.28, "$\\to \\mathbf{h}_{t+1}^i$", ha="center", fontsize=7, color="#15803d")

    # Arrow: graph → dynamics (edge probs)
    ax.annotate("", xy=(6.5, 4.2), xytext=(5.5, 4.4),
                arrowprops=dict(arrowstyle="-|>", color="#ec4899", lw=1.8, linestyle="--"))
    ax.text(6.3, 4.45, "$e_{ij}, \\mathbf{w}_{ij}$", fontsize=7, color="#be185d",
            ha="center", style="italic")

    # Arrow: dynamics output
    ax.annotate("", xy=(9.7, 3.6), xytext=(9.35, 3.6),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=2))

    # ===== PHASE 3: DECODER + OUTPUT =====

    dec_x, dec_y = 11.4, 4.0
    dec = FancyBboxPatch((dec_x, dec_y), 1.4, 1.1, boxstyle="round,pad=0.1",
                          facecolor="#fef3c7", edgecolor="#f59e0b", linewidth=2, zorder=2)
    ax.add_patch(dec)
    ax.text(dec_x+0.7, dec_y+0.75, "State Decoder", ha="center", fontsize=8,
            fontweight="bold", color="#92400e")
    ax.text(dec_x+0.7, dec_y+0.4, "$f_{\\rm dec}$: MLP", ha="center", fontsize=7, color="#f59e0b")
    ax.text(dec_x+0.7, dec_y+0.15, "$\\mathbf{h}_{t+1}^i \\to \\hat{\\mathbf{s}}_{t+1}^i$",
            ha="center", fontsize=7, color="#92400e")

    # Arrow: dynamics → decoder
    ax.annotate("", xy=(11.4, 4.5), xytext=(9.7, 3.6),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=2,
                                connectionstyle="arc3,rad=-0.15"))

    # Output predicted scene
    out_cx, out_cy = 12.5, 5.6
    out_bg = FancyBboxPatch((out_cx-0.55, out_cy-0.55), 1.1, 1.1,
                             boxstyle="round,pad=0.05", facecolor="#1e293b",
                             edgecolor="#475569", linewidth=1.5, zorder=2, linestyle="--")
    ax.add_patch(out_bg)
    # Predicted positions (slightly shifted)
    pred_objs = [(out_cx-0.3, out_cy+0.15, 0.08, "#ef4444"),
                 (out_cx+0.05, out_cy-0.15, 0.06, "#3b82f6"),
                 (out_cx+0.2, out_cy+0.25, 0.10, "#22c55e"),
                 (out_cx+0.35, out_cy-0.1, 0.07, "#eab308")]
    for ox, oy, r, c in pred_objs:
        ax.add_patch(Circle((ox, oy), r, color=c, zorder=3, alpha=0.8))
    ax.text(out_cx, out_cy-0.7, "Predicted\n$\\hat{\\mathbf{s}}_{t+1}$", ha="center",
            fontsize=7, color="#475569")

    # Arrow: decoder → output
    ax.annotate("", xy=(12.1, 5.6), xytext=(12.1, 5.1),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.5))

    # ===== AUTOREGRESSIVE ROLLOUT ARROW =====
    ax.annotate("", xy=(0.6, 3.0), xytext=(12.8, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2.5,
                                connectionstyle="arc3,rad=0.25", linestyle=":",
                                mutation_scale=15, zorder=1))
    ax.text(6.5, 1.5, "Autoregressive rollout: $\\hat{\\mathbf{s}}_{t+1}$ feeds back as $\\mathbf{s}_{t+1}$",
            ha="center", fontsize=9, color="#7c3aed", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ede9fe", edgecolor="#c4b5fd", alpha=0.9))

    # ===== BOTTOM: KEY EQUATIONS + MINI RESULTS =====

    # Equation box
    eq_box = FancyBboxPatch((0.3, 0.15), 6.0, 1.1, boxstyle="round,pad=0.1",
                             facecolor="white", edgecolor="#e2e8f0", linewidth=1.5, zorder=2)
    ax.add_patch(eq_box)
    ax.text(3.3, 1.0, "Update rule:", ha="center", fontsize=8, fontweight="bold", color="#1e293b")
    ax.text(3.3, 0.6,
            "$\\mathbf{h}_{t+1}^i = \\mathrm{MLP}\\left[f_{\\rm self}(\\mathbf{h}_t^i),\\;"
            "\\sum_{j} e_{ij} \\sum_{\\tau} w_{ij}^\\tau \\cdot f^\\tau_{\\rm inter}(\\mathbf{h}_t^j, \\mathbf{h}_t^i)\\right]$",
            ha="center", fontsize=8, color="#334155")

    # Mini result box
    res_box = FancyBboxPatch((6.6, 0.15), 3.8, 1.1, boxstyle="round,pad=0.1",
                              facecolor="white", edgecolor="#e2e8f0", linewidth=1.5, zorder=2)
    ax.add_patch(res_box)
    ax.text(8.5, 1.0, "Discovered graph statistics:", ha="center", fontsize=7,
            fontweight="bold", color="#1e293b")
    ax.text(8.5, 0.7, "Colliding pairs: $e_{ij} > 0.5$ ($\\sim$6.4 edges)",
            ha="center", fontsize=6.5, color="#dc2626")
    ax.text(8.5, 0.42, "Non-colliding: $e_{ij} < 0.1$ ($\\sim$20.4 edges)",
            ha="center", fontsize=6.5, color="#2563eb")

    # Key insight box
    key_box = FancyBboxPatch((10.7, 0.15), 3.1, 1.1, boxstyle="round,pad=0.1",
                              facecolor="#fef2f2", edgecolor="#fca5a5", linewidth=1.5, zorder=2)
    ax.add_patch(key_box)
    ax.text(12.25, 1.0, "Key insight:", ha="center", fontsize=7,
            fontweight="bold", color="#991b1b")
    ax.text(12.25, 0.65, "Typed modules $f^\\tau$ learned on\ntraining pairs transfer to novel\nobject combinations at test time",
            ha="center", fontsize=6.5, color="#7f1d1d")

    plt.savefig("paper/figures/architecture.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved paper/figures/architecture.pdf")


if __name__ == "__main__":
    draw()
