"""
Architecture diagram v4: grid-aligned, even spacing, clear rollout arrow.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})

def draw():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    ARROW = "#444444"

    # ===== Layout grid =====
    # 4 main boxes evenly spaced at y=1.8, height=1.6
    box_w = 2.2
    box_h = 1.6
    box_y = 1.8
    gap = 0.55
    x_positions = [0.4, 0.4 + box_w + gap, 0.4 + 2*(box_w + gap), 0.4 + 3*(box_w + gap)]
    # x_positions = [0.4, 3.15, 5.9, 8.65]

    box_specs = [
        (x_positions[0], "State\nEncoder", "$f_{\\mathrm{enc}}$", "#c5ddf5", "#5a9ac7"),
        (x_positions[1], "Causal Graph\nDiscovery", "$g_{\\mathrm{graph}}$", "#f5d0e0", "#c44e78"),
        (x_positions[2], "Modular Causal\nDynamics", "$f_{\\mathrm{dyn}}$", "#c8f0d0", "#3a9e52"),
        (x_positions[3], "State\nDecoder", "$f_{\\mathrm{dec}}$", "#fce8b8", "#c49430"),
    ]

    # Draw boxes
    for bx, title, sub, fc, ec in box_specs:
        ax.add_patch(FancyBboxPatch((bx, box_y), box_w, box_h, boxstyle="round,pad=0.15",
                                     facecolor=fc, edgecolor=ec, lw=2))
        ax.text(bx + box_w/2, box_y + box_h*0.65, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#222")
        ax.text(bx + box_w/2, box_y + box_h*0.25, sub, ha="center", va="center",
                fontsize=12, color=ec, style="italic")

    # ===== Arrows between boxes =====
    for i in range(3):
        x_from = x_positions[i] + box_w
        x_to = x_positions[i+1]
        mid_y = box_y + box_h / 2
        ax.annotate("", xy=(x_to, mid_y), xytext=(x_from, mid_y),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=2, mutation_scale=16))

    # Label on arrows
    ax.text((x_positions[0] + box_w + x_positions[1]) / 2, box_y + box_h/2 + 0.2,
            "$\\mathbf{h}_t^i$", ha="center", fontsize=10, color="#555")
    ax.text((x_positions[1] + box_w + x_positions[2]) / 2, box_y + box_h/2 + 0.2,
            "$e_{ij}, \\mathbf{w}_{ij}$", ha="center", fontsize=9, color="#c44e78", style="italic")
    ax.text((x_positions[2] + box_w + x_positions[3]) / 2, box_y + box_h/2 + 0.2,
            "$\\mathbf{h}_{t+1}^i$", ha="center", fontsize=10, color="#555")

    # ===== Input above encoder =====
    # Scene box
    sc_w, sc_h = 1.2, 1.0
    sc_x = x_positions[0] + (box_w - sc_w) / 2
    sc_y = box_y + box_h + 0.35
    ax.add_patch(FancyBboxPatch((sc_x, sc_y), sc_w, sc_h, boxstyle="round,pad=0.05",
                                 facecolor="#1e293b", edgecolor="#555", lw=1))
    objs_in = [(sc_x+0.25, sc_y+0.65, 0.08, "#EE6677"), (sc_x+0.55, sc_y+0.6, 0.06, "#4477AA"),
               (sc_x+0.4, sc_y+0.3, 0.09, "#228833"), (sc_x+0.85, sc_y+0.4, 0.07, "#CCBB44")]
    for ox, oy, r, c in objs_in:
        ax.add_patch(Circle((ox, oy), r, fc=c, ec="white", lw=0.7, zorder=3))
    ax.text(sc_x + sc_w/2, sc_y - 0.15, "$\\mathbf{s}_t^{1..K}$", ha="center", fontsize=11, color="#333")

    # Arrow: scene → encoder
    ax.annotate("", xy=(x_positions[0] + box_w/2, box_y + box_h),
                xytext=(sc_x + sc_w/2, sc_y),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.5, mutation_scale=14))

    # ===== Output above decoder =====
    out_x = x_positions[3] + (box_w - sc_w) / 2
    out_y = sc_y
    ax.add_patch(FancyBboxPatch((out_x, out_y), sc_w, sc_h, boxstyle="round,pad=0.05",
                                 facecolor="#1e293b", edgecolor="#555", lw=1, linestyle="--"))
    pred_objs = [(out_x+0.3, out_y+0.6, 0.08, "#EE6677"), (out_x+0.6, out_y+0.55, 0.06, "#4477AA"),
                 (out_x+0.45, out_y+0.25, 0.09, "#228833"), (out_x+0.9, out_y+0.35, 0.07, "#CCBB44")]
    for ox, oy, r, c in pred_objs:
        ax.add_patch(Circle((ox, oy), r, fc=c, ec="white", lw=0.7, alpha=0.85, zorder=3))
    ax.text(out_x + sc_w/2, out_y - 0.15, "$\\hat{\\mathbf{s}}_{t+1}^{1..K}$",
            ha="center", fontsize=11, color="#333")

    # Arrow: decoder → output
    ax.annotate("", xy=(x_positions[3] + box_w/2, box_y + box_h),
                xytext=(out_x + sc_w/2, out_y),
                arrowprops=dict(arrowstyle="<|-", color=ARROW, lw=1.5, mutation_scale=14))

    # ===== Typed modules inside dynamics box =====
    mod_y = box_y + 0.12
    mod_h = 0.45
    mod_w = 0.42
    mod_labels = ["$f^1$", "$f^2$", "$f^3$", "$\\cdots$", "$f^M$"]
    mod_colors = ["#EE6677", "#4477AA", "#AA3377", "#999", "#CCBB44"]
    mod_start = x_positions[2] + 0.15
    for k, (lbl, c) in enumerate(zip(mod_labels, mod_colors)):
        mx = mod_start + k * (mod_w + 0.08)
        ax.add_patch(FancyBboxPatch((mx, mod_y), mod_w, mod_h, boxstyle="round,pad=0.04",
                                     facecolor=c, alpha=0.2, edgecolor=c, lw=1))
        ax.text(mx + mod_w/2, mod_y + mod_h/2, lbl, ha="center", va="center",
                fontsize=9, fontweight="bold", color=c)
    ax.text(x_positions[2] + box_w/2, mod_y + mod_h + 0.08,
            "typed interaction modules", ha="center", fontsize=7.5, color="#2a7a3c", style="italic")

    # ===== Mini graph inside graph discovery box =====
    gx_base = x_positions[1] + 0.35
    gy_base = box_y + 0.25
    gnodes = [(gx_base, gy_base+0.35, "#EE6677"), (gx_base+0.45, gy_base+0.35, "#4477AA"),
              (gx_base+0.22, gy_base, "#228833"), (gx_base+0.65, gy_base+0.05, "#CCBB44")]
    gedges = [(0,1,0.85), (0,2,0.1), (1,3,0.7), (2,3,0.15)]
    for i, j, w in gedges:
        ax.plot([gnodes[i][0], gnodes[j][0]], [gnodes[i][1], gnodes[j][1]],
                color="#c44e78", lw=w*3.5, alpha=max(w, 0.15), zorder=2, solid_capstyle="round")
    for gx, gy, gc in gnodes:
        ax.add_patch(Circle((gx, gy), 0.07, fc=gc, ec="white", lw=0.8, zorder=3))

    # ===== Autoregressive rollout arrow =====
    # From predicted output (right) back to state encoder input (left)
    # Clear path: output bottom → below all boxes → encoder bottom
    arr_y = box_y - 0.15  # below the boxes
    out_center_x = out_x + sc_w / 2
    enc_center_x = sc_x + sc_w / 2

    # Right end: down from output
    ax.annotate("", xy=(out_center_x, arr_y + 0.3), xytext=(out_center_x, out_y - 0.15),
                arrowprops=dict(arrowstyle="-", color="#7c3aed", lw=2, linestyle=":"))
    # Horizontal line below boxes
    ax.plot([out_center_x, enc_center_x], [arr_y + 0.3, arr_y + 0.3],
            color="#7c3aed", lw=2, linestyle=":", zorder=1)
    # Left end: up to encoder
    ax.annotate("", xy=(enc_center_x, box_y),
                xytext=(enc_center_x, arr_y + 0.3),
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2, linestyle=":",
                                mutation_scale=15))

    # Label on the rollout arrow
    ax.text(6.0, arr_y + 0.05,
            "Autoregressive rollout:  $\\hat{\\mathbf{s}}_{t+1}$  feeds back as  $\\mathbf{s}_{t+1}$",
            ha="center", fontsize=9, color="#7c3aed", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0ebff", edgecolor="#c4b5fd", lw=0.7))

    plt.savefig("paper/figures/architecture.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved paper/figures/architecture.pdf")


if __name__ == "__main__":
    draw()
