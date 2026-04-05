"""Generate val_loss trajectory chart across 3 iterations of auto-research."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"

# --- Data ---
v1_exps = list(range(20))
v1_vals = [1.381, 1.315, 1.607, 2.291, 1.308, 1.295, 1.296, 1.297, 1.296, 1.335,
           1.280, 1.281, 1.282, 1.307, 1.280, 1.284, 1.307, 1.284, None, 1.280]
v1_kept = {1, 4, 5, 10, 14}

v2_exps = list(range(9))
v2_vals = [1.387, 1.390, 1.338, 1.314, 1.319, 1.338, 1.355, 1.333, 1.304]
v2_kept = set()  # none marked as kept

# v3 is missing exp 3
v3_exps = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
v3_vals = [1.381, 1.608, 1.316, 1.298, 1.297, 1.290, 1.291, 1.271, 1.275, 1.259,
           1.261, 1.259, 1.272, 1.259, 1.257, 1.257, 1.254, 1.257, 1.259, 1.255, 1.256, 1.255]
v3_kept = {2, 4, 5, 6, 8, 10, 12, 15, 16, 17}

BASELINE = 1.381

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 6))

# Helper: plot line + kept/reverted markers
def plot_series(exps, vals, kept, color, label):
    # Filter out None (OOM)
    valid_x = [x for x, v in zip(exps, vals) if v is not None]
    valid_y = [v for v in vals if v is not None]
    ax.plot(valid_x, valid_y, color=color, linewidth=1.5, label=label, zorder=2)

    for x, v in zip(exps, vals):
        if v is None:
            continue
        if x in kept:
            ax.scatter(x, v, color=color, s=40, zorder=3, edgecolors=color, linewidths=1.2)
        else:
            ax.scatter(x, v, facecolors="white", edgecolors=color, s=40, zorder=3, linewidths=1.2)

plot_series(v1_exps, v1_vals, v1_kept, "#3375b8", "v1 \u2013 schedule fallback")
plot_series(v2_exps, v2_vals, v2_kept, "#e08838", "v2 \u2013 config tuning")
plot_series(v3_exps, v3_vals, v3_kept, "#2ca02c", "v3 \u2013 code editing")

# Baseline
ax.axhline(y=BASELINE, color="#999999", linestyle="--", linewidth=1, zorder=1)
ax.text(22.3, BASELINE, "baseline", va="center", fontsize=8, color="#999999")

# Annotations
ax.annotate("lr=5e\u22124", xy=(1, 1.315), xytext=(3, 1.42),
            fontsize=8, color="#3375b8",
            arrowprops=dict(arrowstyle="->", color="#3375b8", lw=0.8))

ax.annotate("MLP modules", xy=(2, 1.316), xytext=(4.5, 1.20),
            fontsize=8, color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8))

# Axes
ax.set_xlabel("Experiment number", fontsize=10)
ax.set_ylabel("val_loss", fontsize=10)
ax.set_title("Auto-Research: val_loss Trajectory Across 3 Iterations", fontsize=13, fontweight="bold")

# Legend: add custom entries for kept/reverted
from matplotlib.lines import Line2D
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray", markersize=6, linestyle="None"))
labels.append("kept")
handles.append(Line2D([0], [0], marker="o", color="gray", markerfacecolor="white", markersize=6, linestyle="None"))
labels.append("reverted")
ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=9, framealpha=0.9)

# Style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)
ax.set_xlim(-0.5, 23.5)

fig.tight_layout()

# Save
out_dir = "/Users/praneeth.paikray/Documents/Code/articles/auto-research-on-databricks-serverless/assets"
fig.savefig(f"{out_dir}/val_loss_trajectory.png", dpi=150)
fig.savefig(f"{out_dir}/val_loss_trajectory.svg")
print(f"Saved to {out_dir}/val_loss_trajectory.png and .svg")
