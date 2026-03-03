"""
Replot comparison with outlier-robust statistics (median + IQR).
Reads from graphs/vscore_comparison_embed_v2.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
graphs_dir = os.path.join(project_root, "graphs")

# Load data
df = pd.read_csv(os.path.join(graphs_dir, "vscore_comparison_embed_v2.csv"))
sigmas = np.sort(df["sigma_test"].unique())

# Compute robust stats per sigma
def robust_stats(values):
    median = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    return median, q25, q75

med_orig, q25_orig, q75_orig = [], [], []
med_dual, q25_dual, q75_dual = [], [], []

for s in sigmas:
    mask = df["sigma_test"] == s
    v_o = df.loc[mask, "v_score_original"].values
    v_d = df.loc[mask, "v_score_dual"].values

    m, q25, q75 = robust_stats(v_o)
    med_orig.append(m); q25_orig.append(q25); q75_orig.append(q75)

    m, q25, q75 = robust_stats(v_d)
    med_dual.append(m); q25_dual.append(q25); q75_dual.append(q75)

med_orig = np.array(med_orig); q25_orig = np.array(q25_orig); q75_orig = np.array(q75_orig)
med_dual = np.array(med_dual); q25_dual = np.array(q25_dual); q75_dual = np.array(q75_dual)

sigma_train = 0.1

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: V-score vs sigma (median + IQR) ---
ax = axes[0]
ax.plot(sigmas, med_orig, "o-", color="tab:blue", label="Original (single embed)", markersize=3, linewidth=1.5)
ax.fill_between(sigmas, q25_orig, q75_orig, color="tab:blue", alpha=0.2, label="IQR Original")
ax.plot(sigmas, med_dual, "s-", color="tab:orange", label="DualEmbed (separate)", markersize=3, linewidth=1.5)
ax.fill_between(sigmas, q25_dual, q75_dual, color="tab:orange", alpha=0.2, label="IQR DualEmbed")

ax.axvline(sigma_train, color="gray", linestyle="--", alpha=0.7, label=r"$\sigma_{train}$" + f"={sigma_train}")
ax.set_xlabel(r"$\sigma_{test}$ (disorder strength)", fontsize=12)
ax.set_ylabel("V-score (median)", fontsize=12)
ax.set_yscale("log")
ax.set_title(
    "V-score vs disorder — L=16, $h_0$=1.0\n"
    f"Trained on $\\sigma={sigma_train}$, 20 replicas, 200 iters\n"
    "Shaded: interquartile range (Q25–Q75)",
    fontsize=11,
)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)

# --- Right: Ratio (median-based) ---
ax2 = axes[1]
ratio = med_dual / med_orig
ax2.plot(sigmas, ratio, "k-o", markersize=3, linewidth=1.5)
ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
ax2.axvline(sigma_train, color="gray", linestyle="--", alpha=0.5, label=r"$\sigma_{train}$" + f"={sigma_train}")

ax2.fill_between(sigmas, 0, 1, alpha=0.08, color="green")
ax2.fill_between(sigmas, 1, max(ratio.max() * 1.1, 1.1), alpha=0.08, color="red")
ax2.annotate("DualEmbed better", xy=(0.02, 0.15), fontsize=10, color="green", alpha=0.7)
ax2.annotate("Original better", xy=(0.02, 1.02), fontsize=10, color="red", alpha=0.7)

ax2.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax2.set_ylabel("V-score ratio (DualEmbed / Original)", fontsize=12)
ax2.set_title("Relative performance (median-based)", fontsize=12)
ax2.set_ylim(0, max(ratio.max() * 1.1, 1.1))
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

pdf_path = os.path.join(graphs_dir, "vscore_comparison_embed_v2_clean.pdf")
png_path = os.path.join(graphs_dir, "vscore_comparison_embed_v2_clean.png")
plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")

print(f"Plots saved to:")
print(f"  {pdf_path}")
print(f"  {png_path}")
