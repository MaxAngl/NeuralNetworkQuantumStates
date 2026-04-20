import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

TRAINS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../logs/Trains_1D_geants",
)
EXACT_FILE = os.path.join(TRAINS_DIR, "ising1d_pbc_data.json")

H0_MIN, H0_MAX = 0.7, 1.3

# Load exact data (sigma=0)
with open(EXACT_FILE) as f:
    exact_data = json.load(f)
exact_obs = exact_data["observables"]
exact_h = np.array(exact_data["parameters"]["h_values"])
mask_exact = (exact_h >= H0_MIN) & (exact_h <= H0_MAX)

# Map L -> run directory
run_dirs = {}
for name in sorted(os.listdir(TRAINS_DIR)):
    if not name.startswith("run_L"):
        continue
    path = os.path.join(TRAINS_DIR, name)
    if not os.path.isdir(path):
        continue
    L = int(name.split("_")[1][1:])
    run_dirs[L] = path

FONTSIZE_LABEL  = 16
FONTSIZE_TICK   = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TITLE  = 17
LW = 1.8

for L, run_path in sorted(run_dirs.items()):
    files = glob.glob(os.path.join(run_path, f"is_data_1D_L{L}_full.npz"))
    if not files:
        print(f"Fichier manquant pour L={L}, passage.")
        continue

    d = np.load(files[0])
    h0_grid    = d["h0_grid"]
    sigma_grid = d["sigma_grid"]
    mz2_raw    = d["mz2_raw"]   # (n_sigma, n_h0, n_disorder)

    mask = (h0_grid >= H0_MIN) & (h0_grid <= H0_MAX)
    if mask.sum() < 2:
        print(f"L={L}: moins de 2 points dans [0.7, 1.3], passage.")
        continue

    h0_plot  = h0_grid[mask]
    mz2_mean = mz2_raw[:, mask, :].mean(axis=2)
    mz2_std  = mz2_raw[:, mask, :].std(axis=2) / np.sqrt(mz2_raw.shape[2])

    # Color palette (sigma=0 en noir, reste en plasma)
    sigma_nonzero = sigma_grid[sigma_grid > 0]
    cmap = cm.get_cmap("plasma", len(sigma_nonzero))
    sigma_colors = {s: cmap(i) for i, s in enumerate(sigma_nonzero)}

    fig, ax = plt.subplots(figsize=(8, 5))

    # Exact curve sigma=0
    L_key = str(L)
    if L_key in exact_obs:
        mz2_exact = np.array(exact_obs[L_key]["Mz2"] if "Mz2" in exact_obs[L_key] else exact_obs[L_key]["mz2"])
        ax.plot(
            exact_h[mask_exact], mz2_exact[mask_exact],
            color="black", lw=LW + 0.4, ls="--",
            label=r"$\sigma=0$ (exact)",
            zorder=10,
        )

    # NQS curves
    for i_s, sigma in enumerate(sigma_grid):
        if sigma == 0.0:
            color, ls, label = "black", "-", r"$\sigma=0$ (NQS)"
        else:
            color, ls, label = sigma_colors[sigma], "-", rf"$\sigma={sigma:.2g}$"

        ax.plot(h0_plot, mz2_mean[i_s], color=color, lw=LW, ls=ls, label=label)
        ax.fill_between(
            h0_plot,
            mz2_mean[i_s] - mz2_std[i_s],
            mz2_mean[i_s] + mz2_std[i_s],
            color=color, alpha=0.15,
        )

    ax.set_xlabel(r"Champ transverse $h_0$", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r"Magnétisation au carré $\langle M_z^2 \rangle$", fontsize=FONTSIZE_LABEL)
    ax.set_title(rf"Chaîne d'Ising 1D, $L={L}$ spins (CLP)", fontsize=FONTSIZE_TITLE)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax.set_xlim(H0_MIN, H0_MAX)
    ax.set_ylim(bottom=0)
    ax.legend(
        fontsize=FONTSIZE_LEGEND,
        loc="upper right",
        framealpha=0.9,
        title=r"Désordre $\sigma$",
        title_fontsize=FONTSIZE_LEGEND,
    )
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(run_path, f"Mz2_vs_h0_L{L}.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"L={L} → {run_path}/Mz2_vs_h0_L{L}.pdf")

print("Terminé.")
