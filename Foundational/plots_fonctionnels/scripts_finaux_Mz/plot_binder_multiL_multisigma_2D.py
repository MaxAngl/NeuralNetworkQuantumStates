"""
Binder cumulant 2D — un graphe par sigma, toutes les tailles L sur le même graphe.
Moyenne sur les réalisations de désordre + barre d'erreur (SEM).

Lit : Trains_autour_transi_2D/L={L}/binder_data_2D_L{L}.npz
Produit : binder_multiL_multisigma_2D.pdf / .png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR   = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_autour_transi_2D"
L_LIST     = [3, 4, 5, 6, 8, 10]
HC_INF     = 3.044

# ==========================================
# CHARGEMENT
# ==========================================
data = {}   # L -> npz
for L in L_LIST:
    path = os.path.join(BASE_DIR, f"L={L}", f"binder_data_2D_L{L}.npz")
    if os.path.exists(path):
        data[L] = np.load(path)
        print(f"L={L} chargé")
    else:
        print(f"L={L} manquant : {path}")

if not data:
    raise FileNotFoundError("Aucun fichier binder_data trouvé.")

# Union de tous les sigmas disponibles (triée)
all_sigmas = sorted({float(s) for d in data.values() for s in d["sigma_grid"]})
sigma_grid = np.array(all_sigmas)
n_sigma    = len(sigma_grid)

# ==========================================
# FIGURE : n_sigma sous-graphes
# ==========================================
ncols = 4
nrows = int(np.ceil(n_sigma / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8),
                         sharex=False, sharey=True)
axes_flat = axes.flatten()

colors    = plt.cm.plasma(np.linspace(0.1, 0.9, len(L_LIST)))
color_map = {L: colors[i] for i, L in enumerate(L_LIST)}

for i_s, sigma in enumerate(sigma_grid):
    ax = axes_flat[i_s]

    for L, d in data.items():
        # Cherche l'index de ce sigma dans ce fichier (skip si absent)
        idx_s_L = np.where(np.abs(d["sigma_grid"] - sigma) < 1e-9)[0]
        if len(idx_s_L) == 0:
            continue
        h0_grid   = d["h0_grid"]                         # (n_h0,)
        b_raw     = d["binder_raw"][idx_s_L[0]]          # (n_h0, n_disorder)

        b_mean = np.nanmean(b_raw, axis=1)           # (n_h0,)
        n_dis  = np.sum(~np.isnan(b_raw[0]))
        b_sem  = np.nanstd(b_raw, axis=1, ddof=1) / np.sqrt(n_dis)

        c = color_map[L]
        ax.errorbar(h0_grid, b_mean, yerr=b_sem,
                    fmt='-o', color=c, markersize=2.5, linewidth=0.8,
                    elinewidth=0.7, capsize=2, capthick=0.7,
                    label=f"$L={L}$")

    ax.axvline(x=HC_INF, color='gray', linestyle=':', alpha=0.6)
    ax.axhline(y=2/3, color='k', linestyle='--', linewidth=0.7, alpha=0.4)
    ax.axhline(y=0.0, color='k', linestyle='-.', linewidth=0.7, alpha=0.4)

    sigma_str = rf"$\sigma = {sigma}$"
    ax.set_title(sigma_str, fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, ls="--", alpha=0.25)
    ax.set_xlabel(r"$h_0$", fontsize=11)
    if i_s % ncols == 0:
        ax.set_ylabel(r"$B$", fontsize=11)

# Légende commune
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", fontsize=10,
           bbox_to_anchor=(1.0, 0.02), ncol=1)

# Cacher les sous-graphes vides
for j in range(n_sigma, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle(
    r"Binder cumulant — 2D TFIM, toutes tailles, par $\sigma$"
    f"\n" + r"$h_c^\infty = 3.044$, barres = SEM sur les tirages de désordre",
    fontsize=13, fontweight='bold'
)
plt.tight_layout(rect=[0, 0, 0.88, 0.93])

for ext in ("pdf", "png"):
    out = os.path.join(BASE_DIR, f"binder_multiL_multisigma_2D.{ext}")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Sauvegardé : {out}")

plt.close()
