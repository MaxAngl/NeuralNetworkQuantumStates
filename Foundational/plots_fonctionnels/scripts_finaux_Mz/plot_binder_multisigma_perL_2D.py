"""
Binder cumulant 2D — un graphe par taille L, tous les sigmas superposés.
Sauvegardé dans Trains_autour_transi_2D/L={L}/binder_multisigma_L{L}_2D.pdf/.png

Lit : Trains_autour_transi_2D/L={L}/binder_data_2D_L{L}.npz
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE_DIR = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_autour_transi_2D"
L_LIST   = [3, 4, 5, 6, 8, 10]
HC_INF   = 3.044

for L in L_LIST:
    path = os.path.join(BASE_DIR, f"L={L}", f"binder_data_2D_L{L}.npz")
    if not os.path.exists(path):
        print(f"L={L} : fichier manquant, skip.")
        continue

    d          = np.load(path)
    h0_grid    = d["h0_grid"]      # (n_h0,)
    sigma_grid = d["sigma_grid"]   # (n_sigma,)
    binder_raw = d["binder_raw"]   # (n_sigma, n_h0, n_disorder)

    n_sigma = len(sigma_grid)
    colors  = cm.viridis(np.linspace(0, 0.92, n_sigma))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for i_s, sigma in enumerate(sigma_grid):
        b_raw  = binder_raw[i_s]                          # (n_h0, n_disorder)
        b_mean = np.nanmean(b_raw, axis=1)
        n_dis  = np.sum(~np.isnan(b_raw[0]))
        b_sem  = np.nanstd(b_raw, axis=1, ddof=1) / np.sqrt(n_dis)

        ax.errorbar(h0_grid, b_mean, yerr=b_sem,
                    fmt='-o', color=colors[i_s],
                    markersize=2.5, linewidth=0.8,
                    elinewidth=0.7, capsize=2, capthick=0.7,
                    label=rf"$\sigma={sigma}$")

    ax.axvline(x=HC_INF, color='gray', linestyle=':', linewidth=0.8,
               alpha=0.7, label=rf"$h_c^\infty={HC_INF}$")
    ax.axhline(y=2/3, color='k', linestyle='--', linewidth=0.6,
               alpha=0.4, label=r"$B=2/3$")
    ax.axhline(y=0.0, color='k', linestyle='-.', linewidth=0.6, alpha=0.4)

    ax.set_xlabel(r"Champ transverse $h_0$", fontsize=12)
    ax.set_ylabel(r"$B = 1 - \langle M_z^4\rangle\,/\,(3\langle M_z^2\rangle^2)$",
                  fontsize=11)
    ax.set_title(rf"Binder cumulant — 2D TFIM, $L={L}$, tous $\sigma$", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, ls="--", alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()

    out_dir = os.path.join(BASE_DIR, f"L={L}")
    for ext in ("pdf", "png"):
        out = os.path.join(out_dir, f"binder_multisigma_L{L}_2D.{ext}")
        plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"L={L} sauvegardé dans {out_dir}/")
