"""
Plot du maximum de la susceptibilité (|d<mz²>/dh0|) en fonction de L,
pour chaque sigma, en 1D et 2D.

Lit les données IS depuis Trains_autour_transi_1D et Trains_autour_transi_2D,
fusionne les chunks, calcule la susceptibilité et son maximum avec incertitude.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PROJECT_ROOT = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
BASE_1D = os.path.join(PROJECT_ROOT, "Foundational/logs/Trains_autour_transi_1D")
BASE_2D = os.path.join(PROJECT_ROOT, "Foundational/logs/Trains_autour_transi_2D")


def robust_derivative(y, x, step=1):
    """Dérivée robuste centrée (ou unilatérale aux bords)."""
    dy = np.zeros_like(y)
    n = len(y)
    for i in range(n):
        left = max(0, i - step)
        right = min(n - 1, i + step)
        if right == left:
            dy[i] = 0.0
        else:
            dy[i] = (y[right] - y[left]) / (x[right] - x[left])
    return np.abs(dy)


def merge_chunks(base_dir, L, dim):
    """
    Fusionne tous les chunks is_data pour un L donné.
    Aligne par valeur de h0 (robuste aux grilles hétérogènes entre chunks).
    """
    prefix = f"is_data_{dim}D_L{L}"
    folder = os.path.join(base_dir, f"L={L}")
    chunks = sorted(
        [f for f in os.listdir(folder) if f.startswith(prefix + "_chunk") and f.endswith(".npz")],
        key=lambda f: int(f.replace(prefix + "_chunk", "").replace(".npz", ""))
    )
    if not chunks:
        return None

    # Collecter toutes les valeurs h0 uniques à travers tous les chunks
    all_h0 = set()
    sigma_grid = None
    n_disorder = None
    for fname in chunks:
        c = np.load(os.path.join(folder, fname), allow_pickle=True)
        all_h0.update(np.round(c["h0_grid"], 8).tolist())
        if sigma_grid is None:
            sigma_grid = c["sigma_grid"].copy()
        if n_disorder is None:
            n_disorder = c["mz2_raw"].shape[2]

    h0_grid = np.array(sorted(all_h0))
    n_sigma = len(sigma_grid)
    n_h0 = len(h0_grid)

    mz2_raw = np.full((n_sigma, n_h0, n_disorder), np.nan)

    # Remplir par valeur h0
    for fname in chunks:
        c = np.load(os.path.join(folder, fname), allow_pickle=True)
        c_h0 = np.round(c["h0_grid"], 8)
        c_mz2 = c["mz2_raw"]
        for ih_c, h0_val in enumerate(c_h0):
            # Trouver l'indice dans la grille globale
            ih_global = np.searchsorted(h0_grid, h0_val)
            if ih_global >= n_h0 or not np.isclose(h0_grid[ih_global], h0_val):
                continue
            for s in range(n_sigma):
                if not np.isnan(c_mz2[s, ih_c, 0]):
                    mz2_raw[s, ih_global, :c_mz2.shape[2]] = c_mz2[s, ih_c, :]

    return h0_grid, sigma_grid, mz2_raw


def compute_max_suscept(h0_grid, mz2_raw, deriv_step=1):
    """
    Pour chaque sigma, calcule le max de la susceptibilité |d<mz²>/dh0|
    et l'incertitude associée via l'approche par config de désordre.

    Returns:
        max_mean : (n_sigma,) — moyenne du max sur les configs de désordre
        max_err  : (n_sigma,) — erreur standard (SEM)
    """
    n_sigma, n_h0, n_disorder = mz2_raw.shape
    max_mean = np.full(n_sigma, np.nan)
    max_err  = np.full(n_sigma, np.nan)

    for idx_s in range(n_sigma):
        # Approche 1 : dérivée de la courbe moyenne
        mean_curve = np.nanmean(mz2_raw[idx_s], axis=1)   # (n_h0,)
        # (utilisée pour vérification, mais l'incertitude vient des configs individuelles)

        # Approche 2 (pour l'incertitude) : max de la dérivée par config de désordre
        max_per_config = []
        for k in range(n_disorder):
            curve_k = mz2_raw[idx_s, :, k]
            valid = ~np.isnan(curve_k)
            if np.sum(valid) > 2 * deriv_step:
                h0_v = h0_grid[valid]
                mz2_v = curve_k[valid]
                dk = robust_derivative(mz2_v, h0_v, step=deriv_step)
                max_per_config.append(np.max(dk))

        if max_per_config:
            arr = np.array(max_per_config)
            max_mean[idx_s] = arr.mean()
            max_err[idx_s]  = arr.std(ddof=1) / np.sqrt(len(arr))

    return max_mean, max_err


def gather_data(base_dir, dim):
    """Collecte les données de tous les L disponibles pour une dimension."""
    L_dirs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith("L=") and os.path.isdir(os.path.join(base_dir, d))],
        key=lambda d: int(d.split("=")[1])
    )
    L_values = [int(d.split("=")[1]) for d in L_dirs]

    all_max_mean = []  # liste de (n_sigma,) pour chaque L
    all_max_err  = []
    sigma_grid_ref = None
    valid_L = []

    for L in L_values:
        result = merge_chunks(base_dir, L, dim)
        if result is None:
            print(f"  [WARN] Pas de données pour {dim}D L={L}")
            continue
        h0_grid, sigma_grid, mz2_raw = result

        # Vérifier que les données sont complètes
        nan_frac = np.mean(np.isnan(mz2_raw))
        if nan_frac > 0.05:
            print(f"  [WARN] {dim}D L={L}: {nan_frac*100:.1f}% NaN dans mz2_raw")

        if sigma_grid_ref is None:
            sigma_grid_ref = sigma_grid

        max_mean, max_err = compute_max_suscept(h0_grid, mz2_raw, deriv_step=1)
        all_max_mean.append(max_mean)
        all_max_err.append(max_err)
        valid_L.append(L)
        print(f"  {dim}D L={L}: max_suscept = {max_mean}")

    return (
        np.array(valid_L),
        sigma_grid_ref,
        np.array(all_max_mean),   # (n_L, n_sigma)
        np.array(all_max_err),    # (n_L, n_sigma)
    )


def make_plot(L_values, sigma_grid, max_mean, max_err, dim, output_dir):
    """Trace max(χ) vs L pour chaque sigma."""
    n_sigma = len(sigma_grid)
    colors = cm.viridis(np.linspace(0, 1, n_sigma))

    fig, ax = plt.subplots(figsize=(7, 5))

    for idx_s, sigma in enumerate(sigma_grid):
        c = colors[idx_s]
        y     = max_mean[:, idx_s]
        yerr  = max_err[:, idx_s]
        valid = ~np.isnan(y)
        ax.errorbar(
            L_values[valid], y[valid], yerr=yerr[valid],
            fmt='-o', color=c, markerfacecolor=c, markeredgecolor=c,
            linewidth=1.2, markersize=4, elinewidth=1.0, capsize=3,
            label=rf"$\sigma = {sigma}$",
        )

    hc_label = r"$h_c^\infty = 3.044$" if dim == 2 else r"$h_c^\infty = 1.0$"
    ax.set_xlabel(r"System size $L$", fontsize=12)
    ax.set_ylabel(
        r"$\max_{h_0} \left| \dfrac{\partial \langle M_z^2 \rangle}{\partial h_0} \right|$",
        fontsize=12,
    )
    dim_str = f"{dim}D"
    ax.set_title(
        rf"Maximum susceptibility vs $L$ — Ising {dim_str}",
        fontsize=13,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, ls="--", alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = os.path.join(output_dir, f"max_suscept_vs_L_{dim_str}.{ext}")
        plt.savefig(out, dpi=150)
        print(f"Sauvegardé : {out}")

    plt.close()


# ==========================================
# MAIN
# ==========================================
output_dir = os.path.join(PROJECT_ROOT, "Foundational/logs")

for dim, base_dir in [(1, BASE_1D), (2, BASE_2D)]:
    print(f"\n=== {dim}D ===")
    L_values, sigma_grid, max_mean, max_err = gather_data(base_dir, dim)
    if len(L_values) == 0:
        print("  Aucune donnée trouvée.")
        continue
    make_plot(L_values, sigma_grid, max_mean, max_err, dim, output_dir)

print("\nTerminé.")
