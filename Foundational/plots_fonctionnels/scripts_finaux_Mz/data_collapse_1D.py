"""
Extraction des exposants critiques par data collapse de Mz^2.

Principe: au voisinage de hc, le finite-size scaling predit
    Mz^2(h, L) = L^{-2*beta/nu} * f( (h - hc) * L^{1/nu} )

On optimise (hc, 1/nu, 2*beta/nu) pour minimiser la dispersion
des courbes rescalees.

Usage:
    python data_collapse_1D.py
    python data_collapse_1D.py --sigma-idx 0
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# ==========================================
# ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--sigma-idx", type=int, default=None)
parser.add_argument("--h-range", type=float, nargs=2, default=[0.6, 1.4],
                    help="Fenetre en h pour le collapse")
args = parser.parse_args()

# ==========================================
# CHARGEMENT
# ==========================================
PROJECT = "/users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates"
base = os.path.join(PROJECT, "Foundational/logs/Trains_finaux_disordered_1D")

SIZES = [16, 25, 36]
data_all = {}
for L in SIZES:
    p1 = os.path.join(base, f"run_L={L}", f"is_data_1D_L{L}_full.npz")
    p2 = os.path.join(base, f"run_L={L}", f"is_data_L{L}_full.npz")
    p = p1 if os.path.exists(p1) else p2
    data_all[L] = np.load(p)

sigma_grid = data_all[SIZES[0]]["sigma_grid"]
h0_grid = data_all[SIZES[0]]["h0_grid"]

if args.sigma_idx is not None:
    sigma_indices = [args.sigma_idx]
else:
    sigma_indices = [i for i, s in enumerate(sigma_grid) if s <= 0.3]


# ==========================================
# DATA COLLAPSE
# ==========================================
def collapse_cost(params, sizes, h_grid, curves, h_min, h_max):
    """
    Cout du data collapse.
    params = [hc, inv_nu, two_beta_over_nu]

    On rescale:
      x = (h - hc) * L^{1/nu}
      y = Mz^2 * L^{2*beta/nu}

    Puis on mesure la dispersion entre courbes interpolees
    sur un x-grid commun.
    """
    hc, inv_nu, two_beta_nu = params

    # Bornes de securite
    if hc < 0.5 or hc > 1.5:
        return 1e10
    if inv_nu < 0.1 or inv_nu > 5.0:
        return 1e10
    if two_beta_nu < -2.0 or two_beta_nu > 2.0:
        return 1e10

    # Rescaler chaque courbe
    rescaled = []
    for L, curve in zip(sizes, curves):
        mask = (h_grid >= h_min) & (h_grid <= h_max) & ~np.isnan(curve)
        if np.sum(mask) < 5:
            return 1e10
        h_sel = h_grid[mask]
        mz2_sel = curve[mask]

        x = (h_sel - hc) * L ** inv_nu
        y = mz2_sel * L ** two_beta_nu

        rescaled.append((x, y))

    # Trouver x-range commun
    x_min = max(r[0].min() for r in rescaled)
    x_max = min(r[0].max() for r in rescaled)
    if x_min >= x_max:
        return 1e10

    x_common = np.linspace(x_min, x_max, 100)

    # Interpoler chaque courbe sur x_common
    y_interp = []
    for x, y in rescaled:
        # Trier par x
        order = np.argsort(x)
        x_s, y_s = x[order], y[order]
        # Supprimer doublons en x
        _, unique_idx = np.unique(x_s, return_index=True)
        x_s, y_s = x_s[unique_idx], y_s[unique_idx]
        if len(x_s) < 3:
            return 1e10
        f = interp1d(x_s, y_s, kind='linear', fill_value='extrapolate')
        y_interp.append(f(x_common))

    y_interp = np.array(y_interp)

    # Cout = variance moyenne entre courbes a chaque x
    y_mean = np.mean(y_interp, axis=0)
    cost = np.mean(np.sum((y_interp - y_mean[None, :]) ** 2, axis=0))

    return cost


# ==========================================
# ANALYSE
# ==========================================
# Valeurs attendues 1D Ising transverse (classe 2D Ising):
#   hc = 1.0, nu = 1.0, beta = 1/8
#   -> inv_nu = 1.0, 2*beta/nu = 0.25
EXPECTED = {"hc": 1.0, "inv_nu": 1.0, "two_beta_nu": 0.25}

results = {}

for idx_s in sigma_indices:
    sigma = sigma_grid[idx_s]
    print(f"\n=== sigma = {sigma:.3f} ===")

    curves = [np.nanmean(data_all[L]["mz2_raw"][idx_s], axis=1) for L in SIZES]

    # Optimisation multi-start
    best_cost = 1e10
    best_params = None

    starts = [
        [1.0, 1.0, 0.25],    # valeurs exactes
        [0.95, 0.8, 0.2],
        [1.05, 1.2, 0.3],
        [1.0, 0.5, 0.1],
        [1.0, 1.5, 0.5],
        [0.9, 1.0, 0.25],
        [1.1, 1.0, 0.25],
    ]

    for p0 in starts:
        res = minimize(collapse_cost, p0,
                       args=(SIZES, h0_grid, curves, args.h_range[0], args.h_range[1]),
                       method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-5, 'fatol': 1e-10})
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x

    hc, inv_nu, two_beta_nu = best_params
    nu = 1.0 / inv_nu
    beta_over_nu = two_beta_nu / 2

    print(f"  hc       = {hc:.4f}  (attendu {EXPECTED['hc']:.3f})")
    print(f"  1/nu     = {inv_nu:.4f}  (attendu {EXPECTED['inv_nu']:.3f})")
    print(f"  nu       = {nu:.4f}  (attendu 1.0)")
    print(f"  2beta/nu = {two_beta_nu:.4f}  (attendu {EXPECTED['two_beta_nu']:.3f})")
    print(f"  beta/nu  = {beta_over_nu:.4f}  (attendu 0.125)")
    print(f"  cout     = {best_cost:.2e}")

    results[sigma] = {
        "hc": hc, "inv_nu": inv_nu, "nu": nu,
        "two_beta_nu": two_beta_nu, "beta_over_nu": beta_over_nu,
        "cost": best_cost,
    }


# ==========================================
# FIGURE 1: Data collapse pour chaque sigma
# ==========================================
print("\n--- Figure 1: Data collapse ---")

n_sigma = len(sigma_indices)
ncols = min(n_sigma, 3)
nrows = (n_sigma + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

colors_L = {16: 'C0', 25: 'C1', 36: 'C2'}

for i, idx_s in enumerate(sigma_indices):
    ax = axes[i // ncols][i % ncols]
    sigma = sigma_grid[idx_s]
    if sigma not in results:
        continue

    res = results[sigma]
    hc = res["hc"]
    inv_nu = res["inv_nu"]
    two_beta_nu = res["two_beta_nu"]

    for L in SIZES:
        curve = np.nanmean(data_all[L]["mz2_raw"][idx_s], axis=1)
        mask = (h0_grid >= args.h_range[0]) & (h0_grid <= args.h_range[1])
        h_sel = h0_grid[mask]
        mz2_sel = curve[mask]

        x = (h_sel - hc) * L ** inv_nu
        y = mz2_sel * L ** two_beta_nu

        ax.plot(x, y, '-', color=colors_L[L], linewidth=1.5, label=f"L={L}")

    ax.set_xlabel(r"$(h - h_c) \cdot L^{1/\nu}$")
    ax.set_ylabel(r"$M_z^2 \cdot L^{2\beta/\nu}$")
    ax.set_title(rf"$\sigma={sigma:.2f}$ — $h_c$={hc:.3f}, $\nu$={res['nu']:.2f}, "
                 rf"$\beta/\nu$={res['beta_over_nu']:.3f}")
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.3)

# Cacher axes vides
for i in range(n_sigma, nrows * ncols):
    axes[i // ncols][i % ncols].set_visible(False)

fig.suptitle("Data Collapse de $M_z^2$ — 1D Ising Transverse",
             fontsize=14, fontweight='bold')
plt.tight_layout()

out1 = os.path.join(base, "data_collapse_1D.png")
plt.savefig(out1, dpi=150)
plt.savefig(out1.replace(".png", ".pdf"))
print(f"  -> {out1}")


# ==========================================
# FIGURE 2: Avant/apres collapse (sigma=0)
# ==========================================
if 0.0 in results:
    print("\n--- Figure 2: Avant/apres (sigma=0) ---")
    res = results[0.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Avant
    for L in SIZES:
        curve = np.nanmean(data_all[L]["mz2_raw"][0], axis=1)
        mask = (h0_grid >= args.h_range[0]) & (h0_grid <= args.h_range[1])
        ax1.plot(h0_grid[mask], curve[mask], '-', color=colors_L[L],
                 linewidth=1.5, label=f"L={L}")

    ax1.axvline(res["hc"], color='red', ls='--', alpha=0.5,
                label=f"$h_c$ = {res['hc']:.3f}")
    ax1.set_xlabel("$h$")
    ax1.set_ylabel("$M_z^2$")
    ax1.set_title("Donnees brutes")
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.3)

    # Apres
    for L in SIZES:
        curve = np.nanmean(data_all[L]["mz2_raw"][0], axis=1)
        mask = (h0_grid >= args.h_range[0]) & (h0_grid <= args.h_range[1])
        h_sel = h0_grid[mask]
        mz2_sel = curve[mask]

        x = (h_sel - res["hc"]) * L ** res["inv_nu"]
        y = mz2_sel * L ** res["two_beta_nu"]

        ax2.plot(x, y, '-', color=colors_L[L], linewidth=1.5, label=f"L={L}")

    ax2.set_xlabel(r"$(h - h_c) \cdot L^{1/\nu}$")
    ax2.set_ylabel(r"$M_z^2 \cdot L^{2\beta/\nu}$")
    ax2.set_title(f"Data collapse ($\\nu$={res['nu']:.2f}, "
                  f"$\\beta/\\nu$={res['beta_over_nu']:.3f})")
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.3)

    fig.suptitle(f"Data Collapse — $\\sigma=0$, $h_c$={res['hc']:.3f}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out2 = os.path.join(base, "data_collapse_before_after_1D.png")
    plt.savefig(out2, dpi=150)
    plt.savefig(out2.replace(".png", ".pdf"))
    print(f"  -> {out2}")


# ==========================================
# FIGURE 3: Exposants vs desordre
# ==========================================
if len(results) >= 2:
    print("\n--- Figure 3: Exposants vs desordre ---")

    sigmas = sorted(results.keys())
    hc_arr = [results[s]["hc"] for s in sigmas]
    nu_arr = [results[s]["nu"] for s in sigmas]
    beta_nu_arr = [results[s]["beta_over_nu"] for s in sigmas]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.plot(sigmas, hc_arr, 'ro-', markersize=7)
    ax1.axhline(1.0, color='gray', ls=':', alpha=0.5, label="$h_c=1.0$ (exact)")
    ax1.set_xlabel(r"$\sigma$")
    ax1.set_ylabel(r"$h_c$")
    ax1.set_title("Point critique")
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.3)

    ax2.plot(sigmas, nu_arr, 'bo-', markersize=7)
    ax2.axhline(1.0, color='gray', ls=':', alpha=0.5, label=r"$\nu=1.0$ (2D Ising)")
    ax2.set_xlabel(r"$\sigma$")
    ax2.set_ylabel(r"$\nu$")
    ax2.set_title(r"Exposant $\nu$")
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.3)

    ax3.plot(sigmas, beta_nu_arr, 'go-', markersize=7)
    ax3.axhline(0.125, color='gray', ls=':', alpha=0.5, label=r"$\beta/\nu=0.125$ (2D Ising)")
    ax3.set_xlabel(r"$\sigma$")
    ax3.set_ylabel(r"$\beta/\nu$")
    ax3.set_title(r"Exposant $\beta/\nu$")
    ax3.legend()
    ax3.grid(True, ls='--', alpha=0.3)

    fig.suptitle("Exposants critiques vs desordre — Data Collapse 1D",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out3 = os.path.join(base, "exponents_vs_disorder_collapse_1D.png")
    plt.savefig(out3, dpi=150)
    plt.savefig(out3.replace(".png", ".pdf"))
    print(f"  -> {out3}")

print("\nTermine.")
