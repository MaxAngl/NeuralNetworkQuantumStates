"""
Extraction de hc et des exposants critiques par la methode du crossing de Mz^2.

1. Plot Mz^2(h) pour chaque L -> reperage visuel du crossing
2. Estimation de hc par intersection des courbes interpolees
3. Extraction de 1/nu via la pente dMz^2/dh au crossing
4. Extraction de 2*beta/nu via Mz^2(hc) vs L

Usage:
    python crossing_exponents_1D.py
    python crossing_exponents_1D.py --sigma-idx 0   (sigma=0 seulement)
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# ==========================================
# ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--sigma-idx", type=int, default=None,
                    help="Index sigma a analyser (defaut: tous <= 0.3)")
parser.add_argument("--deriv-step", type=int, default=3)
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

# Sigma indices a traiter
if args.sigma_idx is not None:
    sigma_indices = [args.sigma_idx]
else:
    sigma_indices = [i for i, s in enumerate(sigma_grid) if s <= 0.3]

# ==========================================
# FONCTIONS
# ==========================================
def mean_mz2(L, sigma_idx):
    """Mz^2 moyen sur les realisations de desordre."""
    return np.nanmean(data_all[L]["mz2_raw"][sigma_idx], axis=1)


def find_crossing(h_grid, curve1, curve2, h_min=0.5, h_max=1.5):
    """Trouve le point de croisement entre deux courbes par interpolation."""
    f1 = interp1d(h_grid, curve1, kind='cubic')
    f2 = interp1d(h_grid, curve2, kind='cubic')

    # Difference
    def diff(h):
        return f1(h) - f2(h)

    # Chercher un changement de signe dans [h_min, h_max]
    h_search = np.linspace(h_min, h_max, 500)
    d_vals = [diff(h) for h in h_search]

    crossings = []
    for i in range(len(d_vals) - 1):
        if d_vals[i] * d_vals[i + 1] < 0:
            hc = brentq(diff, h_search[i], h_search[i + 1])
            crossings.append(hc)

    return crossings


def slope_at_hc(h_grid, curve, hc, step=3):
    """Pente de Mz^2(h) au voisinage de hc par interpolation."""
    f = interp1d(h_grid, curve, kind='cubic')
    dh = (h_grid[1] - h_grid[0]) * step
    h_left = max(h_grid[0], hc - dh)
    h_right = min(h_grid[-1], hc + dh)
    return abs((f(h_right) - f(h_left)) / (h_right - h_left))


def mz2_at_hc(h_grid, curve, hc):
    """Mz^2 interpole au point hc."""
    f = interp1d(h_grid, curve, kind='cubic')
    return f(hc)


# ==========================================
# ANALYSE
# ==========================================
results = {}

for idx_s in sigma_indices:
    sigma = sigma_grid[idx_s]
    print(f"\n=== sigma = {sigma:.3f} ===")

    curves = {L: mean_mz2(L, idx_s) for L in SIZES}

    # Trouver les crossings entre toutes les paires
    all_hc = []
    pairs = [(SIZES[i], SIZES[j]) for i in range(len(SIZES))
             for j in range(i + 1, len(SIZES))]

    for L1, L2 in pairs:
        xings = find_crossing(h0_grid, curves[L1], curves[L2])
        if xings:
            hc = xings[0]  # premier crossing
            all_hc.append(hc)
            print(f"  Crossing L={L1}/L={L2}: hc = {hc:.4f}")
        else:
            print(f"  Crossing L={L1}/L={L2}: pas trouve")

    if len(all_hc) < 2:
        print("  Pas assez de crossings, skip.")
        continue

    hc_mean = np.mean(all_hc)
    hc_std = np.std(all_hc)
    print(f"  => hc = {hc_mean:.4f} +/- {hc_std:.4f}")

    # Pentes et valeurs au crossing
    slopes = {}
    mz2_vals = {}
    for L in SIZES:
        slopes[L] = slope_at_hc(h0_grid, curves[L], hc_mean, step=args.deriv_step)
        mz2_vals[L] = mz2_at_hc(h0_grid, curves[L], hc_mean)
        print(f"  L={L}: pente = {slopes[L]:.4f}, Mz^2(hc) = {mz2_vals[L]:.4f}")

    results[sigma] = {
        "hc": hc_mean, "hc_err": hc_std,
        "slopes": slopes, "mz2_vals": mz2_vals,
    }

# ==========================================
# FIGURE 1: Crossing des courbes Mz^2(h)
# ==========================================
print("\n--- Figure 1: Crossing ---")

n_sigma = len(sigma_indices)
fig, axes = plt.subplots(1, min(n_sigma, 4), figsize=(5 * min(n_sigma, 4), 5),
                         squeeze=False)
axes = axes[0]

colors_L = {16: 'C0', 25: 'C1', 36: 'C2'}

for i, idx_s in enumerate(sigma_indices[:4]):
    ax = axes[i]
    sigma = sigma_grid[idx_s]

    for L in SIZES:
        curve = mean_mz2(L, idx_s)
        ax.plot(h0_grid, curve, '-', color=colors_L[L], linewidth=1.5,
                label=f"L={L}")

    if sigma in results:
        hc = results[sigma]["hc"]
        ax.axvline(hc, color='red', ls='--', alpha=0.7,
                   label=f"$h_c$ = {hc:.3f}")

    ax.set_xlabel("$h_0$")
    ax.set_ylabel("$M_z^2$")
    ax.set_title(f"$\\sigma = {sigma:.2f}$")
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, 1.8)
    ax.grid(True, ls='--', alpha=0.3)

fig.suptitle("Crossing de $M_z^2$ — 1D Ising Transverse", fontweight='bold')
plt.tight_layout()
out1 = os.path.join(base, "crossing_Mz2_1D.png")
plt.savefig(out1, dpi=150)
plt.savefig(out1.replace(".png", ".pdf"))
print(f"  -> {out1}")

# ==========================================
# FIGURE 2: Extraction des exposants (sigma=0)
# ==========================================
sigma0 = 0.0
if sigma0 in results:
    print("\n--- Figure 2: Exposants critiques (sigma=0) ---")
    res = results[sigma0]

    L_arr = np.array(SIZES, dtype=float)
    slope_arr = np.array([res["slopes"][L] for L in SIZES])
    mz2_arr = np.array([res["mz2_vals"][L] for L in SIZES])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panneau gauche: 1/nu depuis les pentes
    log_L = np.log(L_arr)
    log_slope = np.log(slope_arr)

    ax1.plot(log_L, log_slope, 'ro', markersize=10, zorder=5)

    if len(log_L) >= 2:
        coeffs = np.polyfit(log_L, log_slope, 1)
        inv_nu = coeffs[0]
        L_fine = np.linspace(log_L[0] - 0.1, log_L[-1] + 0.3, 100)
        ax1.plot(L_fine, np.polyval(coeffs, L_fine), 'b--',
                 label=rf"Fit: $1/\nu$ = {inv_nu:.3f}")
        # Valeur theorique 1D Ising: nu = 1, donc 1/nu = 1
        ax1.plot(L_fine, 1.0 * L_fine + (log_slope[0] - 1.0 * log_L[0]),
                 'g:', alpha=0.5, label=r"Attendu: $1/\nu = 1.0$")
        print(f"  1/nu = {inv_nu:.3f} (attendu 1.0)")

    ax1.set_xlabel(r"$\ln(L)$")
    ax1.set_ylabel(r"$\ln(dM_z^2/dh|_{h_c})$")
    ax1.set_title(r"Extraction de $1/\nu$")
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.3)

    # Panneau droit: 2*beta/nu depuis Mz^2(hc)
    log_mz2 = np.log(mz2_arr)

    ax2.plot(log_L, log_mz2, 'ro', markersize=10, zorder=5)

    if len(log_L) >= 2:
        coeffs2 = np.polyfit(log_L, log_mz2, 1)
        minus_2beta_nu = coeffs2[0]
        ax2.plot(L_fine, np.polyval(coeffs2, L_fine), 'b--',
                 label=rf"Fit: pente = {minus_2beta_nu:.3f}")
        # 1D Ising: beta = 1/8, nu = 1 -> -2*beta/nu = -0.25
        ax2.plot(L_fine, -0.25 * L_fine + (log_mz2[0] + 0.25 * log_L[0]),
                 'g:', alpha=0.5, label=r"Attendu: $-2\beta/\nu = -0.25$")
        beta_over_nu = -minus_2beta_nu / 2
        print(f"  -2*beta/nu = {minus_2beta_nu:.3f} (attendu -0.25)")
        print(f"  => beta/nu = {beta_over_nu:.3f} (attendu 0.125)")

    ax2.set_xlabel(r"$\ln(L)$")
    ax2.set_ylabel(r"$\ln(M_z^2(h_c))$")
    ax2.set_title(r"Extraction de $\beta/\nu$")
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.3)

    hc = res["hc"]
    fig.suptitle(f"Exposants critiques — 1D Ising ($h_c$ = {hc:.3f}, $\\sigma=0$)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out2 = os.path.join(base, "critical_exponents_crossing_1D.png")
    plt.savefig(out2, dpi=150)
    plt.savefig(out2.replace(".png", ".pdf"))
    print(f"  -> {out2}")

# ==========================================
# FIGURE 3: hc et exposants vs desordre
# ==========================================
if len(results) >= 2:
    print("\n--- Figure 3: hc et exposants vs desordre ---")

    sigmas = sorted(results.keys())
    hc_arr = np.array([results[s]["hc"] for s in sigmas])
    hc_err_arr = np.array([results[s]["hc_err"] for s in sigmas])

    inv_nu_arr = []
    inv_nu_err_arr = []
    beta_nu_arr = []

    for s in sigmas:
        res = results[s]
        L_arr = np.array(SIZES, dtype=float)
        log_L = np.log(L_arr)

        slope_arr = np.array([res["slopes"][L] for L in SIZES])
        coeffs, cov = np.polyfit(log_L, np.log(slope_arr), 1, cov=True)
        inv_nu_arr.append(coeffs[0])
        inv_nu_err_arr.append(np.sqrt(cov[0, 0]))

        mz2_arr = np.array([res["mz2_vals"][L] for L in SIZES])
        coeffs2 = np.polyfit(log_L, np.log(mz2_arr), 1)
        beta_nu_arr.append(-coeffs2[0] / 2)

    inv_nu_arr = np.array(inv_nu_arr)
    inv_nu_err_arr = np.array(inv_nu_err_arr)
    beta_nu_arr = np.array(beta_nu_arr)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # hc vs sigma
    ax1.errorbar(sigmas, hc_arr, yerr=hc_err_arr, fmt='-o', color='red',
                 capsize=3, markersize=6)
    ax1.axhline(1.0, color='gray', ls=':', alpha=0.5, label="$h_c = 1.0$ (exact)")
    ax1.set_xlabel(r"$\sigma$")
    ax1.set_ylabel(r"$h_c$")
    ax1.set_title("Position critique")
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.3)

    # 1/nu vs sigma
    ax2.errorbar(sigmas, inv_nu_arr, yerr=inv_nu_err_arr, fmt='-o', color='blue',
                 capsize=3, markersize=6)
    ax2.axhline(1.0, color='gray', ls=':', alpha=0.5, label=r"$1/\nu = 1.0$ (2D Ising)")
    ax2.set_xlabel(r"$\sigma$")
    ax2.set_ylabel(r"$1/\nu$")
    ax2.set_title(r"Exposant $1/\nu$")
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.3)

    # beta/nu vs sigma
    ax3.plot(sigmas, beta_nu_arr, '-o', color='green', markersize=6)
    ax3.axhline(0.125, color='gray', ls=':', alpha=0.5, label=r"$\beta/\nu = 0.125$ (2D Ising)")
    ax3.set_xlabel(r"$\sigma$")
    ax3.set_ylabel(r"$\beta/\nu$")
    ax3.set_title(r"Exposant $\beta/\nu$")
    ax3.legend()
    ax3.grid(True, ls='--', alpha=0.3)

    fig.suptitle("Exposants critiques vs desordre — 1D Ising (crossing method)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out3 = os.path.join(base, "exponents_vs_disorder_crossing_1D.png")
    plt.savefig(out3, dpi=150)
    plt.savefig(out3.replace(".png", ".pdf"))
    print(f"  -> {out3}")

print("\nTermine.")
