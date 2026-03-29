"""
Script de calcul et visualisation des exposants critiques.

Genere 3 figures :
  1. finite_size_scaling_{dim}D.png  — hc(L) et hauteur du pic en log-log (sigma=0)
  2. exponent_vs_disorder_{dim}D.png — alpha_N vs sigma
  3. fss_disorder_{dim}D.png         — finite-size scaling pour chaque sigma

Usage:
    python critical_exponents.py --dim 1
    python critical_exponents.py --dim 2
    python critical_exponents.py --dim 2 --no-outlier-filter
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="Critical exponents from IS data")
parser.add_argument("--dim", type=int, required=True, choices=[1, 2])
parser.add_argument("--deriv-step", type=int, default=3)
parser.add_argument("--sigma-max", type=float, default=0.3)
parser.add_argument("--no-outlier-filter", action="store_true", help="Desactive le filtrage IQR")
parser.add_argument("--exclude-L", type=int, nargs="+", default=[], help="Tailles a exclure du fit (ex: --exclude-L 81)")
args = parser.parse_args()

DIM = args.dim
DERIVATIVE_STEP = args.deriv_step

# ==========================================
# CHEMINS DES DONNEES
# ==========================================
PROJECT = "/users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates"

if DIM == 1:
    SIZES = [16, 25, 36]  # L>=49 donne des points absurdes
    DATA_FILES = {}
    base_1d = os.path.join(PROJECT, "Foundational/logs/Trains_finaux_disordered_1D")
    for L in SIZES:
        # Essayer les deux conventions de nommage
        p1 = os.path.join(base_1d, f"run_L={L}", f"is_data_1D_L{L}_full.npz")
        p2 = os.path.join(base_1d, f"run_L={L}", f"is_data_L{L}_full.npz")
        if os.path.exists(p1):
            DATA_FILES[L] = p1
        elif os.path.exists(p2):
            DATA_FILES[L] = p2
        else:
            print(f"  ATTENTION: pas de donnees pour L={L}")
    h0_min, h0_max = 0.5, 1.5
    hc_inf = 1.0
    hc_label = r"$h_c^\infty = 1.0$"
    expected_alpha = 0.75  # (1-2*beta)/(d*nu) pour 2D Ising
    alpha_label = r"2D Ising pur: $(1-2\beta)/\nu$ = 0.75"
    OUTPUT_DIR = base_1d
else:
    SIZES = [4, 5, 6, 7, 8, 9, 10]
    DATA_FILES = {}
    base_2d = os.path.join(PROJECT, "Foundational/rami_perso/2D_FNQS")
    for L in SIZES:
        p = os.path.join(base_2d, f"Run_2D_L{L}_FNQS", f"is_data_2D_L{L}_full.npz")
        if os.path.exists(p):
            DATA_FILES[L] = p
        else:
            print(f"  ATTENTION: pas de donnees pour L={L}")
    h0_min, h0_max = 2.0, 3.5
    hc_inf = 3.04
    hc_label = r"$h_c^\infty = 3.04$"
    expected_alpha = 0.275  # (1-2*beta)/(d*nu) pour 3D Ising
    alpha_label = r"3D Ising pur: $(1-2\beta)/(d\nu)$ = 0.275"
    OUTPUT_DIR = base_2d

dim_label = f"{DIM}D"
available_L = sorted(DATA_FILES.keys())
print(f"\n=== {dim_label} — Tailles disponibles: {available_L} ===\n")

# ==========================================
# FONCTIONS
# ==========================================
def robust_derivative(y, x, step=1):
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


def filter_outliers_iqr(arr, factor=1.5):
    """Filtre IQR: retourne array sans outliers et nombre retire."""
    if len(arr) < 5:
        return arr, 0
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    keep = (arr >= q1 - factor * iqr) & (arr <= q3 + factor * iqr)
    return arr[keep], np.sum(~keep)


# ==========================================
# CHARGEMENT ET CALCUL
# ==========================================
# Structure: results[L] = {sigma_grid, max_deriv_mean[sigma], max_deriv_err[sigma],
#                          peak_pos[sigma], h0_grid, mean_mz2[sigma]}
results = {}

for L in available_L:
    print(f"--- L={L} ---")
    data = np.load(DATA_FILES[L])
    sigma_grid = data["sigma_grid"]
    h0_grid = data["h0_grid"]
    mz2_raw = data["mz2_raw"]
    nb_spins = int(data["nb_spins"]) if "nb_spins" in data else L

    max_deriv_mean = []
    max_deriv_err = []
    peak_positions = []

    for idx_s, sigma in enumerate(sigma_grid):
        # Derivee de la moyenne (pour position du pic)
        mean_curve = np.nanmean(mz2_raw[idx_s], axis=1)
        deriv_mean = robust_derivative(mean_curve, h0_grid, step=DERIVATIVE_STEP)
        trans_mask = (h0_grid >= h0_min) & (h0_grid <= h0_max)
        if np.any(trans_mask):
            idx_peak = np.argmax(deriv_mean[trans_mask])
            peak_positions.append(h0_grid[trans_mask][idx_peak])
        else:
            peak_positions.append(np.nan)

        # Max derivee par realisation (micro)
        max_d_list = []
        for k in range(mz2_raw.shape[2]):
            curve_k = mz2_raw[idx_s, :, k]
            valid = ~np.isnan(curve_k)
            if np.sum(valid) > 2 * DERIVATIVE_STEP:
                dk = robust_derivative(curve_k[valid], h0_grid[valid], step=DERIVATIVE_STEP)
                tm = (h0_grid[valid] >= h0_min) & (h0_grid[valid] <= h0_max)
                if np.any(tm):
                    max_d_list.append(np.max(dk[tm]))

        arr = np.array(max_d_list)

        # Filtrage outliers
        if not args.no_outlier_filter:
            arr, n_removed = filter_outliers_iqr(arr)
            if n_removed > 0:
                print(f"  sigma={sigma:.3f}: {n_removed} outliers retires ({len(arr)} restants)")

        max_deriv_mean.append(np.mean(arr) if len(arr) > 0 else np.nan)
        max_deriv_err.append(np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0)

    results[L] = {
        "sigma_grid": sigma_grid,
        "nb_spins": nb_spins,
        "max_deriv_mean": np.array(max_deriv_mean),
        "max_deriv_err": np.array(max_deriv_err),
        "peak_positions": np.array(peak_positions),
    }

# ==========================================
# sigma_grid commun
# ==========================================
sigma_grid = results[available_L[0]]["sigma_grid"]
sigma_mask = sigma_grid <= args.sigma_max

# ==========================================
# FIGURE 1: Finite-Size Scaling (sigma=0)
# ==========================================
print("\n--- Figure 1: Finite-Size Scaling (sigma=0) ---")

L_arr = np.array(available_L, dtype=float)
N_arr = np.array([results[L]["nb_spins"] for L in available_L], dtype=float)  # N = nb_spins reel

# Masque pour exclure certaines tailles du fit
fit_mask = np.array([L not in args.exclude_L for L in available_L])
if args.exclude_L:
    print(f"  Tailles exclues du fit: {args.exclude_L}")

# Trouver index sigma=0
idx_s0 = np.argmin(np.abs(sigma_grid - 0.0))

hc_vals = np.array([results[L]["peak_positions"][idx_s0] for L in available_L])
height_vals = np.array([results[L]["max_deriv_mean"][idx_s0] for L in available_L])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panneau gauche: hc(L)
ax1.plot(L_arr, hc_vals, 'ro', markersize=8, label="Donnees")

# Fit hc(L) = hc_inf - a * L^(-1/nu)
def hc_fit(L, hc_inf_fit, a, inv_nu):
    return hc_inf_fit - a * L**(-inv_nu)

try:
    p0 = [hc_inf, 1.0, 1.0]
    popt, pcov = curve_fit(hc_fit, L_arr[fit_mask], hc_vals[fit_mask], p0=p0, maxfev=10000)
    L_fine = np.linspace(min(L_arr) * 0.8, max(L_arr) * 1.5, 200)
    ax1.plot(L_fine, hc_fit(L_fine, *popt), 'b--',
             label=rf"Fit: $h_c^\infty$={popt[0]:.2f}, $1/\nu$={popt[2]:.2f}")
    print(f"  hc_inf = {popt[0]:.3f}, a = {popt[1]:.3f}, 1/nu = {popt[2]:.3f}")
except Exception as e:
    print(f"  Fit hc(L) echoue: {e}")

ax1.axhline(y=hc_inf, color='gray', linestyle=':', alpha=0.5, label=f"$h_c^\\infty$ = {hc_inf} (litt.)")
ax1.set_xlabel("L")
ax1.set_ylabel(r"$h_c(L)$")
ax1.set_title("Position du pic de susceptibilite")
ax1.legend()
ax1.grid(True, ls="--", alpha=0.3)

# Panneau droit: hauteur du pic en log-log (vs N = nb_spins)
ax2.plot(np.log(N_arr), np.log(height_vals), 'ro', markersize=8, label="Donnees")

# Fit lineaire log(height) = alpha * log(N) + b
valid = ~np.isnan(height_vals) & (height_vals > 0) & fit_mask
if np.sum(valid) >= 2:
    coeffs = np.polyfit(np.log(N_arr[valid]), np.log(height_vals[valid]), 1)
    alpha_fit = coeffs[0]
    log_N_fine = np.linspace(np.log(min(N_arr)) - 0.2, np.log(max(N_arr)) + 0.5, 100)
    ax2.plot(log_N_fine, np.polyval(coeffs, log_N_fine), 'b--',
             label=rf"Fit: $N^{{{alpha_fit:.2f}}}$ (attendu {expected_alpha:.2f})")
    print(f"  alpha (sigma=0) = {alpha_fit:.3f}")

ax2.set_xlabel(r"$\ln(N)$")
ax2.set_ylabel(r"$\ln(\max\ |dM_z^2/dh|)$")
ax2.set_title("Hauteur du pic (log-log)")
ax2.legend()
ax2.grid(True, ls="--", alpha=0.3)

class_label = "3D Ising" if DIM == 2 else "2D Ising"
fig.suptitle(f"Finite-Size Scaling — {dim_label} Ising Transverse (classe {class_label})",
             fontsize=13, fontweight='bold')
plt.tight_layout()

out1 = os.path.join(OUTPUT_DIR, f"finite_size_scaling_{dim_label}.png")
plt.savefig(out1, dpi=150)
plt.savefig(out1.replace(".png", ".pdf"))
print(f"  -> {out1}")

# ==========================================
# FIGURE 2: Exposant critique vs desordre
# ==========================================
print("\n--- Figure 2: Exposant alpha_N vs sigma ---")

alpha_vs_sigma = []
alpha_err_vs_sigma = []

for idx_s in range(len(sigma_grid)):
    if not sigma_mask[idx_s]:
        alpha_vs_sigma.append(np.nan)
        alpha_err_vs_sigma.append(np.nan)
        continue

    log_N = []
    log_h = []
    for L in available_L:
        if L in args.exclude_L:
            continue
        h = results[L]["max_deriv_mean"][idx_s]
        N = results[L]["nb_spins"]
        if not np.isnan(h) and h > 0:
            log_N.append(np.log(N))
            log_h.append(np.log(h))

    if len(log_N) >= 3:
        log_N = np.array(log_N)
        log_h = np.array(log_h)
        coeffs, cov = np.polyfit(log_N, log_h, 1, cov=True)
        alpha_vs_sigma.append(coeffs[0])
        alpha_err_vs_sigma.append(np.sqrt(cov[0, 0]))
    else:
        alpha_vs_sigma.append(np.nan)
        alpha_err_vs_sigma.append(np.nan)

alpha_vs_sigma = np.array(alpha_vs_sigma)
alpha_err_vs_sigma = np.array(alpha_err_vs_sigma)

fig, ax = plt.subplots(figsize=(8, 6))
valid = sigma_mask & ~np.isnan(alpha_vs_sigma)
ax.errorbar(sigma_grid[valid], alpha_vs_sigma[valid], yerr=alpha_err_vs_sigma[valid],
            fmt='-o', color='red', markerfacecolor='red', markeredgecolor='red',
            linewidth=1.0, markersize=6, elinewidth=1.0, capsize=3)
ax.axhline(y=expected_alpha, color='blue', linestyle='--', alpha=0.7, label=alpha_label)
ax.set_xlabel(r"Disorder strength $\sigma$")
ax.set_ylabel(r"Exposant $\alpha_N$ ($\max|dM_z^2/dh| \sim N^{\alpha_N}$)")
ax.set_title(f"{dim_label} — Exposant critique vs desordre")
ax.legend(fontsize=10)
ax.grid(True, ls="--", alpha=0.3)
plt.tight_layout()

out2 = os.path.join(OUTPUT_DIR, f"exponent_vs_disorder_{dim_label}.png")
plt.savefig(out2, dpi=150)
plt.savefig(out2.replace(".png", ".pdf"))
print(f"  -> {out2}")

# ==========================================
# FIGURE 3: FSS par sigma (log-log multi-sigma)
# ==========================================
print("\n--- Figure 3: FSS pour chaque sigma ---")

fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.viridis(np.linspace(0, 1, int(np.sum(sigma_mask))))

c_idx = 0
for idx_s, sigma in enumerate(sigma_grid):
    if not sigma_mask[idx_s]:
        continue

    log_N = []
    log_h = []
    for L in available_L:
        h = results[L]["max_deriv_mean"][idx_s]
        N = results[L]["nb_spins"]
        if not np.isnan(h) and h > 0:
            log_N.append(np.log(N))
            log_h.append(np.log(h))

    if len(log_N) >= 2:
        ax.plot(log_N, log_h, 'o-', color=colors[c_idx], markersize=5,
                label=rf"$\sigma={sigma:.2f}$")
    c_idx += 1

ax.set_xlabel(r"$\ln(N)$")
ax.set_ylabel(r"$\ln(\max\ |dM_z^2/dh|)$")
ax.set_title(f"{dim_label} — Finite-Size Scaling par desordre")
ax.legend(fontsize=9)
ax.grid(True, ls="--", alpha=0.3)
plt.tight_layout()

out3 = os.path.join(OUTPUT_DIR, f"fss_disorder_{dim_label}.png")
plt.savefig(out3, dpi=150)
plt.savefig(out3.replace(".png", ".pdf"))
print(f"  -> {out3}")

print("\nTermine.")
