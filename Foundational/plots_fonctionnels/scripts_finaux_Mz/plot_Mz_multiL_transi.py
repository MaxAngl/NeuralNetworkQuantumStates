"""
Plot multi-L des résultats IS pour Trains_autour_transi_1D.

Pour chaque sigma, trace :
  Panneau 1 : Mz² vs h0 pour toutes les tailles L (avec barres d'erreur totales)
  Panneau 2 : |dMz²/dh0| vs h0 pour toutes les tailles L
  Panneau 3 : max(|dMz²/dh0|) vs L (pic de susceptibilité)

Barres d'erreur sur Mz² = erreur totale :
  sigma=0  : erreur MC pure (estimateur IS, w_norm corrects)
  sigma>0  : sqrt(sigma_MC^2 + sigma_desordre^2 / N_d)
  Note : sigma_MC n'est PAS divisée par sqrt(N_d) car toutes les configs
  partagent les mêmes échantillons MCMC (corrélation totale).

Usage:
    python plot_Mz_multiL_transi.py [--sigma 0.0] [--deriv-step 2] [--output-dir DIR]
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=0.0, help="Valeur de sigma a tracer (defaut: 0.0)")
parser.add_argument("--deriv-step", type=int, default=2, help="Pas pour derivee robuste (defaut: 2)")
parser.add_argument("--output-dir", type=str, default=None, help="Dossier de sortie")
args = parser.parse_args()

PROJECT = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
BASE_DIR = os.path.join(PROJECT, "Foundational/logs/Trains_autour_transi_1D")
L_LIST = [16, 24, 36, 48, 64, 80]
DERIVATIVE_STEP = args.deriv_step
TARGET_SIGMA = args.sigma
output_dir = args.output_dir or BASE_DIR

H0_MIN, H0_MAX = 0.8, 1.2
HC_INF = 1.0


def robust_derivative(y, x, step=1):
    dy = np.zeros_like(y)
    n = len(y)
    for i in range(n):
        left = max(0, i - step)
        right = min(n - 1, i + step)
        dy[i] = 0.0 if right == left else (y[right] - y[left]) / (x[right] - x[left])
    return np.abs(dy)


# ==========================================
# CHARGEMENT
# ==========================================
data_by_L = {}
for L in L_LIST:
    path = os.path.join(BASE_DIR, f"L={L}", f"is_data_1D_L{L}_full.npz")
    if not os.path.exists(path):
        print(f"Fichier manquant pour L={L}: {path}")
        continue
    data = np.load(path)
    sigma_grid = data["sigma_grid"]
    idx_s = int(np.argmin(np.abs(sigma_grid - TARGET_SIGMA)))
    actual_sigma = float(sigma_grid[idx_s])
    h0_grid = data["h0_grid"]
    mz2_raw = data["mz2_raw"]          # (n_sigma, n_h0, n_disorder)
    has_err = "mz2_err_raw" in data
    mz2_err_raw = data["mz2_err_raw"] if has_err else None

    N_d = mz2_raw.shape[2]

    # Moyenne sur le désordre (axe k)
    mz2_mean = np.nanmean(mz2_raw[idx_s], axis=1)   # (n_h0,)
    mz2_std  = np.nanstd(mz2_raw[idx_s], axis=1, ddof=1)  # std désordre

    # Erreur totale :
    #   sigma=0 : erreur MC (tous k identiques, prendre k=0)
    #   sigma>0 : sqrt(sigma_MC^2 + sigma_desordre^2 / N_d)
    if has_err:
        # Seule vraie erreur statistique : erreur MC de l'estimateur IS.
        # La dispersion entre réalisations de désordre est un observable physique,
        # pas une erreur — elle ne doit pas apparaître dans les barres d'erreur.
        # Tous les k partagent les mêmes échantillons MCMC, donc mc_err est identique
        # pour tous k : on prend k=0.
        total_err = mz2_err_raw[idx_s, :, 0]
    else:
        total_err = np.zeros_like(mz2_mean)

    data_by_L[L] = {
        "h0": h0_grid,
        "mz2": mz2_mean,
        "err": total_err,
        "sigma_used": actual_sigma,
        "N_d": N_d,
    }
    print(f"L={L}: sigma={actual_sigma}, {np.sum(~np.isnan(mz2_mean))}/{len(mz2_mean)} pts, N_d={N_d}")

if not data_by_L:
    print("Aucune donnée trouvée. Lance d'abord merge_chunks.py pour chaque L.")
    sys.exit(1)

# ==========================================
# CALCULS DÉRIVÉES ET PICS
# ==========================================
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(L_LIST)))
color_map = {L: colors[i] for i, L in enumerate(L_LIST)}

peak_L = []
peak_mean = []

for L, d in data_by_L.items():
    h0, mz2 = d["h0"], d["mz2"]
    valid = ~np.isnan(mz2)
    if np.sum(valid) < 2 * DERIVATIVE_STEP + 1:
        continue
    h0_v, mz2_v = h0[valid], mz2[valid]
    deriv = robust_derivative(mz2_v, h0_v, step=DERIVATIVE_STEP)
    mask = (h0_v >= H0_MIN) & (h0_v <= H0_MAX)
    if np.any(mask):
        peak_L.append(L)
        peak_mean.append(np.max(deriv[mask]))
    d["h0_v"] = h0_v
    d["deriv"] = deriv

# ==========================================
# PLOT
# ==========================================
sigma_label = rf"$\sigma={TARGET_SIGMA}$"
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

# --- Panneau 1 : Mz² vs h0 avec barres d'erreur ---
ax = axs[0]
for L, d in data_by_L.items():
    c = color_map[L]
    h0, mz2, err = d["h0"], d["mz2"], d["err"]
    ax.plot(h0, mz2, marker='.', markersize=3, linewidth=1.0, color=c, label=f"L={L}")
    ax.fill_between(h0, mz2 - err, mz2 + err, color=c, alpha=0.2, edgecolor='none')
ax.axvline(x=HC_INF, color='gray', linestyle=':', alpha=0.6, label=rf"$h_c^{{\infty}}={HC_INF}$")
ax.set_xlabel(r"Champ transverse $h_0$")
ax.set_ylabel(r"$\langle M_z^2 \rangle$")
ax.set_title(rf"$M_z^2$ — {sigma_label}, toutes tailles")
ax.legend(fontsize=9)
ax.grid(True, ls="--", alpha=0.3)

# --- Panneau 2 : |dMz²/dh0| vs h0 ---
ax = axs[1]
for L, d in data_by_L.items():
    if "deriv" not in d:
        continue
    c = color_map[L]
    ax.plot(d["h0_v"], d["deriv"], marker='.', markersize=3, linewidth=1.0, color=c, label=f"L={L}")
ax.axvline(x=HC_INF, color='gray', linestyle=':', alpha=0.6, label=rf"$h_c^{{\infty}}={HC_INF}$")
ax.set_xlabel(r"Champ transverse $h_0$")
ax.set_ylabel(r"$\left|\partial M_z^2 / \partial h_0\right|$")
ax.set_title(rf"Dérivée de $M_z^2$ (step={DERIVATIVE_STEP}) — {sigma_label}")
ax.legend(fontsize=9)
ax.grid(True, ls="--", alpha=0.3)

# --- Panneau 3 : max(|dMz²/dh0|) vs L en log-log avec fit affine (loi de puissance) ---
ax = axs[2]
if peak_L:
    L_arr = np.array(peak_L, dtype=float)
    p_arr = np.array(peak_mean, dtype=float)
    ax.plot(L_arr, p_arr, 'o', color='crimson', markersize=6, zorder=3)
    # Fit affine en log-log : log(pic) = alpha*log(L) + log(A)  =>  pic ~ A * L^alpha
    coeffs = np.polyfit(np.log(L_arr), np.log(p_arr), 1)
    alpha, logA = coeffs[0], coeffs[1]
    A = np.exp(logA)
    L_fit = np.linspace(L_arr[0] * 0.9, L_arr[-1] * 1.1, 300)
    ax.plot(L_fit, A * L_fit**alpha, '--', color='crimson', linewidth=1.2,
            label=rf"$\propto L^{{{alpha:.3f}}}$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(L_arr.astype(int))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=9)
    ax.set_xlabel(r"Taille $L$")
    ax.set_ylabel(r"$\max\left|\partial M_z^2 / \partial h_0\right|$")
    ax.set_title(rf"Pic de susceptibilité vs $L$ — {sigma_label} (log-log)")
    ax.grid(True, which='both', ls="--", alpha=0.3)

plt.suptitle(
    rf"Importance Sampling 1D — Trains_autour_transi_1D, {sigma_label}",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()

sigma_str = str(TARGET_SIGMA).replace(".", "p")
out_pdf = os.path.join(output_dir, f"mz2_multiL_transi_sigma{sigma_str}.pdf")
out_png = os.path.join(output_dir, f"mz2_multiL_transi_sigma{sigma_str}.png")
plt.savefig(out_pdf)
plt.savefig(out_png, dpi=150)
print(f"Sauvegardé: {out_pdf}")
