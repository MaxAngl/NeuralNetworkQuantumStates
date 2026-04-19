"""
FSS (Finite-Size Scaling) du cumulant de Binder — L=8 et L=10, 2D TFIM.

Pour chaque sigma commun aux deux tailles :
  1. Calcule B̄(h, L) et δB(h, L) sur les 100 réalisations de désordre
  2. Optimise (h_c, ν) par minimisation du χ² du collapse FSS :
       B(h, L) = Φ((h - h_c) · L^(1/ν)),   Φ = polynôme degré 3
     Le polynôme est ajusté analytiquement (moindres carrés pondérés)
     pour chaque (h_c, ν) testé, seul χ² résiduel est minimisé.
  3. Bootstrap (N=200) sur les réalisations de désordre pour barres d'erreur

Sorties :
  fss_results_2D.npz           : h_c, nu, leurs erreurs bootstrap, par sigma
  fss_collapse_sigma{s}.pdf/png : plot du collapse pour chaque sigma
  fss_summary_2D.pdf/png        : h_c(σ) et ν(σ) avec barres d'erreur

Référence méthode : O. Melchert (2009), arXiv:0910.5403

Usage:
    python fss_binder_2D.py [--n-bootstrap 200] [--poly-degree 3] [--output-dir DIR]
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import lstsq

# ==========================================
# ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--n-bootstrap", type=int, default=200,
                    help="Nombre de rééchantillonnages bootstrap (défaut: 200)")
parser.add_argument("--poly-degree", type=int, default=3,
                    help="Degré du polynôme pour la fonction d'échelle Φ (défaut: 3)")
parser.add_argument("--output-dir",  type=str, default=None,
                    help="Dossier de sortie (défaut: BASE_DIR)")
args = parser.parse_args()

N_BOOTSTRAP = args.n_bootstrap
POLY_DEG    = args.poly_degree
L_LIST      = [8, 10]

BASE_DIR   = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_autour_transi_2D"
OUTPUT_DIR = args.output_dir or os.path.join(BASE_DIR, "fss_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# CHARGEMENT DES DONNÉES
# ==========================================
print("=== FSS Binder 2D — L=8 et L=10 ===\n")

data = {}
for L in L_LIST:
    path = os.path.join(BASE_DIR, f"L={L}", f"binder_data_2D_L{L}.npz")
    if not os.path.exists(path):
        print(f"ERREUR : fichier introuvable — {path}")
        sys.exit(1)
    d = np.load(path)
    data[L] = {
        "h0_grid":    d["h0_grid"],       # (n_h0,)
        "sigma_grid": d["sigma_grid"],    # (n_sigma,)
        "binder_raw": d["binder_raw"],    # (n_sigma, n_h0, N_DISORDER)
    }
    print(f"L={L}: {len(d['sigma_grid'])} sigmas, binder shape = {d['binder_raw'].shape}")

# Sigmas communs aux deux tailles (en préservant l'ordre croissant)
sigma_L8  = data[8]["sigma_grid"]
sigma_L10 = data[10]["sigma_grid"]
sigma_common = np.array([s for s in sigma_L8
                         if any(abs(s - s2) < 1e-9 for s2 in sigma_L10)])
print(f"\nSigmas communs : {sigma_common}")

h0_grid = data[8]["h0_grid"]   # identique pour L=8 et L=10
n_h0    = len(h0_grid)

# ==========================================
# FONCTIONS FSS
# ==========================================

def poly_fit_weighted(x, y, w, deg):
    """
    Ajustement polynomial pondéré : min Σ w_i (y_i - p(x_i))².
    Les poids sont normalisés (somme = n) pour éviter l'instabilité numérique
    quand tous les B_err sont très petits (cas σ=0).
    Retourne coefficients numpy (degré décroissant, comme np.polyval).
    """
    w_norm = w / w.mean()                   # normalisation → moyenne = 1
    A  = np.vander(x, deg + 1)
    Aw = np.diag(np.sqrt(w_norm)) @ A
    yw = np.sqrt(w_norm) * y
    coeffs, _, _, _ = lstsq(Aw, yw)
    return coeffs


def chi2_collapse(params, B_mean, B_err, h0_grid, L_list, deg):
    """
    χ² résiduel du collapse FSS (normalisé par nombre de points).

    Paramètres optimisés :
      params[0] = h_c
      params[1] = log(ν)   — on optimise le log pour forcer ν > 0

    Pour un (h_c, ν) fixé :
      - calcule x_Li = (h_i - h_c) · L^(1/ν)
      - ajuste Φ̂(x) = polynôme deg par moindres carrés pondérés (analytique)
      - retourne χ²/n = moyenne pondérée des résidus au carré
    """
    h_c, log_nu = params
    nu = np.exp(log_nu)

    x_all, y_all, w_all = [], [], []
    for L in L_list:
        x = (h0_grid - h_c) * L ** (1.0 / nu)
        x_all.append(x)
        y_all.append(B_mean[L])
        w_all.append(1.0 / B_err[L] ** 2)

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    w_all = np.concatenate(w_all)

    if not np.all(np.isfinite(x_all)):
        return 1e10

    try:
        coeffs  = poly_fit_weighted(x_all, y_all, w_all, deg)
    except Exception:
        return 1e10

    phi_hat = np.polyval(coeffs, x_all)
    # Normaliser w pour que le χ² soit comparable entre σ
    w_norm  = w_all / w_all.mean()
    chi2    = float(np.mean((y_all - phi_hat) ** 2 * w_norm))
    return chi2


def fit_fss(B_mean, B_err, h0_grid, L_list, deg,
            hc_init=None, nu_init=0.63):
    """
    Optimise (h_c, ν) par minimisation de chi2_collapse.
    Effectue d'abord une recherche sur grille grossière pour l'initialisation,
    puis affine avec L-BFGS-B.
    Retourne (h_c, ν, chi2_min, success).
    """
    bounds = [(h0_grid.min(), h0_grid.max()), (np.log(0.05), np.log(10.0))]

    # Recherche grossière sur grille pour éviter les minima locaux
    hc_grid  = np.linspace(h0_grid.min(), h0_grid.max(), 30)
    nu_grid  = np.array([0.3, 0.5, 0.63, 0.8, 1.0, 1.5, 2.0])
    best_chi2 = np.inf
    best_x0   = [hc_init or 3.044, np.log(nu_init)]

    for hc_try in hc_grid:
        for nu_try in nu_grid:
            c = chi2_collapse([hc_try, np.log(nu_try)],
                              B_mean, B_err, h0_grid, L_list, deg)
            if c < best_chi2:
                best_chi2 = c
                best_x0   = [hc_try, np.log(nu_try)]

    res = minimize(
        chi2_collapse,
        best_x0,
        args=(B_mean, B_err, h0_grid, L_list, deg),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-9},
    )

    if np.isfinite(res.fun):
        return res.x[0], np.exp(res.x[1]), res.fun, True
    return np.nan, np.nan, np.nan, False


def get_B_stats(binder_raw_2d, disorder_indices=None):
    """
    Calcule B̄ et δB à partir de binder_raw_2d (n_h0, N_DISORDER).
    Si disorder_indices fourni, rééchantillonne (bootstrap).
    """
    if disorder_indices is None:
        B = binder_raw_2d                     # (n_h0, 100)
    else:
        B = binder_raw_2d[:, disorder_indices]  # (n_h0, 100)

    B_mean = np.mean(B, axis=1)                          # (n_h0,)
    B_std  = np.std(B, axis=1, ddof=1) / np.sqrt(B.shape[1])
    B_std  = np.where(B_std < 1e-8, 1e-8, B_std)        # évite division par 0
    return B_mean, B_std


# ==========================================
# BOUCLE PRINCIPALE PAR SIGMA
# ==========================================
rng = np.random.default_rng(42)

out_sigma   = []
out_hc      = []
out_hc_err  = []
out_nu      = []
out_nu_err  = []
out_chi2    = []

colors_L = {8: "#e6553a", 10: "#3a7ee6"}

for sigma in sigma_common:
    print(f"\n--- σ = {sigma:.3f} ---")

    idx_s8  = np.argmin(np.abs(data[8]["sigma_grid"]  - sigma))
    idx_s10 = np.argmin(np.abs(data[10]["sigma_grid"] - sigma))

    raw8  = data[8]["binder_raw"][idx_s8]    # (n_h0, 100)
    raw10 = data[10]["binder_raw"][idx_s10]  # (n_h0, 100)

    # --- Fit central ---
    B_mean_full = {}
    B_err_full  = {}
    B_mean_full[8],  B_err_full[8]  = get_B_stats(raw8)
    B_mean_full[10], B_err_full[10] = get_B_stats(raw10)

    hc_hat, nu_hat, chi2_hat, ok = fit_fss(
        B_mean_full, B_err_full, h0_grid, L_LIST, POLY_DEG
    )
    if not ok:
        print("  ATTENTION : fit central non convergé")

    dof = 2 * n_h0 - (POLY_DEG + 1) - 2   # 2 tailles × n_h0 pts - coefs poly - (hc,nu)
    print(f"  h_c = {hc_hat:.4f},  ν = {nu_hat:.4f},  χ²/dof = {chi2_hat/dof:.2f}")

    # --- Bootstrap ---
    hc_boot = np.full(N_BOOTSTRAP, np.nan)
    nu_boot = np.full(N_BOOTSTRAP, np.nan)

    for b in range(N_BOOTSTRAP):
        idx_b = rng.integers(0, 100, size=100)

        Bm, Be = {}, {}
        Bm[8],  Be[8]  = get_B_stats(raw8,  idx_b)
        Bm[10], Be[10] = get_B_stats(raw10, idx_b)

        hc_b, nu_b, _, ok_b = fit_fss(
            Bm, Be, h0_grid, L_LIST, POLY_DEG,
            hc_init=hc_hat, nu_init=nu_hat   # initialisation au fit central
        )
        if ok_b:
            hc_boot[b] = hc_b
            nu_boot[b] = nu_b

    valid    = ~np.isnan(hc_boot)
    n_valid  = int(np.sum(valid))
    hc_err   = float(np.std(hc_boot[valid], ddof=1)) if n_valid > 1 else np.nan
    nu_err   = float(np.std(nu_boot[valid], ddof=1)) if n_valid > 1 else np.nan

    print(f"  Bootstrap ({n_valid}/{N_BOOTSTRAP} valides) :")
    print(f"    h_c = {hc_hat:.4f} ± {hc_err:.4f}")
    print(f"    ν   = {nu_hat:.4f} ± {nu_err:.4f}")

    out_sigma.append(sigma)
    out_hc.append(hc_hat)
    out_hc_err.append(hc_err)
    out_nu.append(nu_hat)
    out_nu_err.append(nu_err)
    out_chi2.append(chi2_hat)

    # ==========================================
    # PLOT DU COLLAPSE POUR CE SIGMA
    # ==========================================
    fit_ok = np.isfinite(hc_hat) and np.isfinite(nu_hat)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # — Panneau gauche : courbes brutes B(h) —
    ax = axes[0]
    for L in L_LIST:
        Bm = B_mean_full[L]
        Be = B_err_full[L]
        ax.errorbar(h0_grid, Bm, yerr=Be, fmt='o-', markersize=4,
                    linewidth=1.2, color=colors_L[L], label=f"$L={L}$",
                    elinewidth=0.8, capsize=2)
    if fit_ok:
        ax.axvline(hc_hat, color="gray", ls=":", lw=1.2,
                   label=rf"$h_c={hc_hat:.4f}$")
    ax.set_xlabel(r"$h_0$", fontsize=12)
    ax.set_ylabel(r"$\bar{B}(h, L)$", fontsize=12)
    ax.set_title(rf"$\sigma={sigma:.3f}$ — courbes brutes", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)

    # — Panneau droit : collapse (h - hc)·L^(1/ν) vs B —
    ax = axes[1]
    if fit_ok:
        x_min_plot, x_max_plot = np.inf, -np.inf
        for L in L_LIST:
            Bm = B_mean_full[L]
            Be = B_err_full[L]
            x_red = (h0_grid - hc_hat) * L ** (1.0 / nu_hat)
            ax.errorbar(x_red, Bm, yerr=Be, fmt='o', markersize=5,
                        color=colors_L[L], label=f"$L={L}$",
                        elinewidth=0.8, capsize=2)
            x_min_plot = min(x_min_plot, x_red.min())
            x_max_plot = max(x_max_plot, x_red.max())

        # Courbe polynomiale Φ̂
        x_all_collapse = np.concatenate([
            (h0_grid - hc_hat) * L ** (1.0 / nu_hat) for L in L_LIST
        ])
        y_all_collapse = np.concatenate([B_mean_full[L] for L in L_LIST])
        w_all_collapse = np.concatenate([1.0 / B_err_full[L]**2 for L in L_LIST])
        coeffs_plot = poly_fit_weighted(x_all_collapse, y_all_collapse,
                                        w_all_collapse, POLY_DEG)
        x_fine   = np.linspace(x_min_plot, x_max_plot, 300)
        phi_fine = np.polyval(coeffs_plot, x_fine)
        ax.plot(x_fine, phi_fine, "k--", lw=1.2, label=rf"$\Phi$ (deg {POLY_DEG})")
    else:
        ax.text(0.5, 0.5, "Fit non convergé", transform=ax.transAxes,
                ha="center", va="center", fontsize=13, color="red")

    ax.axvline(0, color="gray", ls=":", lw=1.0)
    ax.set_xlabel(r"$(h_0 - h_c)\, L^{1/\nu}$", fontsize=12)
    ax.set_ylabel(r"$\bar{B}$", fontsize=12)
    hc_str = f"{hc_hat:.4f}±{hc_err:.4f}" if fit_ok else "N/A"
    nu_str = f"{nu_hat:.4f}±{nu_err:.4f}" if fit_ok else "N/A"
    ax.set_title(rf"Collapse — $h_c=${hc_str}, $\nu=${nu_str}", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)

    chi2_str = f"{chi2_hat/dof:.2f}" if fit_ok else "N/A"
    fig.suptitle(
        rf"FSS Binder 2D TFIM — $\sigma={sigma:.3f}$, $\chi^2/\mathrm{{dof}}={chi2_str}$",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    sigma_str = f"{sigma:.3f}".replace(".", "p")
    for ext in ("pdf", "png"):
        out = os.path.join(OUTPUT_DIR, f"fss_collapse_sigma{sigma_str}.{ext}")
        plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Collapse sauvegardé : fss_collapse_sigma{sigma_str}.pdf/.png")

# ==========================================
# TABLEAU RÉCAPITULATIF
# ==========================================
print("\n" + "=" * 55)
print(f"{'σ':>6}  {'h_c':>8}  {'±':>6}  {'ν':>7}  {'±':>6}  χ²/dof")
print("-" * 55)
dof = 2 * n_h0 - (POLY_DEG + 1) - 2
for i, sigma in enumerate(out_sigma):
    print(f"{sigma:6.3f}  {out_hc[i]:8.4f}  {out_hc_err[i]:6.4f}  "
          f"{out_nu[i]:7.4f}  {out_nu_err[i]:6.4f}  {out_chi2[i]/dof:6.2f}")
print("=" * 55)

# ==========================================
# SAUVEGARDE NUMÉRIQUE
# ==========================================
out_npz = os.path.join(OUTPUT_DIR, "fss_results_2D.npz")
np.savez(
    out_npz,
    sigma_grid = np.array(out_sigma),
    hc         = np.array(out_hc),
    hc_err     = np.array(out_hc_err),
    nu         = np.array(out_nu),
    nu_err     = np.array(out_nu_err),
    chi2       = np.array(out_chi2),
    L_list     = np.array(L_LIST),
    poly_degree= np.array(POLY_DEG),
    n_bootstrap= np.array(N_BOOTSTRAP),
)
print(f"\nRésultats sauvegardés : {out_npz}")

# ==========================================
# PLOT RÉCAPITULATIF h_c(σ) et ν(σ)
# ==========================================
sigma_arr  = np.array(out_sigma)
hc_arr     = np.array(out_hc)
hc_err_arr = np.array(out_hc_err)
nu_arr     = np.array(out_nu)
nu_err_arr = np.array(out_nu_err)
valid      = ~np.isnan(hc_arr)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# h_c(σ)
ax1.errorbar(sigma_arr[valid], hc_arr[valid], yerr=hc_err_arr[valid],
             fmt='o-', color="#e6553a", markersize=6,
             elinewidth=1.2, capsize=4, linewidth=1.2)
ax1.axhline(3.044, color="gray", ls="--", lw=1.0,
            label=r"$h_c^\infty=3.044$ ($\sigma=0$)")
ax1.set_xlabel(r"Désordre $\sigma$", fontsize=12)
ax1.set_ylabel(r"$h_c(\sigma)$", fontsize=12)
ax1.set_title(r"Point critique $h_c$ vs désordre", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, ls="--", alpha=0.3)

# ν(σ)
ax2.errorbar(sigma_arr[valid], nu_arr[valid], yerr=nu_err_arr[valid],
             fmt='s-', color="#3a7ee6", markersize=6,
             elinewidth=1.2, capsize=4, linewidth=1.2)
ax2.axhline(0.63, color="gray", ls="--", lw=1.0,
            label=r"$\nu=0.63$ (3D Ising pur)")
ax2.set_xlabel(r"Désordre $\sigma$", fontsize=12)
ax2.set_ylabel(r"$\nu(\sigma)$", fontsize=12)
ax2.set_title(r"Exposant critique $\nu$ vs désordre", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, ls="--", alpha=0.3)

fig.suptitle(
    rf"FSS Binder 2D TFIM — L=8 et L=10, polynôme deg {POLY_DEG}, "
    rf"{N_BOOTSTRAP} bootstraps",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(OUTPUT_DIR, f"fss_summary_2D.{ext}")
    plt.savefig(out, dpi=150)
plt.close()
print(f"Résumé sauvegardé : fss_summary_2D.pdf/.png")
print("\nTerminé.")
