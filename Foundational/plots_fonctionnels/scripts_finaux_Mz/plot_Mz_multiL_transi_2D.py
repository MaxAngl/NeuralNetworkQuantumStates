"""
Plots 2D Trains_autour_transi_2D :
  - Pour chaque L : mz2_IS_analysis_2D_L=X.png/pdf (3 panneaux : Mz², dérivée, max dérivée vs sigma)
  - Récapitulatif multi-L : mz2_multiL_transi_2D_sigma0p0.pdf (Mz², dérivée, pic vs L)

Usage:
    python plot_Mz_multiL_transi_2D.py [--sigma 0.0] [--deriv-step 2]
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# CONFIG
# ==========================================
PROJECT = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
BASE_DIR = os.path.join(PROJECT, "Foundational/logs/Trains_autour_transi_2D")

L_LIST = [3, 4, 5, 6, 8, 10]
HC_INF = 3.044              # hc thermodynamique 2D Ising transverse
H0_MIN, H0_MAX = 2.8, 3.3  # plage pour chercher le pic de susceptibilité

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=0.0)
parser.add_argument("--deriv-step", type=int, default=2)
args = parser.parse_args()

TARGET_SIGMA   = args.sigma
DERIV_STEP     = args.deriv_step
LW, MS         = 1.0, 2.0
ALPHA_L, ALPHA_F = 0.85, 0.15


# ==========================================
# HELPERS
# ==========================================
def merge_chunks(L):
    """Fusionne les chunks en mémoire, renvoie un dict avec les données complètes."""
    files = sorted(glob.glob(os.path.join(BASE_DIR, f"L={L}", f"is_data_2D_L{L}_chunk*.npz")))
    if not files:
        return None
    first = np.load(files[0])
    mz2   = first["mz2_raw"].copy()
    ess   = first["ess_raw"].copy()
    has_err = "mz2_err_raw" in first
    err   = first["mz2_err_raw"].copy() if has_err else None

    for f in files[1:]:
        c = np.load(f)
        for s in range(mz2.shape[0]):
            for h in range(mz2.shape[1]):
                if not np.isnan(c["mz2_raw"][s, h, 0]):
                    mz2[s, h, :] = c["mz2_raw"][s, h, :]
                    ess[s, h, :] = c["ess_raw"][s, h, :]
                    if has_err:
                        err[s, h, :] = c["mz2_err_raw"][s, h, :]

    return dict(
        h0_grid=first["h0_grid"],
        sigma_grid=first["sigma_grid"],
        mz2_raw=mz2,
        mz2_err_raw=err,
        ess_raw=ess,
        N_SAMPLES_IS=int(first["N_SAMPLES_IS"]),
        L=int(first["L"]),
        nb_spins=int(first["nb_spins"]),
        dim=int(first["dim"]),
    )


def robust_derivative(y, x, step=1):
    dy = np.zeros_like(y)
    n = len(y)
    for i in range(n):
        left  = max(0, i - step)
        right = min(n - 1, i + step)
        dy[i] = 0.0 if right == left else (y[right] - y[left]) / (x[right] - x[left])
    return np.abs(dy)


# ==========================================
# 1. PLOTS INDIVIDUELS PAR L
# ==========================================
print("=== Plots individuels par L ===")

for L in L_LIST:
    d = merge_chunks(L)
    if d is None:
        print(f"  L={L}: pas de données, ignoré")
        continue

    sigma_grid  = d["sigma_grid"]
    h0_grid     = d["h0_grid"]
    mz2_raw     = d["mz2_raw"]
    mz2_err_raw = d["mz2_err_raw"]
    nb_spins    = d["nb_spins"]
    N_SAMPLES   = d["N_SAMPLES_IS"]

    # Calculs par sigma
    mean_of_mz2   = []
    deriv_of_mean = []
    micro_mean_max = []
    micro_err_max  = []

    for idx_s, sigma in enumerate(sigma_grid):
        mean_curve = np.nanmean(mz2_raw[idx_s], axis=1)
        mean_of_mz2.append(mean_curve)
        deriv_of_mean.append(robust_derivative(mean_curve, h0_grid, step=DERIV_STEP))

        max_d_list = []
        for k in range(mz2_raw.shape[2]):
            curve_k = mz2_raw[idx_s, :, k]
            valid   = ~np.isnan(curve_k)
            if np.sum(valid) > 2 * DERIV_STEP:
                h_v, m_v = h0_grid[valid], curve_k[valid]
                dk = robust_derivative(m_v, h_v, step=DERIV_STEP)
                mask = (h_v >= H0_MIN) & (h_v <= H0_MAX)
                if np.any(mask):
                    max_d_list.append(np.max(dk[mask]))

        micro_mean_max.append(np.mean(max_d_list) if max_d_list else np.nan)
        micro_err_max.append(
            np.std(max_d_list, ddof=1) / np.sqrt(len(max_d_list))
            if len(max_d_list) > 1 else 0.0
        )

    colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_grid)))
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panneau 1 : Mz² vs h0
    ax = axs[0]
    for idx, sigma in enumerate(sigma_grid):
        c = colors[idx]
        ax.plot(h0_grid, mean_of_mz2[idx], marker='.', markersize=MS,
                color=c, linewidth=LW, alpha=ALPHA_L, label=rf"$\sigma={sigma}$")
        if mz2_err_raw is not None:
            mc_err = mz2_err_raw[idx, :, 0]
            ax.fill_between(h0_grid,
                            mean_of_mz2[idx] - mc_err,
                            mean_of_mz2[idx] + mc_err,
                            color=c, alpha=0.5, edgecolor='none')
    ax.axvline(x=HC_INF, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r"Transverse Field $h_0$")
    ax.set_ylabel(r"$\langle M_z^2 \rangle$")
    ax.set_title(rf"Mean Squared Magnetization — 2D $L={L}$ ({nb_spins} spins)")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, ls="--", alpha=0.3)

    # Panneau 2 : |dMz²/dh0| vs h0
    ax = axs[1]
    for idx, sigma in enumerate(sigma_grid):
        c = colors[idx]
        ax.plot(h0_grid, deriv_of_mean[idx], marker='.', markersize=MS,
                color=c, linewidth=LW, alpha=ALPHA_L)
    ax.axvline(x=HC_INF, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r"Transverse Field $h_0$")
    ax.set_ylabel(r"Susceptibility (Robust Derivative)")
    ax.set_title(f"Derivative of Mean Mz² (step={DERIV_STEP})")
    ax.grid(True, ls="--", alpha=0.3)

    # Panneau 3 : max dérivée vs sigma
    ax = axs[2]
    ax.errorbar(sigma_grid, micro_mean_max, yerr=micro_err_max,
                fmt='-o', color='red', markerfacecolor='red',
                linewidth=1.0, markersize=3, elinewidth=1.0, capsize=3, alpha=0.9)
    ax.set_xlabel(r"Disorder strength $\sigma$")
    ax.set_ylabel(r"$\langle \max \left| \partial M_z^2 / \partial h_0 \right| \rangle$")
    ax.set_title(rf"Mean of Max Derivatives ($h_0 \in [{H0_MIN}, {H0_MAX}]$)")
    ax.grid(True, ls="--", alpha=0.3)

    plt.suptitle(
        f"Importance Sampling 2D — L={L} ({nb_spins} spins), {N_SAMPLES} samples",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    out_dir = os.path.join(BASE_DIR, f"L={L}")
    base    = f"mz2_IS_analysis_2D_L={L}"
    plt.savefig(os.path.join(out_dir, f"{base}.png"), dpi=150)
    plt.savefig(os.path.join(out_dir, f"{base}.pdf"))
    plt.close()
    print(f"  L={L}: sauvegardé dans {out_dir}/{base}.{{png,pdf}}")


# ==========================================
# 2. PLOT RÉCAPITULATIF MULTI-L (sigma=0)
# ==========================================
print(f"\n=== Plot récapitulatif multi-L (sigma={TARGET_SIGMA}) ===")

colors_L  = plt.cm.plasma(np.linspace(0.1, 0.9, len(L_LIST)))
color_map = {L: colors_L[i] for i, L in enumerate(L_LIST)}

data_by_L = {}
for L in L_LIST:
    d = merge_chunks(L)
    if d is None:
        continue

    sigma_grid  = d["sigma_grid"]
    idx_s       = int(np.argmin(np.abs(sigma_grid - TARGET_SIGMA)))
    actual_sigma = float(sigma_grid[idx_s])
    h0_grid     = d["h0_grid"]
    mz2_raw     = d["mz2_raw"]

    mz2_mean = np.nanmean(mz2_raw[idx_s], axis=1)

    if d["mz2_err_raw"] is not None:
        total_err = d["mz2_err_raw"][idx_s, :, 0]
    else:
        total_err = np.zeros_like(mz2_mean)

    data_by_L[L] = {
        "h0": h0_grid, "mz2": mz2_mean, "err": total_err,
        "sigma_used": actual_sigma, "nb_spins": d["nb_spins"],
    }
    print(f"  L={L}: sigma={actual_sigma}, {np.sum(~np.isnan(mz2_mean))}/{len(mz2_mean)} pts")

# Calcul dérivées et pics
peak_L    = []
peak_mean = []

for L, d in data_by_L.items():
    h0, mz2 = d["h0"], d["mz2"]
    valid   = ~np.isnan(mz2)
    if np.sum(valid) < 2 * DERIV_STEP + 1:
        continue
    h0_v, mz2_v = h0[valid], mz2[valid]
    deriv = robust_derivative(mz2_v, h0_v, step=DERIV_STEP)
    mask  = (h0_v >= H0_MIN) & (h0_v <= H0_MAX)
    if np.any(mask):
        peak_L.append(L)
        peak_mean.append(np.max(deriv[mask]))
    d["h0_v"]  = h0_v
    d["deriv"] = deriv

sigma_label = rf"$\sigma={TARGET_SIGMA}$"
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

# Panneau 1 : Mz² vs h0
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

# Panneau 2 : |dMz²/dh0| vs h0
ax = axs[1]
for L, d in data_by_L.items():
    if "deriv" not in d:
        continue
    c = color_map[L]
    ax.plot(d["h0_v"], d["deriv"], marker='.', markersize=3, linewidth=1.0, color=c, label=f"L={L}")
ax.axvline(x=HC_INF, color='gray', linestyle=':', alpha=0.6, label=rf"$h_c^{{\infty}}={HC_INF}$")
ax.set_xlabel(r"Champ transverse $h_0$")
ax.set_ylabel(r"$\left|\partial M_z^2 / \partial h_0\right|$")
ax.set_title(rf"Dérivée de $M_z^2$ (step={DERIV_STEP}) — {sigma_label}")
ax.legend(fontsize=9)
ax.grid(True, ls="--", alpha=0.3)

# Panneau 3 : max(|dMz²/dh0|) vs L (log-log + fit puissance)
ax = axs[2]
if peak_L:
    L_arr = np.array(peak_L, dtype=float)
    p_arr = np.array(peak_mean, dtype=float)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(L_arr, p_arr, 'o', color='crimson', markersize=6, zorder=3)
    if len(L_arr) >= 2:
        coeffs = np.polyfit(np.log(L_arr), np.log(p_arr), 1)
        alpha, logA = coeffs[0], coeffs[1]
        A = np.exp(logA)
        L_fit = np.linspace(L_arr[0] * 0.9, L_arr[-1] * 1.1, 300)
        ax.plot(L_fit, A * L_fit**alpha, '--', color='crimson', linewidth=1.2,
                label=rf"$\propto L^{{{alpha:.3f}}}$")
        ax.legend(fontsize=9)
    ax.set_xticks(L_arr.astype(int))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"Taille $L$")
    ax.set_ylabel(r"$\max\left|\partial M_z^2 / \partial h_0\right|$")
    ax.set_title(rf"Pic de susceptibilité sans désordre vs $L$ — {sigma_label} (log-log)")
    ax.grid(True, which='both', ls="--", alpha=0.3)

plt.suptitle(
    rf"Importance Sampling 2D — Trains_autour_transi_2D, {sigma_label}",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()

sigma_str = str(TARGET_SIGMA).replace(".", "p")
out_pdf = os.path.join(BASE_DIR, f"mz2_multiL_2D_transi_sigma{sigma_str}.pdf")
out_png = os.path.join(BASE_DIR, f"mz2_multiL_2D_transi_sigma{sigma_str}.png")
plt.savefig(out_pdf)
plt.savefig(out_png, dpi=150)
plt.close()
print(f"\nRécapitulatif sauvegardé: {out_pdf}")
