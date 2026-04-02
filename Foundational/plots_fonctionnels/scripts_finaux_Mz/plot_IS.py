"""
Script unifie de visualisation des resultats d'importance sampling.

Usage:
    python plot_IS.py <fichier.npz> [--dim 1|2] [--deriv-step 3] [--h0-min 0.5] [--h0-max 1.5]
                                     [--output-dir DIR] [--sigma-max 0.3]

Exemples:
    python plot_IS.py is_data_1D_L49_full.npz --dim 1
    python plot_IS.py is_data_2D_L8_full.npz --dim 2
    python plot_IS.py is_data_2D_L8_full.npz --h0-min 2.0 --h0-max 3.5
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot IS results")
parser.add_argument("data_path", help="Chemin vers le fichier .npz")
parser.add_argument("--dim", type=int, default=None, help="Dimension (auto-detecte si possible)")
parser.add_argument("--deriv-step", type=int, default=1, help="Pas pour derivee robuste (defaut: 3)")
parser.add_argument("--h0-min", type=float, default=None, help="h0 min pour max derivee (defaut: auto)")
parser.add_argument("--h0-max", type=float, default=None, help="h0 max pour max derivee (defaut: auto)")
parser.add_argument("--output-dir", type=str, default=None, help="Dossier de sortie (defaut: meme que data)")
parser.add_argument("--sigma-max", type=float, default=0.3, help="Sigma max pour panneau 3 (defaut: 0.3)")
args = parser.parse_args()

# ==========================================
# CHARGEMENT
# ==========================================
print(f"Chargement: {args.data_path}")
data = np.load(args.data_path)

sigma_grid = data["sigma_grid"]
h0_grid = data["h0_grid"]
mz2_raw = data["mz2_raw"]
N_SAMPLES = int(data["N_SAMPLES_IS"])
L = int(data["L"])
nb_spins = int(data["nb_spins"]) if "nb_spins" in data else L

# Auto-detect dimension
if args.dim:
    DIM = args.dim
elif "dim" in data:
    DIM = int(data["dim"])
elif nb_spins != L:
    DIM = 2
else:
    DIM = 1

# Plage h0 pour max derivee
if args.h0_min is not None:
    h0_min = args.h0_min
else:
    h0_min = 2.0 if DIM == 2 else 0.5

if args.h0_max is not None:
    h0_max = args.h0_max
else:
    h0_max = 3.5 if DIM == 2 else 1.5

output_dir = args.output_dir or os.path.dirname(args.data_path) or "."
DERIVATIVE_STEP = args.deriv_step

dim_label = f"{DIM}D"
hc_inf = 3.04 if DIM == 2 else 1.0

# ==========================================
# DERIVEE ROBUSTE
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

# ==========================================
# CALCULS
# ==========================================
print(f"Traitement {dim_label} L={L} ({nb_spins} spins), derivee step={DERIVATIVE_STEP}, h0 range [{h0_min}, {h0_max}]")

mean_of_mz2 = []
std_of_mz2 = []
deriv_of_mean = []
micro_mean_max = []
micro_err_max = []

for idx_s, sigma in enumerate(sigma_grid):
    mean_curve = np.nanmean(mz2_raw[idx_s], axis=1)
    std_curve = np.nanstd(mz2_raw[idx_s], axis=1, ddof=1)
    mean_of_mz2.append(mean_curve)
    std_of_mz2.append(std_curve)
    deriv_of_mean.append(robust_derivative(mean_curve, h0_grid, step=DERIVATIVE_STEP))

    max_d_micro_list = []
    for k in range(mz2_raw.shape[2]):
        curve_k = mz2_raw[idx_s, :, k]
        valid_mask = ~np.isnan(curve_k)
        if np.sum(valid_mask) > 2 * DERIVATIVE_STEP:
            h0_valid = h0_grid[valid_mask]
            mz2_valid = curve_k[valid_mask]
            dk_micro = robust_derivative(mz2_valid, h0_valid, step=DERIVATIVE_STEP)
            transition_mask = (h0_valid >= h0_min) & (h0_valid <= h0_max)
            if np.any(transition_mask):
                max_d_micro_list.append(np.max(dk_micro[transition_mask]))

    micro_mean_max.append(np.mean(max_d_micro_list) if max_d_micro_list else np.nan)
    micro_err_max.append(np.std(max_d_micro_list, ddof=1) / np.sqrt(len(max_d_micro_list)) if len(max_d_micro_list) > 1 else 0.0)

# ==========================================
# PLOTTING
# ==========================================
print("Generation de la figure...")
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

colors_mz2 = plt.cm.viridis(np.linspace(0, 1, len(sigma_grid)))
LW = 1.0
MS = 2.0
ALPHA_LINE = 0.85
ALPHA_FILL = 0.15

# Panneau 1 : Mz^2 vs h0
ax = axs[0]
for idx, sigma in enumerate(sigma_grid):
    c = colors_mz2[idx]
    ax.plot(h0_grid, mean_of_mz2[idx], marker='.', markersize=MS, color=c, linewidth=LW, alpha=ALPHA_LINE, label=rf"$\sigma = {sigma}$")
    ax.fill_between(h0_grid, mean_of_mz2[idx] - std_of_mz2[idx], mean_of_mz2[idx] + std_of_mz2[idx], color=c, alpha=ALPHA_FILL, edgecolor='none')
ax.axvline(x=hc_inf, color='gray', linestyle=':', alpha=0.5, label=rf"$h_c^{{\infty}} \approx {hc_inf}$")
ax.set_xlabel(r"Transverse Field $h_0$")
ax.set_ylabel(r"$\langle M_z^2 \rangle$")
ax.set_title(rf"Mean Squared Magnetization — {dim_label} $L={L}$ ({nb_spins} spins)")
ax.grid(True, ls="--", alpha=0.3)
ax.legend(fontsize=9, loc='upper right')

# Panneau 2 : Derivees vs h0
ax = axs[1]
for idx, sigma in enumerate(sigma_grid):
    c = colors_mz2[idx]
    ax.plot(h0_grid, deriv_of_mean[idx], marker='.', markersize=MS, color=c, linewidth=LW, alpha=ALPHA_LINE)
ax.axvline(x=hc_inf, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel(r"Transverse Field $h_0$")
ax.set_ylabel(r"Susceptibility (Robust Derivative)")
ax.set_title(f"Derivative of Mean Mz² (span={DERIVATIVE_STEP})")
ax.grid(True, ls="--", alpha=0.3)

# Panneau 3 : Max derivee vs Sigma
ax = axs[2]
mask = sigma_grid <= args.sigma_max
ax.errorbar(sigma_grid[mask], np.array(micro_mean_max)[mask], yerr=np.array(micro_err_max)[mask],
            fmt='-o', color='red', markerfacecolor='red', markeredgecolor='red',
            linewidth=1.0, markersize=3, elinewidth=1.0, capsize=3, alpha=0.9)
ax.set_xlabel(r"Disorder strength $\sigma$")
ax.set_ylabel(r"$\langle \max \left| \frac{\partial M_z^2}{\partial h_0} \right| \rangle$")
ax.set_title(rf"Mean of Max Derivatives ($h_0 \in [{h0_min}, {h0_max}]$)")
ax.grid(True, ls="--", alpha=0.3)

plt.suptitle(f"Importance Sampling {dim_label} — L={L} ({nb_spins} spins), {N_SAMPLES} samples", fontsize=13, fontweight='bold')
plt.tight_layout()

basename = f"mz2_IS_analysis_{dim_label}_L={L}"
output_png = os.path.join(output_dir, f"{basename}.png")
output_pdf = os.path.join(output_dir, f"{basename}.pdf")
plt.savefig(output_png, dpi=150)
plt.savefig(output_pdf)
print(f"Sauvegarde: {output_png}")
