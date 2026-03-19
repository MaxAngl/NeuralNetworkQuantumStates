import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_finaux_disordered_1D/run_L=25/mz2_full_data_L=25.npz"

# --- Extraction automatique de L depuis le path ---
match = re.search(r'L=(\d+)', DATA_PATH)
if match:
    L = int(match.group(1))
else:
    raise ValueError("Impossible de trouver L dans le DATA_PATH.")

# --- Choix de la méthode de lissage ---
SMOOTHING_METHOD = "gaussian"  # Choisir "savgol" ou "gaussian"

# Paramètres Savitzky-Golay
SG_WINDOW = 5  # Taille de la fenêtre (doit être impair)
SG_ORDER = 2   # Degré du polynôme

# Paramètres Gaussien
GAUSS_SIGMA = 1.5  # Largeur de la fenêtre de lissage

# ==========================================
# 2. CHARGEMENT DES DONNÉES
# ==========================================
print(f"📂 Chargement des données pour L={L}")
print(f"🛠️  Méthode de lissage sélectionnée : {SMOOTHING_METHOD.upper()}")
data = np.load(DATA_PATH)

sigma_grid = data["sigma_grid"]
h0_grid = data["h0_grid"]
mz2_raw = data["mz2_raw"] 

# ==========================================
# 3. CALCULS STATISTIQUES
# ==========================================
print("⚙️ Calcul des grandeurs (Brutes et Lissées)...")

mean_of_mz2, std_of_mz2 = [], []
deriv_mean_raw, deriv_mean_smooth = [], []
max_deriv_mean_raw, max_deriv_mean_smooth = [], []
mean_max_deriv_raw, err_max_deriv_raw = [], []
mean_max_deriv_smooth, err_max_deriv_smooth = [], []

# Détermination du nombre minimal de points requis pour le lissage choisi
min_points_required = SG_WINDOW if SMOOTHING_METHOD == "savgol" else 2

for idx_s, sigma in enumerate(sigma_grid):
    # --------------------------------------------------
    # A. MACROSCOPIQUE
    # --------------------------------------------------
    mean_curve = np.nanmean(mz2_raw[idx_s], axis=1)
    std_curve = np.nanstd(mz2_raw[idx_s], axis=1, ddof=1) 
    
    mean_of_mz2.append(mean_curve)
    std_of_mz2.append(std_curve)
    
    # 1. Sans lissage
    d_raw = np.abs(np.gradient(mean_curve, h0_grid))
    deriv_mean_raw.append(d_raw)
    max_deriv_mean_raw.append(np.max(d_raw))
    
    # 2. Avec lissage (dynamique)
    if SMOOTHING_METHOD == "savitzky":
        mean_curve_smooth = savgol_filter(mean_curve, window_length=SG_WINDOW, polyorder=SG_ORDER)
    elif SMOOTHING_METHOD == "gaussian":
        mean_curve_smooth = gaussian_filter1d(mean_curve, sigma=GAUSS_SIGMA)
    else:
        raise ValueError("SMOOTHING_METHOD doit être 'savitzky' ou 'gaussian'")
        
    d_smooth = np.abs(np.gradient(mean_curve_smooth, h0_grid))
    deriv_mean_smooth.append(d_smooth)
    max_deriv_mean_smooth.append(np.max(d_smooth))
    
    # --------------------------------------------------
    # B. MICROSCOPIQUE
    # --------------------------------------------------
    max_d_raw_list = []
    max_d_smooth_list = []
    MAX_KEEP = mz2_raw.shape[2]
    
    for k in range(MAX_KEEP):
        curve_k = mz2_raw[idx_s, :, k]
        valid_mask = ~np.isnan(curve_k)
        
        # Sécurité : vérifier qu'on a assez de points pour la méthode choisie
        if np.sum(valid_mask) >= min_points_required:
            h0_valid = h0_grid[valid_mask]
            mz2_valid = curve_k[valid_mask]
            
            # 1. Sans lissage
            dk_raw = np.abs(np.gradient(mz2_valid, h0_valid))
            max_d_raw_list.append(np.max(dk_raw))
            
            # 2. Avec lissage (dynamique)
            if SMOOTHING_METHOD == "savitzky":
                mz2_smooth = savgol_filter(mz2_valid, window_length=SG_WINDOW, polyorder=SG_ORDER)
            elif SMOOTHING_METHOD == "gaussian":
                mz2_smooth = gaussian_filter1d(mz2_valid, sigma=GAUSS_SIGMA)
                
            dk_smooth = np.abs(np.gradient(mz2_smooth, h0_valid))
            max_d_smooth_list.append(np.max(dk_smooth))
            
    # Bilan Brut (Raw)
    mean_max_deriv_raw.append(np.mean(max_d_raw_list))
    err_max_deriv_raw.append(np.std(max_d_raw_list, ddof=1) / np.sqrt(len(max_d_raw_list)))
    
    # Bilan Lissé (Smooth)
    mean_max_deriv_smooth.append(np.mean(max_d_smooth_list))
    err_max_deriv_smooth.append(np.std(max_d_smooth_list, ddof=1) / np.sqrt(len(max_d_smooth_list)))

# ==========================================
# 4. PLOTTING
# ==========================================
print("📈 Génération de la figure...")
fig = plt.figure(figsize=(14, 15))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])

colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_grid)))
LW = 0.8
MS = 2.0
ALPHA_LINE = 0.8
ALPHA_FILL = 0.15

# --- LIGNE 1 : Mz^2 ---
ax1 = fig.add_subplot(gs[0, :])
for idx, sigma in enumerate(sigma_grid):
    c = colors[idx]
    ax1.plot(h0_grid, mean_of_mz2[idx], marker='o', markersize=MS, color=c, linewidth=LW, alpha=ALPHA_LINE, label=rf"$\sigma = {sigma}$")
    ax1.fill_between(h0_grid, mean_of_mz2[idx] - std_of_mz2[idx], mean_of_mz2[idx] + std_of_mz2[idx], color=c, alpha=ALPHA_FILL, edgecolor='none')
ax1.set_xlabel(r"Transverse Field $h_0$")
ax1.set_ylabel(r"$\langle M_z^2 \rangle$")
ax1.set_title(r"Mean Squared Magnetization (Shaded = Sample Std Dev)")
ax1.grid(True, ls="--", alpha=0.3)
ax1.legend(fontsize=9, loc='upper right')

# --- LIGNE 2 : Comparaison des Dérivées ---
ax2 = fig.add_subplot(gs[1, 0])
for idx, sigma in enumerate(sigma_grid):
    ax2.plot(h0_grid, deriv_mean_raw[idx], marker='o', markersize=MS, color=colors[idx], linewidth=LW, alpha=ALPHA_LINE)
ax2.set_xlabel(r"Transverse Field $h_0$")
ax2.set_ylabel(r"Raw Derivative")
ax2.set_title(r"Derivative of Mean Mz² (NO SMOOTHING)")
ax2.grid(True, ls="--", alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
for idx, sigma in enumerate(sigma_grid):
    ax3.plot(h0_grid, deriv_mean_smooth[idx], marker='o', markersize=MS, color=colors[idx], linewidth=LW, alpha=ALPHA_LINE)
ax3.set_xlabel(r"Transverse Field $h_0$")
ax3.set_ylabel(r"Smoothed Derivative")
ax3.set_title(f"Derivative of Mean Mz² (WITH {SMOOTHING_METHOD.upper()} SMOOTHING)")
ax3.grid(True, ls="--", alpha=0.3)

# --- LIGNE 3 : Les bilans finaux ---
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(sigma_grid, max_deriv_mean_raw, marker='x', linestyle='--', color='crimson', label="Raw (Noisy)", linewidth=1.5)
ax4.plot(sigma_grid, max_deriv_mean_smooth, marker='s', linestyle='-', color='dodgerblue', label=f"Smoothed ({SMOOTHING_METHOD})", linewidth=1.5)
ax4.set_xlabel(r"Disorder strength $\sigma$")
ax4.set_ylabel(r"$\max \left| \frac{\partial \langle M_z^2 \rangle}{\partial h_0} \right|$")
ax4.set_title(r"Max of the Mean Derivative (Macroscopic)")
ax4.grid(True, ls="--", alpha=0.3)
ax4.legend()

ax5 = fig.add_subplot(gs[2, 1])
ax5.errorbar(sigma_grid, mean_max_deriv_raw, yerr=err_max_deriv_raw, fmt='--x', color='crimson', label="Raw (Noisy)", capsize=3, linewidth=1.5)
ax5.errorbar(sigma_grid, mean_max_deriv_smooth, yerr=err_max_deriv_smooth, fmt='-s', color='dodgerblue', label=f"Smoothed ({SMOOTHING_METHOD})", capsize=3, linewidth=1.5)
ax5.set_xlabel(r"Disorder strength $\sigma$")
ax5.set_ylabel(r"$\langle \max \left| \frac{\partial M_z^2}{\partial h_0} \right| \rangle$")
ax5.set_title(r"Mean of the Max Derivatives (Microscopic)")
ax5.grid(True, ls="--", alpha=0.3)
ax5.legend()

plt.tight_layout()

# --- SAUVEGARDE AUTOMATIQUE DANS LE DOSSIER DU RUN ---
run_dir = os.path.dirname(DATA_PATH)
output_name = os.path.join(run_dir, f"mz2_test_smoothing_{SMOOTHING_METHOD}_L={L}.pdf")

plt.savefig(output_name)
print(f"✅ Graphique comparatif sauvegardé sous : \n{output_name}")