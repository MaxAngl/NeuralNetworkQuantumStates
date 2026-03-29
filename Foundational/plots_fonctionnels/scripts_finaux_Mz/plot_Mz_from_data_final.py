import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
L = 64 
DATA_PATH = f"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_finaux_disordered_1D/run_L={L}/mz2_full_data_L={L}.npz"

DERIVATIVE_STEP = 3  # L'écartement choisi pour la dérivée robuste

# ==========================================
# FONCTION DE DÉRIVÉE ROBUSTE
# ==========================================
def robust_derivative(y, x, step=1):
    """
    Calcule la dérivée numérique de y par rapport à x en utilisant 
    un écart de 'step' points pour s'affranchir du bruit stochastique à haute fréquence.
    """
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
# 2. CHARGEMENT DES DONNÉES
# ==========================================
print(f"📂 Chargement des données depuis : {DATA_PATH}")
data = np.load(DATA_PATH)

sigma_grid = data["sigma_grid"]
h0_grid = data["h0_grid"]
mz2_raw = data["mz2_raw"] 

# ==========================================
# 3. CALCULS STATISTIQUES
# ==========================================
print(f"⚙️ Traitement des données (Dérivée Robuste avec step={DERIVATIVE_STEP})...")

mean_of_mz2 = []           
std_of_mz2 = []  
deriv_of_mean = []

micro_mean_max = []
micro_err_max = []

for idx_s, sigma in enumerate(sigma_grid):
    # --- A. Mz2 Moyen et Dérivée de la moyenne ---
    mean_curve = np.nanmean(mz2_raw[idx_s], axis=1)
    std_curve = np.nanstd(mz2_raw[idx_s], axis=1, ddof=1)
    
    mean_of_mz2.append(mean_curve)
    std_of_mz2.append(std_curve)
    
    # On stocke la dérivée robuste de la courbe moyenne pour le Panneau 2
    deriv_of_mean.append(robust_derivative(mean_curve, h0_grid, step=DERIVATIVE_STEP))
            
    # --- B. Susceptibilité Microscopique (Tirage par tirage) ---
    max_d_micro_list = []
    MAX_KEEP = mz2_raw.shape[2]
    
    for k in range(MAX_KEEP):
        curve_k = mz2_raw[idx_s, :, k]
        valid_mask = ~np.isnan(curve_k)
        
        # On s'assure d'avoir assez de points valides pour calculer la dérivée avec ce step
        if np.sum(valid_mask) > 2 * DERIVATIVE_STEP:
            h0_valid = h0_grid[valid_mask]
            mz2_valid = curve_k[valid_mask]
            
            dk_micro = robust_derivative(mz2_valid, h0_valid, step=DERIVATIVE_STEP)
            max_d_micro_list.append(np.max(dk_micro))
                
    # Statistique finale pour le Panneau 3 (Moyenne et Erreur Standard)
    micro_mean_max.append(np.mean(max_d_micro_list))
    micro_err_max.append(np.std(max_d_micro_list, ddof=1) / np.sqrt(len(max_d_micro_list)))

# ==========================================
# 4. PLOTTING (1 ligne, 3 colonnes)
# ==========================================
print("📈 Génération de la figure finale...")
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

colors_mz2 = plt.cm.viridis(np.linspace(0, 1, len(sigma_grid)))

# Styles globaux pour les courbes superposées (Panneaux 1 et 2)
LW = 1.0
MS = 2.0
ALPHA_LINE = 0.85
ALPHA_FILL = 0.15

# --- PANNEAU 1 : Mz^2 vs h0 ---
ax = axs[0]
for idx, sigma in enumerate(sigma_grid):
    c = colors_mz2[idx]
    ax.plot(h0_grid, mean_of_mz2[idx], marker='.', markersize=MS, color=c, linewidth=LW, alpha=ALPHA_LINE, label=rf"$\sigma = {sigma}$")
    ax.fill_between(h0_grid, mean_of_mz2[idx] - std_of_mz2[idx], mean_of_mz2[idx] + std_of_mz2[idx], color=c, alpha=ALPHA_FILL, edgecolor='none')
ax.set_xlabel(r"Transverse Field $h_0$")
ax.set_ylabel(r"$\langle M_z^2 \rangle$")
ax.set_title(r"Mean Squared Magnetization")
ax.grid(True, ls="--", alpha=0.3)
ax.legend(fontsize=9, loc='upper right')

# --- PANNEAU 2 : Dérivées vs h0 ---
ax = axs[1]
for idx, sigma in enumerate(sigma_grid):
    c = colors_mz2[idx]
    ax.plot(h0_grid, deriv_of_mean[idx], marker='.', markersize=MS, color=c, linewidth=LW, alpha=ALPHA_LINE)
ax.set_xlabel(r"Transverse Field $h_0$")
ax.set_ylabel(r"Susceptibility (Robust Derivative)")
ax.set_title(f"Derivative of Mean Mz² (span={DERIVATIVE_STEP})")
ax.grid(True, ls="--", alpha=0.3)

# --- PANNEAU 3 : Moyenne des Max vs Sigma ---
ax = axs[2]
# Trait fin rouge, petits points, barres d'erreur bien visibles
ax.errorbar(sigma_grid, micro_mean_max, yerr=micro_err_max, 
            fmt='-o', color='red', markerfacecolor='red', markeredgecolor='red',
            linewidth=1.0, markersize=3, elinewidth=1.0, capsize=3, alpha=0.9)
ax.set_xlabel(r"Disorder strength $\sigma$")
ax.set_ylabel(r"$\langle \max \left| \frac{\partial M_z^2}{\partial h_0} \right| \rangle$")
ax.set_title(r"Mean of the Max Derivatives (Microscopic)")
ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()

# Sauvegarde
run_dir = os.path.dirname(DATA_PATH)
output_name = os.path.join(run_dir, f"mz2_final_analysis_L={L}.pdf")
plt.savefig(output_name)
print(f"✅ Graphique final sauvegardé sous : \n{output_name}")