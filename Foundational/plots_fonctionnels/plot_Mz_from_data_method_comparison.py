import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# ==========================================
# 1. CONFIGURATION
# ==========================================
L = 36  
DATA_PATH = f"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_finaux_disordered_1D/run_L={L}/mz2_full_data_L={L}.npz"

SIGMA_EXAMPLE = 0.3  # Le paramètre de désordre utilisé pour montrer les courbes de dérivées en bas

# Paramètres des algorithmes de lissage
GAUSS_SIGMA = 1.5
SG_WINDOW = 5
SG_ORDER = 2

# Noms exacts qui apparaîtront dans la légende (en anglais)
METHODS = [
    'Standard Derivative (span=1)', 
    'Robust Derivative (span=2)', 
    'Robust Derivative (span=3)', 
    'Gaussian Smoothing', 
    'Savitzky-Golay Smoothing'
]

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
        # Sécurité pour ne pas déborder du tableau aux extrémités
        left = max(0, i - step)
        right = min(n - 1, i + step)
        
        if right == left:
            dy[i] = 0.0
        else:
            dy[i] = (y[right] - y[left]) / (x[right] - x[left])
            
    return np.abs(dy) # Valeur absolue pour avoir la susceptibilité

# ==========================================
# 2. CHARGEMENT DES DONNÉES
# ==========================================
print(f"📂 Chargement des données depuis : {DATA_PATH}")
data = np.load(DATA_PATH)

sigma_grid = data["sigma_grid"]
h0_grid = data["h0_grid"]
mz2_raw = data["mz2_raw"] 

# Trouver l'indice correspondant au sigma d'exemple choisi
idx_sigma_example = (np.abs(sigma_grid - SIGMA_EXAMPLE)).argmin()

# ==========================================
# 3. CALCULS STATISTIQUES (Macro & Micro)
# ==========================================
print("⚙️ Calcul de toutes les susceptibilités...")

mean_of_mz2 = []           
std_of_mz2 = []  

# Dictionnaires pour stocker les résultats de chaque méthode
macro_max = {m: [] for m in METHODS}
micro_mean_max = {m: [] for m in METHODS}
micro_err_max = {m: [] for m in METHODS}
example_curves = {m: [] for m in METHODS} 

for idx_s, sigma in enumerate(sigma_grid):
    # Moyenne et écart-type de Mz2 pour ce sigma
    mean_curve = np.nanmean(mz2_raw[idx_s], axis=1)
    std_curve = np.nanstd(mz2_raw[idx_s], axis=1, ddof=1)
    
    mean_of_mz2.append(mean_curve)
    std_of_mz2.append(std_curve)
    
    # --- A. MACROSCOPIQUE (Dérivée de la moyenne globale) ---
    d_macro = {
        'Standard Derivative (span=1)': robust_derivative(mean_curve, h0_grid, step=1),
        'Robust Derivative (span=2)': robust_derivative(mean_curve, h0_grid, step=2),
        'Robust Derivative (span=3)': robust_derivative(mean_curve, h0_grid, step=3),
        'Gaussian Smoothing': np.abs(np.gradient(gaussian_filter1d(mean_curve, sigma=GAUSS_SIGMA), h0_grid)),
        'Savitzky-Golay Smoothing': np.abs(np.gradient(savgol_filter(mean_curve, window_length=SG_WINDOW, polyorder=SG_ORDER), h0_grid))
    }
    
    for m in METHODS:
        macro_max[m].append(np.max(d_macro[m]))
        # Si c'est le sigma d'exemple, on sauvegarde la courbe complète pour le tracé
        if idx_s == idx_sigma_example:
            example_curves[m] = d_macro[m]
            
    # --- B. MICROSCOPIQUE (Moyenne des dérivées max par tirage) ---
    max_d_micro = {m: [] for m in METHODS}
    MAX_KEEP = mz2_raw.shape[2]
    
    for k in range(MAX_KEEP):
        curve_k = mz2_raw[idx_s, :, k]
        valid_mask = ~np.isnan(curve_k)
        
        # Savitzky-Golay a besoin d'au moins autant de points que sa taille de fenêtre
        if np.sum(valid_mask) >= max(6, SG_WINDOW):
            h0_valid = h0_grid[valid_mask]
            mz2_valid = curve_k[valid_mask]
            
            d_micro = {
                'Standard Derivative (span=1)': robust_derivative(mz2_valid, h0_valid, step=1),
                'Robust Derivative (span=2)': robust_derivative(mz2_valid, h0_valid, step=2),
                'Robust Derivative (span=3)': robust_derivative(mz2_valid, h0_valid, step=3),
                'Gaussian Smoothing': np.abs(np.gradient(gaussian_filter1d(mz2_valid, sigma=GAUSS_SIGMA), h0_valid)),
                'Savitzky-Golay Smoothing': np.abs(np.gradient(savgol_filter(mz2_valid, window_length=SG_WINDOW, polyorder=SG_ORDER), h0_valid))
            }
            
            for m in METHODS:
                max_d_micro[m].append(np.max(d_micro[m]))
                
    # Calcul de la moyenne et de l'erreur standard (SEM) pour ce sigma
    for m in METHODS:
        micro_mean_max[m].append(np.mean(max_d_micro[m]))
        micro_err_max[m].append(np.std(max_d_micro[m], ddof=1) / np.sqrt(len(max_d_micro[m])))

# ==========================================
# 4. PLOTTING (Mise en page avancée)
# ==========================================
print("📈 Génération de la figure du Grand Comparatif...")
fig = plt.figure(figsize=(20, 11))

# Grille de 2 lignes et 15 colonnes pour aligner 3 grands graphes en haut et 5 petits en bas
gs = fig.add_gridspec(2, 15, height_ratios=[1.2, 1], hspace=0.35, wspace=1.0)

colors_mz2 = plt.cm.viridis(np.linspace(0, 1, len(sigma_grid)))

# Dictionnaire des styles graphiques
style_dict = {
    'Standard Derivative (span=1)': {'color': 'red'},
    'Robust Derivative (span=2)': {'color': 'darkorange'},
    'Robust Derivative (span=3)': {'color': 'goldenrod'}, # Jaune doré pour plus de lisibilité
    'Gaussian Smoothing': {'color': 'blue'},
    'Savitzky-Golay Smoothing': {'color': 'purple'}
}

LW = 1.5 # Épaisseur des lignes principales

# --------------------------------------------------
# LIGNE DU HAUT : Les 3 bilans (5 colonnes chacun)
# --------------------------------------------------
ax_mz2 = fig.add_subplot(gs[0, 0:5])
for idx, sigma in enumerate(sigma_grid):
    c = colors_mz2[idx]
    ax_mz2.plot(h0_grid, mean_of_mz2[idx], marker='.', markersize=2, color=c, linewidth=1.0, alpha=0.8, label=rf"$\sigma = {sigma}$")
    ax_mz2.fill_between(h0_grid, mean_of_mz2[idx] - std_of_mz2[idx], mean_of_mz2[idx] + std_of_mz2[idx], color=c, alpha=0.15, edgecolor='none')
ax_mz2.set_xlabel(r"Transverse Field $h_0$")
ax_mz2.set_ylabel(r"$\langle M_z^2 \rangle$")
ax_mz2.set_title(r"Mean Squared Magnetization (Shaded = Sample Std Dev)")
ax_mz2.grid(True, ls="--", alpha=0.3)
ax_mz2.legend(fontsize=8, loc='upper right')

ax_macro = fig.add_subplot(gs[0, 5:10])
for m in METHODS:
    st = style_dict[m]
    ax_macro.plot(sigma_grid, macro_max[m], linestyle='-', color=st['color'], marker='o', label=m, linewidth=LW, markersize=4, alpha=0.9)
ax_macro.set_xlabel(r"Disorder strength $\sigma$")
ax_macro.set_ylabel(r"$\max \left| \frac{\partial \langle M_z^2 \rangle}{\partial h_0} \right|$")
ax_macro.set_title(r"Max of the Mean Derivative (Macroscopic)")
ax_macro.grid(True, ls="--", alpha=0.3)
ax_macro.legend(fontsize=9)

ax_micro = fig.add_subplot(gs[0, 10:15])
for m in METHODS:
    st = style_dict[m]
    ax_micro.errorbar(sigma_grid, micro_mean_max[m], yerr=micro_err_max[m], fmt='-o', color=st['color'], label=m, capsize=3, linewidth=LW, markersize=4, alpha=0.9)
ax_micro.set_xlabel(r"Disorder strength $\sigma$")
ax_micro.set_ylabel(r"$\langle \max \left| \frac{\partial M_z^2}{\partial h_0} \right| \rangle$")
ax_micro.set_title(r"Mean of the Max Derivatives (Microscopic)")
ax_micro.grid(True, ls="--", alpha=0.3)
ax_micro.legend(fontsize=9)

# --------------------------------------------------
# LIGNE DU BAS : Les 5 exemples de dérivées côte à côte (3 colonnes chacun)
# --------------------------------------------------
axs_d = []
for i, m in enumerate(METHODS):
    if i == 0:
        ax = fig.add_subplot(gs[1, i*3:(i+1)*3])
        ax.set_ylabel(r"Susceptibility (Derivative)")
    else:
        # On partage les axes X et Y avec le premier graphique pour une comparaison parfaite à l'œil
        ax = fig.add_subplot(gs[1, i*3:(i+1)*3], sharey=axs_d[0], sharex=axs_d[0])
        plt.setp(ax.get_yticklabels(), visible=False) # Masque les valeurs Y pour aérer la figure
    
    axs_d.append(ax)
    st = style_dict[m]
    
    ax.plot(h0_grid, example_curves[m], color=st['color'], linestyle='-', linewidth=1.2, marker='.', markersize=2, alpha=0.9)
    
    # Formatage du titre pour qu'il tienne bien dans les petits cadres
    title_split = m.replace(" Smoothing", "").replace(" (", "\n(")
    ax.set_title(f"{title_split}", fontsize=11, color=st['color'], fontweight='bold')
    ax.set_xlabel(r"$h_0$")
    ax.grid(True, ls="--", alpha=0.3)

# Titre global centré pour la ligne du bas
fig.text(0.5, 0.45, f"Comparison of Derivative Methods for a single disorder ($\\sigma = {sigma_grid[idx_sigma_example]}$)", ha='center', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajuste les marges pour laisser la place au titre du bas

# Sauvegarde automatique
run_dir = os.path.dirname(DATA_PATH)
output_name = os.path.join(run_dir, f"mz2_smoothing_comparison_L={L}.pdf")
plt.savefig(output_name)
print(f"✅ Graphique comparatif ultime sauvegardé sous : \n{output_name}")