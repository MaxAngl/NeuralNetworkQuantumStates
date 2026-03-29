import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 👇 Liste des tailles de système à analyser 👇
L_LIST = [16, 25, 36, 49, 64, 81]  # Ajoute ou retire tes L ici

# Dossier de base contenant les sous-dossiers "run_L=..."
BASE_DIR = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_finaux_disordered_1D"

DERIVATIVE_STEP = 3  # L'écartement pour la dérivée robuste

# ==========================================
# FONCTION DE DÉRIVÉE ROBUSTE
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
# 2. EXTRACTION ET TRAITEMENT DES DONNÉES
# ==========================================
print("⚙️ Extraction de la susceptibilité pour chaque taille L...")

# Dictionnaire pour stocker les résultats organisés par sigma
# Format : { sigma_1: {'L': [], 'mean': [], 'err': []}, sigma_2: ... }
results_by_sigma = {}
sigma_grid_ref = None

for L in L_LIST:
    data_path = os.path.join(BASE_DIR, f"run_L={L}", f"mz2_full_data_L={L}.npz")
    
    if not os.path.exists(data_path):
        print(f"⚠️ Fichier introuvable pour L={L}, ignoré. ({data_path})")
        continue
        
    print(f"📂 Chargement L={L}...")
    data = np.load(data_path)
    
    sigma_grid = data["sigma_grid"]
    h0_grid = data["h0_grid"]
    mz2_raw = data["mz2_raw"] 
    
    # Initialisation du dictionnaire avec les sigmas du premier fichier trouvé
    if sigma_grid_ref is None:
        sigma_grid_ref = sigma_grid
        for s in sigma_grid_ref:
            results_by_sigma[float(s)] = {'L': [], 'mean': [], 'err': []}

    # Calcul pour chaque sigma de ce fichier
    for idx_s, sigma in enumerate(sigma_grid):
        s_key = float(sigma)
        
        # Sécurité : si le sigma n'est pas dans notre liste de référence, on le zappe
        if s_key not in results_by_sigma:
            continue
            
        max_d_micro_list = []
        MAX_KEEP = mz2_raw.shape[2]
        
        for k in range(MAX_KEEP):
            curve_k = mz2_raw[idx_s, :, k]
            valid_mask = ~np.isnan(curve_k)
            
            if np.sum(valid_mask) > 2 * DERIVATIVE_STEP:
                h0_valid = h0_grid[valid_mask]
                mz2_valid = curve_k[valid_mask]
                
                dk_micro = robust_derivative(mz2_valid, h0_valid, step=DERIVATIVE_STEP)
                max_d_micro_list.append(np.max(dk_micro))
                
        # On ajoute le point (L, mean, err) pour ce sigma
        if len(max_d_micro_list) > 0:
            results_by_sigma[s_key]['L'].append(L)
            results_by_sigma[s_key]['mean'].append(np.mean(max_d_micro_list))
            results_by_sigma[s_key]['err'].append(np.std(max_d_micro_list, ddof=1) / np.sqrt(len(max_d_micro_list)))

# ==========================================
# 3. PLOTTING (Linéaire & Log-Log)
# ==========================================
if sigma_grid_ref is None:
    print("❌ Aucune donnée n'a pu être chargée. Vérifie tes chemins.")
    exit()

print("📈 Génération des graphiques (Finite Size Scaling)...")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

colors_sigma = plt.cm.viridis(np.linspace(0, 1, len(sigma_grid_ref)))

LW = 1.5
MS = 5.0
CAPSIZE = 3

# --- BOUCLE SUR LES SIGMAS POUR TRACER LES COURBES ---
for idx, sigma in enumerate(sigma_grid_ref):
    s_key = float(sigma)
    c = colors_sigma[idx]
    
    # On récupère les listes pour ce sigma, converties en arrays pour le plot
    L_arr = np.array(results_by_sigma[s_key]['L'])
    mean_arr = np.array(results_by_sigma[s_key]['mean'])
    err_arr = np.array(results_by_sigma[s_key]['err'])
    
    if len(L_arr) == 0:
        continue
        
    # Trie au cas où la liste L_LIST ne serait pas dans l'ordre croissant
    sort_idx = np.argsort(L_arr)
    L_arr, mean_arr, err_arr = L_arr[sort_idx], mean_arr[sort_idx], err_arr[sort_idx]

    # PANNEAU 1 : Échelle Linéaire
    axs[0].errorbar(L_arr, mean_arr, yerr=err_arr, fmt='-o', color=c, 
                    linewidth=LW, markersize=MS, capsize=CAPSIZE, label=rf"$\sigma = {sigma}$")
    
    # PANNEAU 2 : Échelle Log-Log
    axs[1].errorbar(L_arr, mean_arr, yerr=err_arr, fmt='-o', color=c, 
                    linewidth=LW, markersize=MS, capsize=CAPSIZE, label=rf"$\sigma = {sigma}$")

# --- FORMATAGE PANNEAU 1 (LINÉAIRE) ---
axs[0].set_xlabel(r"System Size $L$")
axs[0].set_ylabel(r"$\langle \max \left| \frac{\partial M_z^2}{\partial h_0} \right| \rangle$")
axs[0].set_title("Peak Susceptibility vs System Size (Linear)")
axs[0].grid(True, ls="--", alpha=0.4)
axs[0].legend(fontsize=9, title="Disorder strength")

# --- FORMATAGE PANNEAU 2 (LOG-LOG) ---
axs[1].set_xscale('log')
axs[1].set_yscale('log')
# On force les labels X en format standard plutôt qu'en puissances de 10 (ex: 16, 25, 36 au lieu de 10^1.5)
axs[1].set_xticks(L_LIST)
axs[1].get_xaxis().set_major_formatter(plt.ScalarFormatter()) 
axs[1].set_xlabel(r"System Size $L$")
axs[1].set_ylabel(r"$\langle \max \left| \frac{\partial M_z^2}{\partial h_0} \right| \rangle$")
axs[1].set_title("Peak Susceptibility vs System Size (Log-Log)")
axs[1].grid(True, which="both", ls="--", alpha=0.4)
axs[1].legend(fontsize=9, title="Disorder strength")

plt.tight_layout()

# Sauvegarde
output_name = os.path.join(BASE_DIR, "finite_size_scaling_susceptibility.pdf")
plt.savefig(output_name)
print(f"✅ Graphique final sauvegardé sous : \n{output_name}")