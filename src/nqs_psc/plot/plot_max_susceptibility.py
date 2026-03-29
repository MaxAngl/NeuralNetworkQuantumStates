import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Chemin de base contenant les dossiers L=...
BASE_DIR = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/logs/Data_courbes_Mz_1D"

# Liste de tes tailles de système
L_LIST = [4, 9, 16, 25, 36, 49, 64, 81, 100] 

# Seuil à partir duquel on réalise le fit linéaire
FIT_THRESHOLD = 4

# ==========================================
# 2. EXTRACTION ET CALCULS
# ==========================================
print(f"📂 Analyse des fichiers CSV dans : {BASE_DIR}")

valid_L = []
max_derivatives = []

for L in L_LIST:
    file_path = os.path.join(BASE_DIR, f"L={L}", "Résultats.csv")
    
    if not os.path.exists(file_path):
        print(f"⚠️  Fichier introuvable pour L={L}, ignoré.")
        continue
        
    print(f"▶ Traitement de la taille L={L}...")
    
    # Lecture du CSV avec Pandas
    df = pd.read_csv(file_path)
    
    # Extraction des colonnes (H et Mz^2)
    h_grid = df['H'].values
    mz2_curve = df['Magnetization_Sq'].values
    
    # Calcul de la dérivée (Susceptibilité)
    susceptibility = np.abs(np.gradient(mz2_curve, h_grid))
    
    # Extraction du maximum
    max_chi = np.max(susceptibility)
    
    valid_L.append(L)
    max_derivatives.append(max_chi)

# Tri par L croissant
sort_idx = np.argsort(valid_L)
valid_L = np.array(valid_L)[sort_idx]
max_derivatives = np.array(max_derivatives)[sort_idx]

# ==========================================
# 3. FIT LINÉAIRE (Loi d'échelle)
# ==========================================
# On isole les points où L >= FIT_THRESHOLD
mask_fit = valid_L >= FIT_THRESHOLD
L_fit = valid_L[mask_fit]
chi_fit = max_derivatives[mask_fit]

slope, intercept = None, None
if len(L_fit) >= 2:
    # Régression linéaire sur les logarithmes : log(chi) = slope * log(L) + intercept
    coeffs = np.polyfit(np.log(L_fit), np.log(chi_fit), 1)
    slope = coeffs[0]      # L'exposant critique gamma/nu
    intercept = coeffs[1]  # Le log de l'amplitude
    
    # Création de la droite de fit pour l'affichage (loi de puissance)
    fit_line = np.exp(intercept) * (L_fit ** slope)
else:
    print(f"⚠️  Pas assez de points pour L >= {FIT_THRESHOLD} afin de faire le fit.")

# ==========================================
# 4. PLOTTING (Linéaire & Log-Log)
# ==========================================
if len(valid_L) == 0:
    print("❌ Aucune donnée n'a été trouvée. Vérifie ton BASE_DIR.")
    exit()

print("\n📈 Génération de la figure de Finite Size Scaling...")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Couleurs et style
COLOR = 'dodgerblue'
MARKER = 'o'
LW = 1.5
MS = 6

# --- PANNEAU 1 : Échelle Linéaire ---
axs[0].plot(valid_L, max_derivatives, marker=MARKER, color=COLOR, linewidth=LW, markersize=MS)
axs[0].set_xlabel(r"System Size $L$")
axs[0].set_ylabel(r"$\max \left| \frac{\partial \langle M_z^2 \rangle}{\partial H} \right|$")
axs[0].set_title("Peak Susceptibility vs System Size (Pure System)")
axs[0].grid(True, ls="--", alpha=0.5)

# --- PANNEAU 2 : Échelle Log-Log ---
axs[1].loglog(valid_L, max_derivatives, marker=MARKER, color=COLOR, linewidth=LW, markersize=MS, label="Données VMC")

# Ajout du Fit sur le graphe Log-Log si calculé
if slope is not None:
    axs[1].loglog(L_fit, fit_line, color='crimson', linestyle='--', linewidth=2, 
                  label=rf"Fit ($L \geq {FIT_THRESHOLD}$) : $\gamma/\nu \approx {slope:.3f}$")
    axs[1].legend(fontsize=11)

axs[1].set_xlabel(r"System Size $L$")
axs[1].set_ylabel(r"$\max \left| \frac{\partial \langle M_z^2 \rangle}{\partial H} \right|$")
axs[1].set_title("Peak Susceptibility vs System Size (Log-Log)")
axs[1].grid(True, which="both", ls="--", alpha=0.5)

# Astuce d'affichage pour l'axe X du log-log (pour voir les vrais L)
axs[1].set_xticks(valid_L)
axs[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())

plt.tight_layout()

# Sauvegarde
output_name = os.path.join(BASE_DIR, "pure_system_finite_size_scaling_fit_all_L.pdf")
plt.savefig(output_name)
print(f"✅ Graphique final sauvegardé sous : \n{output_name}")
if slope is not None:
    print(f"📊 Exposant critique estimé (gamma/nu) : {slope:.4f}")