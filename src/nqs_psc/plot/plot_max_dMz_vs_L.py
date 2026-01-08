"""
Script pour tracer le maximum de la susceptibilité magnétique dM_z/dH 
en fonction du nombre de spins L
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
data_dir = Path("logs/Data_courbes_Mz_2D")

# Extraire toutes les valeurs de L disponibles
L_values = []
for folder in sorted(data_dir.iterdir()):
    if folder.is_dir() and folder.name.startswith("L="):
        L = int(folder.name.split("=")[1])
        L_values.append(L)

L_values.sort()
print(f"Valeurs de L trouvées : {L_values}")

# Stocker les maxima pour chaque L
max_dMz_dH = []
max_dMz_dH_error = []
L_plot = []

# Calculer le maximum pour chaque valeur de L
for L in L_values:
    # Charger les données
    csv_file = data_dir / f"L={L}" / "Résultats.csv"
    
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        
        # Calculer M_z
        H = df['H'].values
        Mz_mean = df['Magnetization_Sq'].values
        Mz_error = df['Magnetization_Sq_Error'].values
        
        # Filtrer les points trop proches pour éviter les oscillations
        min_dH = 0.05  # Distance minimale entre les points
        mask = [0]  # Indices des points à garder, commencer avec le premier point
        for i in range(1, len(H)):
            if H[i] - H[mask[-1]] >= min_dH:
                mask.append(i)
        
        # Appliquer le filtre
        H = H[mask]
        Mz_mean = Mz_mean[mask]
        Mz_error = Mz_error[mask]
        
        # Calculer la dérivée dM_z/dH par différences finies centrées
        dMz_dH = np.zeros(len(H))
        dMz_dH_error = np.zeros(len(H))
        
        for i in range(len(H)):
            if i == 0:
                # Différence avant pour le premier point
                dH = H[i+1] - H[i]
                dMz_dH[i] = (Mz_mean[i+1] - Mz_mean[i]) / dH
                dMz_dH_error[i] = np.sqrt(Mz_error[i]**2 + Mz_error[i+1]**2) / dH
            elif i == len(H) - 1:
                # Différence arrière pour le dernier point
                dH = H[i] - H[i-1]
                dMz_dH[i] = (Mz_mean[i] - Mz_mean[i-1]) / dH
                dMz_dH_error[i] = np.sqrt(Mz_error[i]**2 + Mz_error[i-1]**2) / dH
            else:
                # Différence centrale pour les points intermédiaires
                dH = H[i+1] - H[i-1]
                dMz_dH[i] = (Mz_mean[i+1] - Mz_mean[i-1]) / dH
                dMz_dH_error[i] = np.sqrt(Mz_error[i-1]**2 + Mz_error[i+1]**2) / dH
        
        dMz_dH = np.abs(dMz_dH)
        
        # Trouver le maximum
        max_idx = np.argmax(dMz_dH)
        max_val = dMz_dH[max_idx]
        max_err = dMz_dH_error[max_idx]
        
        L_plot.append(L)
        max_dMz_dH.append(max_val)
        max_dMz_dH_error.append(max_err)
        
        print(f"L={L}: max(|dMz/dH|) = {max_val:.4f} ± {max_err:.4f} à H = {H[max_idx]:.4f}")
    else:
        print(f"Fichier non trouvé : {csv_file}")

# Convertir en arrays numpy
L_plot = np.array(L_plot)
max_dMz_dH = np.array(max_dMz_dH)
max_dMz_dH_error = np.array(max_dMz_dH_error)

# Créer la figure
fig, ax = plt.subplots(figsize=(10, 7))

# Tracer avec barres d'erreur
ax.errorbar(L_plot, max_dMz_dH, yerr=max_dMz_dH_error,
           marker='o', 
           markersize=8,
           capsize=5,
           capthick=2,
           linewidth=2,
           color='royalblue',
           ecolor='darkblue',
           label='Maximum de |dM_z²/dH|')

# Configuration du graphique
ax.set_xlabel('Nombre de spins L', fontsize=14, fontweight='bold')
ax.set_ylabel('max(|dM_z²/dH|)', fontsize=14, fontweight='bold')
ax.set_title('Maximum de la susceptibilité magnétique\nen fonction de la taille du système', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', framealpha=0.9, fontsize=11)

# Ajuster les limites des axes
ax.set_xlim(min(L_plot) - 0.5, max(L_plot) + 0.5)

plt.tight_layout()

# Sauvegarder le graphique
output_file = "max_dMz_vs_L.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nGraphique sauvegardé : {output_file}")

plt.show()
