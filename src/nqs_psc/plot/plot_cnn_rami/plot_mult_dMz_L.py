"""
Script pour tracer la susceptibilité magnétique dM_z/dH en fonction de H
avec propagation des erreurs et un gradient de couleur selon L
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Configuration
data_dir = Path("logs/Data_courbes_Mz_1D")

# Extraire toutes les valeurs de L disponibles
L_values = []
for folder in sorted(data_dir.iterdir()):
    if folder.is_dir() and folder.name.startswith("L="):
        L = int(folder.name.split("=")[1])
        L_values.append(L)

L_values.sort()
print(f"Valeurs de L trouvées : {L_values}")

# Préparer le gradient de couleur
cmap = cm.viridis  # Vous pouvez choisir: viridis, plasma, inferno, magma, coolwarm, etc.
norm = Normalize(vmin=min(L_values), vmax=max(L_values))

# Créer la figure
fig, ax = plt.subplots(figsize=(12, 8))

# Tracer pour chaque valeur de L
for L in L_values:
    # Charger les données
    csv_file = data_dir / f"L={L}" / "Résultats.csv"
    
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        
        # Calculer M_z/L (magnétisation moyenne par spin)
        H = df['H'].values
        Mz_mean = df['Magnetization_Sq'].values
        Mz_error = df['Magnetization_Sq_Error'].values
        
        # Calculer la dérivée dM_z/dH par différences finies centrées
        dMz_dH = np.zeros(len(H))
        dMz_dH_error = np.zeros(len(H))
        
        for i in range(len(H)):
            if i == 0:
                # Différence avant pour le premier point
                dH = H[i+1] - H[i]
                dMz_dH[i] = (Mz_mean[i+1] - Mz_mean[i]) / dH
                # Propagation d'erreur: sqrt((err1/dH)^2 + (err2/dH)^2)
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
        
        
        dMz_dH = abs(dMz_dH)
        # Obtenir la couleur du gradient
        color = cmap(norm(L))
        
        # Tracer avec barres d'erreur
        ax.errorbar(H, dMz_dH, yerr=dMz_dH_error, 
                   label=f'L={L}', 
                   marker='o', 
                   markersize=4,
                   capsize=3,
                   capthick=1,
                   linewidth=1.5,
                   alpha=0.8,
                   color=color)
        
        print(f"L={L}: {len(H)} points tracés")
    else:
        print(f"Fichier non trouvé : {csv_file}")

# Configuration du graphique
ax.set_xlabel('Champ magnétique H', fontsize=14, fontweight='bold')
ax.set_ylabel('$|d{M_z}^{2}/dH|$', fontsize=14, fontweight='bold')
ax.set_title('Dérivée de la magnétisation en fonction du champ magnétique\npour différentes tailles de système 1D', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', framealpha=0.9, fontsize=10)

# Ajouter une barre de couleur pour montrer le gradient de L
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Nombre de spins L')
cbar.set_label('Nombre de spins L', fontsize=12, fontweight='bold')

plt.tight_layout()

# Sauvegarder le graphique
output_file = "dMz^2dH_1D_graph.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nGraphique sauvegardé : {output_file}")

plt.show()
