"""
Script pour tracer l'énergie moyenne E/L en fonction de H
avec propagation des erreurs et un gradient de couleur selon L
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Configuration
data_dir = Path("logs/rami/CNN_2D")

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
        
        # Calculer E/L (énergie moyenne par spin)
        H = df['H'].values
        E_mean = df['Energy'].values / L
        E_error = np.sqrt(df['Energy_Variance'].values) / L
        
        # Obtenir la couleur du gradient
        color = cmap(norm(L))
        
        # Tracer avec barres d'erreur
        ax.errorbar(H, E_mean, yerr=E_error, 
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
ax.set_ylabel('Énergie moyenne $E/L$', fontsize=14, fontweight='bold')
ax.set_title('Énergie moyenne par spin en fonction du champ magnétique\npour différentes tailles de système 1D',
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
output_file = "energy_per_spin_vs_H_gradient.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nGraphique sauvegardé : {output_file}")

plt.show()
