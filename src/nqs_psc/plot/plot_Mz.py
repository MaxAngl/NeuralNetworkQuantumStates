import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Chargement des données
path_csv = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates/logs/Data_courbes_Mz_1D/L=4/Résultats.csv"
df = pd.read_csv(path_csv)

# 2. Configuration du graphique
plt.figure(figsize=(10, 6))

# 3. Tracé avec barres d'erreur
plt.errorbar(
    x=df['H'], 
    y=df['Magnetization'], 
    yerr=df['Magnetization_Error'], # L'erreur statistique
    fmt='-o',        # '-o' = ligne continue + points
    capsize=4,       # Ajoute des petits "chapeaux" aux barres d'erreur
    elinewidth=2,    # Épaisseur de la barre d'erreur
    label='Magnetization (RBM)',
    color='blue',
    ecolor='red'     # Couleur des barres d'erreur (pour bien les distinguer)
)

# 4. Esthétique
plt.xlabel('Champ transverse (h)')
plt.ylabel('Magnétisation moyenne <Mz>')
plt.title('Transition de phase Quantique (Ising 1D, L=4)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 5. Sauvegarde
dossier_sortie = os.path.dirname(path_csv)
chemin_image = os.path.join(dossier_sortie, "Graphique_Mz.png")

plt.savefig(chemin_image, dpi=300) # dpi=300 pour une bonne qualité
print(f"Graphique sauvegardé ici : {chemin_image}")

