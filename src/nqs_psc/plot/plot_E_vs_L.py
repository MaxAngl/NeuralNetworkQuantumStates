import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---

# Vos chemins exacts basés sur vos messages précédents
base_dir = Path("/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates/logs/Energies_taille_paire")
file_pbc = base_dir / "pbc=True/Résultats.csv"
file_obc = base_dir / "pbc=False/Résultats.csv"

# Dossier de sauvegarde
save_dir = base_dir 

# --- FONCTION DE CHARGEMENT ---

def load_data(csv_path):
    """
    Charge le fichier CSV où la colonne 'L' est la taille et 'Energy' l'énergie totale.
    """
    if not csv_path.exists():
        print(f"❌ Fichier introuvable : {csv_path}")
        return None, None, None

    try:
        df = pd.read_csv(csv_path)
        
        # Vérification que la colonne L existe bien
        if 'L' not in df.columns:
            print(f"❌ Erreur : La colonne 'L' n'existe pas dans {csv_path.name}")
            print(f"   Colonnes trouvées : {df.columns.tolist()}")
            return None, None, None

        # Extraction des données
        L = df['L'].values
        E_total = df['Energy'].values
        E_var = df['Energy_Variance'].values
        
        # --- NORMALISATION ---
        # Si c'est une chaîne 1D : N = L
        # Si c'est une grille 2D : N = L**2
        # Vu vos valeurs (-5 pour L=4), c'est une chaîne 1D.
        N_sites = L 
        
        E_mean = E_total / N_sites
        E_error = np.sqrt(E_var) / N_sites
        
        print(f"✅ Chargé {csv_path.parent.name} : {len(L)} points (L={L.min()} à {L.max()})")
        return L, E_mean, E_error

    except Exception as e:
        print(f"❌ Erreur lecture {csv_path}: {e}")
        return None, None, None

# --- EXÉCUTION ---

print("Lecture des fichiers...")
L_pbc, E_pbc, Err_pbc = load_data(file_pbc)
L_obc, E_obc, Err_obc = load_data(file_obc)

# --- TRACÉ ---

if L_pbc is None and L_obc is None:
    print("Rien à tracer.")
else:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Courbe PBC (Conditions Périodiques)
    if L_pbc is not None:
        ax.errorbar(L_pbc, E_pbc, yerr=Err_pbc, 
                    label='PBC (Périodique)', 
                    marker='o', markersize=5, linestyle='-', linewidth=1.5,
                    color='tab:blue', capsize=3)

    # Courbe OBC (Conditions Ouvertes)
    if L_obc is not None:
        ax.errorbar(L_obc, E_obc, yerr=Err_obc, 
                    label='OBC (Ouvert)', 
                    marker='s', markersize=5, linestyle='--', linewidth=1.5,
                    color='tab:red', capsize=3)

    # Mise en forme
    ax.set_xlabel('Taille du système $L$', fontsize=14, fontweight='bold')
    ax.set_ylabel('Énergie par site $E/L$', fontsize=14, fontweight='bold')
    ax.set_title('Énergie par site en fonction de la taille $L$ pour des conditions aux bords périodiques et ouvertes', fontsize=16, pad=15)
    
    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    ax.minorticks_on()
    ax.legend(fontsize=12, framealpha=0.9)
    
    # Sauvegarde
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "Energy_vs_L_pbc=TrueOrFalse.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé avec succès : {output_path}")
    
    # plt.show()