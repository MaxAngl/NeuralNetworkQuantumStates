import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Le chemin vers votre dossier de run
RUN_DIRECTORY = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/logs/run_2026-02-04_17-13-25" 

# ==========================================
# 2. FONCTIONS UTILITAIRES
# ==========================================

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Erreur : Le fichier {filepath} est introuvable.")
        sys.exit(1)
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_energy_convergence(run_path):
    # Vérification du dossier
    if not os.path.exists(run_path):
        print(f"❌ Le dossier spécifié n'existe pas : {run_path}")
        return

    # Définition des chemins
    meta_path = os.path.join(run_path, "meta.json")
    
    # ICI : On cible spécifiquement le fichier .json.log
    log_path = os.path.join(run_path, "log_data.json.log")

    # Fallback au cas où le nom changerait, mais on priorise votre demande
    if not os.path.exists(log_path):
        print(f"⚠️ 'log_data.json.log' introuvable, essai avec 'log_data.log'...")
        log_path = os.path.join(run_path, "log_data.log")

    print(f"📂 Dossier : {run_path}")
    print(f"📄 Fichier log utilisé : {log_path}")

    # Chargement
    meta = load_json(meta_path)
    log_data = load_json(log_path)
    
    # Récupération des paramètres
    try:
        h0_list = meta["hamiltonian"]["h0_train_list"]
        n_replicas_per_h0 = meta["n_replicas_per_h0"]
        L = meta["L"]
    except KeyError as e:
        print(f"❌ Erreur de format dans meta.json : clé manquante {e}")
        return

    # Vérification des énergies exactes
    if "exact_energies" not in log_data:
        print("❌ Les énergies exactes ('exact_energies') sont absentes du log.")
        return

    exact_energies = log_data["exact_energies"]
    
    # Récupération historique
    if "ham" in log_data:
        ham_history = log_data["ham"]
    elif "Energy" in log_data:
        ham_history = log_data["Energy"]
    else:
        print("❌ Données 'ham' ou 'Energy' introuvables.")
        return

    # --- Configuration du Plot (Magma) ---
    norm = mcolors.Normalize(vmin=min(h0_list), vmax=max(h0_list))
    cmap = plt.get_cmap('magma')
    
    plt.figure(figsize=(10, 7))
    print("📈 Génération du graphique...")

    labeled_h0 = set()

    for i in range(len(ham_history)):
        # Calcul de l'index h0 correspondant
        h0_idx = i // n_replicas_per_h0
        if h0_idx >= len(h0_list): 
            continue
            
        current_h0 = h0_list[h0_idx]
        color = cmap(norm(current_h0))

        replica_data = ham_history[i]
        
        try:
            iters = np.array(replica_data["iters"])
            energies = np.array(replica_data["Mean"])
            # Gestion complexe -> réel
            if len(energies.shape) > 1 and energies.shape[1] == 2: 
                 energies = energies[:, 0]
            else:
                energies = np.real(energies)
        except KeyError:
            continue

        e_exact = exact_energies.get(str(i))
        if e_exact is None:
            continue

        # Erreur relative
        rel_error = np.abs((energies - e_exact) / e_exact)

        # Légende
        label = f"$h_0={current_h0}$" if current_h0 not in labeled_h0 else None
        if label:
            labeled_h0.add(current_h0)

        plt.plot(iters, rel_error, color=color, lw=1.5, alpha=0.6, label=label)

    # --- Finitions ---
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Itération", fontsize=12)
    plt.ylabel(r"Erreur Relative Energie $\Delta E / E_{exact}$", fontsize=12)
    plt.title(f"Convergence - Modèle ViTFNQS (L={L})", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(r"Champ moyen $h_0$", fontsize=12)

    # SAUVEGARDE DANS LE DOSSIER DU RUN
    output_filename = "energy_convergence_magma.pdf"
    output_path = os.path.join(run_path, output_filename)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Graphique sauvegardé avec succès : {output_path}")
    plt.show()

# ==========================================
# 3. EXÉCUTION
# ==========================================
if __name__ == "__main__":
    plot_energy_convergence(RUN_DIRECTORY)