import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def produce_plots(run_path):
    log_file = os.path.join(run_path, "log_data.json.log")
    if not os.path.exists(log_file):
        print(f"❌ Erreur : {log_file} introuvable.")
        return

    print(f"📖 Lecture de {log_file}...")
    with open(log_file, 'r') as f:
        data = json.load(f)

    # --- EXTRACTION SÉCURISÉE ---
    try:
        iters = np.array(data["Energy"]["iters"])
        energies = np.array(data["Energy"]["Mean"]["real"])
        variances = np.array(data["Energy"]["Variance"])
    except KeyError as e:
        print(f"❌ Erreur de lecture des données: Clé manquante {e}")
        return

    print(f"📊 Dimensions trouvées :")
    print(f"   - Energies  : {energies.shape}")
    print(f"   - Variances : {variances.shape}")

    # --- PLOT 1 : Convergence de l'Énergie ---
    plt.figure(figsize=(10, 6))
    plt.plot(iters, energies, color='blue', label='Mean Energy')
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Convergence de l'Énergie Globale")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(run_path, "energy_convergence_global.pdf"))
    print("✅ Plot Énergie généré.")

    # --- PLOT 2 : Convergence du V-score Global ---
    plt.figure(figsize=(10, 6))
    v_scores = variances / (energies**2)
    plt.plot(iters, v_scores, color='green', label='Global V-score')
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel(r"V-score (MC) $Var(E)/E^2$")
    plt.title("Convergence du V-score Global")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(run_path, "vscore_convergence_global.pdf"))
    print("✅ Plot V-score généré.")

if __name__ == "__main__":
    # N'oublie pas de vérifier ce chemin !
    run_path = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_finaux_disordered_1D/run_L=36"
    produce_plots(run_path)