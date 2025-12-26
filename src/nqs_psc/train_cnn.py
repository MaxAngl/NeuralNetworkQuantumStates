import netket as nk
from nqs_psc.utils import save_run
import numpy as np
from functools import partial
import copy
import flax.linen as nn
import jax
import jax.numpy as jnp
import nqxpack
import time
import os
import pandas as pd
import ansatz

# Path vers le dossier où on conserve les runs
logs_path = r"/users/eleves-a/2024/rami.chagnaud/Documents/NeuralNetworkQuantumStates/logs/rami/CNN_2D/L=4/Runs"

# Crée le dossier pour les logs s'il n'existe pas
os.makedirs(logs_path, exist_ok=True)

# Path vers le fichier .csv où on conserve le dictionnaire final
output_path = r"/users/eleves-a/2024/rami.chagnaud/Documents/NeuralNetworkQuantumStates/logs/rami/CNN_2D/L=4/Résultats.csv"

# Crée le dossier pour le fichier CSV s'il n'existe pas (nécessaire !)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Taille du système

n_dim= 2
L = 4
H = 2.6
a1 = np.array([1.0, 0.0])
a2 = np.array([0.0, 1.0])
J = -1
H_list = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0]

#Paramètres CNN/optimisation

lattice = nk.graph.Lattice(basis_vectors=[a1, a2], extent=(L, L), pbc=True)
kernel_size = ((2,2),(2,2))
channel = (5, 5)
lr= 0.0001
diag_shift= 1e-3
n_chains = 300
n_samples =1000
n_iter =400

# Définition de l'hamiltonien

g = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)


ham = nk.operator.Ising(hi, g, J=J, h=H)

    # Définition du Modèle CNN

model = ansatz.CNN(
        lattice=lattice,
        kernel_size=kernel_size,
        channels=channel,
        param_dtype=complex
    )

sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=451)

# Optimisation

optimizer = nk.optimizer.Sgd(learning_rate=lr)
gs = nk.driver.VMC_SR(
        ham,
        optimizer,
        variational_state=vstate,
        diag_shift=diag_shift,
    )

# création du logger
log = nk.logging.RuntimeLog()

# meta identique
meta = {
        "L": L,
        "graph": "Hypercube",
        "n_dim": n_dim,
        "pbc": True,
        "hamiltonian": {"type": "Ising", "J": J, "h": H},
        "model": "CNN",
        "kernel_size": "kernel_size",
        "channels": "channel",
        "sampler": {"type": "MetropolisLocal", "n_chains": n_chains, "n_samples": n_samples},
        "optimizer": {"type": "SGD", "lr": lr, "diag_shift": diag_shift},
        "n_iter": n_iter,
        "exact": "?",
    }

# Créer le dossier de run avant l'entraînement
run_dir = save_run(log, meta, create_only=True, base_dir=logs_path)

# One or more logger objects must be passed to the keyword argument `out`.
def save_vstate(step, logdata, driver):
        if step % 50 == 0 or step == n_iter - 1:
            nqxpack.save(driver.state, f"{run_dir}/vstate_step_{step}")
        return True

    # Mesurer le temps d'exécution
start_time = time.time()
gs.run(n_iter=n_iter, out=log, callback=(save_vstate,))
execution_time = time.time() - start_time

# Ajouter le temps d'exécution aux métadonnées
meta["execution_time_seconds"] = execution_time

# Sauvegarder les logs finaux
save_run(log, meta, run_dir=run_dir, base_dir=logs_path)
    
# Récupération de l'Énergie et Variance (pour le V-score)
energy_stats = vstate.expect(ham) 
energy_mean = energy_stats.mean.real
energy_variance = energy_stats.variance # Var(H)
    
# Calcul du V-score : Var(H) / E^2
v_score = energy_variance / (energy_mean**2)

# Sauvegarde
# 1. On crée un dictionnaire pour cette itération seulement
current_data = {
        "H": H,
        "V_Score": v_score,
        "Energy": energy_mean,
        "Energy_Variance": energy_variance
    }

# 2. On le transforme en DataFrame (une seule ligne)
df_step = pd.DataFrame([current_data])

# 3. Logique pour l'entête (Header)
# On écrit les titres seulement si le fichier n'existe pas ou s'il est vide
fichier_existe = os.path.isfile(output_path)
fichier_vide = fichier_existe and os.path.getsize(output_path) == 0
write_header = not fichier_existe or fichier_vide

# 4. Sauvegarde immédiate en mode "append" ('a')
df_step.to_csv(
        output_path, 
        index=False, 
        mode='a', 
        header=write_header
    )

print(f"Données pour H={H} sauvegardées dans le CSV.")