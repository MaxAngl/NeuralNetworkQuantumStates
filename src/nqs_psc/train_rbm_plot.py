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

# Path vers le dossier où on conserve les runs
logs_path = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates/logs/Data_courbes_Mz_1D/L=81/Runs"

# Crée le dossier pour les logs s'il n'existe pas
os.makedirs(logs_path, exist_ok=True)

# Path vers le fichier .csv où on conserve le dictionnaire final
output_path = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates/logs/Data_courbes_Mz_1D/L=81/Résultats.csv"

# Crée le dossier pour le fichier CSV s'il n'existe pas (nécessaire !)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Taille du système

n_dim= 1
L = 81
J = -1
H_list = [0] #, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0]

#Paramètres RBM/optimisation

alpha = 5
lr= 0.01
diag_shift= 1e-3
n_chains = 300
n_samples =1000
n_iter =350

# Définition de l'hamiltonien

g = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

#Définition de la magnétisation
Mz = sum([nk.operator.spin.sigmaz(hi, i) for i in range(g.n_nodes)]) / g.n_nodes
# Magnétisation au carré (|Mz|^2)
Mz_sq = Mz * Mz

# Boucle sur les valeurs de H

for H in H_list:    
    ham = nk.operator.Ising(hi, g, J=J, h=H)

    # Définition du Modèle RBM

    model = nk.models.RBM(
        alpha=alpha,
        param_dtype=complex,
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
        "model": "RBM",
        "alpha": alpha,
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

    # Calcul de la Magnétisation sur l'état final
    mz_stats = vstate.expect(Mz)
    magnetization = mz_stats.mean.real
    magnetization_error = mz_stats.error_of_mean

    # Calcul de la magnétisation carrée <Mz^2>
    mz_sq_stats = vstate.expect(Mz_sq)
    magnetization_sq = mz_sq_stats.mean.real
    magnetization_sq_error = mz_sq_stats.error_of_mean
    
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
        "Magnetization": magnetization,
        "Magnetization_Error": magnetization_error,
        "Magnetization_Sq": magnetization_sq,
        "Magnetization_Sq_Error": magnetization_sq_error,
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

#PYTHONPATH=src python /users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates/src/nqs_psc/train_rbm_plot.py