import os
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import netket as nk
import netket_foundational as nkf
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from netket.utils import struct
import netket_pro.distributed as nkpd
from advanced_drivers._src.callbacks.base import AbstractCallback
from netket_foundational._src.model.vit import ViTFNQS


# --- CONFIGURATION ET DOSSIERS ---
output_dir = Path("/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational")
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = output_dir / "checkpoints"

# --- INITIALISATION SYSTÈME ---
k = jax.random.key(1)
hi = nk.hilbert.Spin(0.5, 10)
ps = nkf.ParameterSpace(N=1, min=0.8, max=1.2)

sampler=sa = nk.sampler.MetropolisLocal(hi, n_chains=5016)

# Créer d'abord le vstate avec la même architecture


# Charger les paramètres sauvegardés
vs= nk.vqs.MCState.load(r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/2/state_990.nk", new_seed=False)

def create_operator(params):
    assert params.shape == (1,)
    h = params[0]
    ha_X = sum(nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_ZZ = sum(
        nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
        for i in range(hi.size)
    )
    return -h * ha_X - ha_ZZ

# --- CALCULS EXACTS ---



Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))
Mz2_mat = (Mz @ Mz).to_sparse()



# --- 1. DEFINITION DES POINTS ---
# Tes 8 points d'entraînement exacts
h_train = np.linspace(0.8, 1.2, 8)

# Tes points de test (on en prend 50 pour avoir une belle courbe entre les deux)
# On élargit la plage comme tu voulais
h_test = np.linspace(0.5, 1.5, 50) 

# --- 2. CALCULS EXACTS (Sur toute la plage pour la ligne bleue) ---
h_all = np.sort(np.unique(np.concatenate([h_train, h_test])))
exact_data = {"h": [], "Energy": [], "Mz2": []}

for h_val in tqdm(h_all, desc="Calculs Exacts"):
    _ha = create_operator(h_val.reshape(-1))
    E0, psi0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=True)
    exact_data["h"].append(h_val)
    exact_data["Energy"].append(E0.item())
    exact_data["Mz2"].append((psi0.T.conj() @ (Mz2_mat @ psi0.reshape(-1))).item())

# --- 1. EVALUATION VMC (Train et Test séparés) ---
def evaluate_vmc(h_list):
    results = {"h": [], "Energy": [], "Mz2": []}
    for val in tqdm(h_list, desc="VMC Eval"):
        pars = jnp.array([val])
        _ha = create_operator(pars)
        _vs = vs.get_state(pars)
        _vs.sample()
        results["h"].append(val)
        results["Energy"].append(_vs.expect(_ha).Mean.real.item())
        results["Mz2"].append(_vs.expect(Mz @ Mz).Mean.real.item())
    return results

print("Évaluation des points d'entraînement...")
vmc_train = evaluate_vmc(h_train)

print("Évaluation des points de test...")
vmc_test = evaluate_vmc(h_test)

# --- 2. FONCTION POUR OBTENIR L'ERREUR RELATIVE ---
def get_relative_error(h_vals, vmc_vals, exact_h, exact_vals):
    errors = []
    for h, vmc_v in zip(h_vals, vmc_vals):
        # Trouver la valeur exacte correspondante pour ce h
        idx = np.argmin(np.abs(np.array(exact_h) - h))
        exact_v = exact_vals[idx]
        errors.append(np.abs((vmc_v - exact_v) / exact_v))
    return np.array(errors)

# Calcul des erreurs
err_e_train = get_relative_error(vmc_train["h"], vmc_train["Energy"], exact_data["h"], exact_data["Energy"])
err_e_test  = get_relative_error(vmc_test["h"], vmc_test["Energy"], exact_data["h"], exact_data["Energy"])

err_m_train = get_relative_error(vmc_train["h"], vmc_train["Mz2"], exact_data["h"], exact_data["Mz2"])
err_m_test  = get_relative_error(vmc_test["h"], vmc_test["Mz2"], exact_data["h"], exact_data["Mz2"])

# --- 3. TRACÉ (Grille 2x2) ---
fig, ax = plt.subplots(2, 2, figsize=(15, 12))

# --- LIGNE 1 : ENERGIE ---
# Valeurs
ax[0, 0].plot(exact_data["h"], exact_data["Energy"], color='blue', label="Exact", zorder=1)
ax[0, 0].scatter(vmc_test["h"], vmc_test["Energy"], color='orange', marker='x', s=40, label="VMC Test", zorder=2)
ax[0, 0].scatter(vmc_train["h"], vmc_train["Energy"], color='red', marker='o', s=60, edgecolors='black', label="VMC Train", zorder=3)
ax[0, 0].set_ylabel("Énergie")
ax[0, 0].set_title("Comparaison Énergie")

# Erreur Relative
ax[0, 1].scatter(vmc_test["h"], err_e_test, color='orange', marker='x', s=40, label="Err. Test")
ax[0, 1].scatter(vmc_train["h"], err_e_train, color='red', marker='o', s=60, edgecolors='black', label="Err. Train")
ax[0, 1].set_yscale('log') # Log scale pour mieux voir les petites erreurs
ax[0, 1].set_ylabel("Erreur Relative $|(E_{VMC}-E_{ex})/E_{ex}|$")
ax[0, 1].set_title("Précision de l'Énergie")

# --- LIGNE 2 : MAGNETISATION Mz2 ---
# Valeurs
ax[1, 0].plot(exact_data["h"], exact_data["Mz2"], color='blue', label="Exact", zorder=1)
ax[1, 0].scatter(vmc_test["h"], vmc_test["Mz2"], color='orange', marker='x', s=40, label="VMC Test", zorder=2)
ax[1, 0].scatter(vmc_train["h"], vmc_train["Mz2"], color='red', marker='o', s=60, edgecolors='black', label="VMC Train", zorder=3)
ax[1, 0].set_ylabel("$M_z^2$")
ax[1, 0].set_title("Comparaison $M_z^2$")

# Erreur Relative
ax[1, 1].scatter(vmc_test["h"], err_m_test, color='orange', marker='x', s=40)
ax[1, 1].scatter(vmc_train["h"], err_m_train, color='red', marker='o', s=60, edgecolors='black')
ax[1, 1].set_yscale('log')
ax[1, 1].set_ylabel("Erreur Relative $|(M_{VMC}-M_{ex})/M_{ex}|$")
ax[1, 1].set_title("Précision de $M_z^2$")

# Mise en forme commune
for a in ax.flat:
    a.axvspan(0.8, 1.2, color='gray', alpha=0.1)
    a.set_xlabel("Champ h")
    a.legend()
    a.grid(True, which="both", alpha=0.2)


plt.tight_layout()
plt.savefig(output_dir / "Foundational_extrapolation_test_with_errors.pdf")
print(f"Graphiques sauvegardés dans {output_dir}")


print(f"\n✅ Simulation terminée. Résultats sauvegardés dans le dossier : {output_dir}")