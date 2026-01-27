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
output_dir = Path("/users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates/Foundational")
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = output_dir / "checkpoints"

# --- INITIALISATION SYSTÈME ---
k = jax.random.key(1)
hi = nk.hilbert.Spin(0.5, 10)
ps = nkf.ParameterSpace(N=1, min=0.8, max=1.2)

sampler=sa = nk.sampler.MetropolisLocal(hi, n_chains=5016)

# Créer d'abord le vstate avec la même architecture


# Charger les paramètres sauvegardés
vs= nk.vqs.MCState.load(r"/users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates/Foundational/2/state_990.nk", new_seed=False)

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

h_exact_range = np.linspace(0.8, 1.2, 40)
exact_data = {"h": [], "Energy": [], "Mz2": []}

for h_val in tqdm(h_exact_range, desc="Calculs Exacts"):
    _ha = create_operator(h_val.reshape(-1))
    E0, psi0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=True)
    psi0 = psi0.reshape(-1)
    exact_data["h"].append(h_val)
    exact_data["Energy"].append(E0.item())
    exact_data["Mz2"].append((psi0.T.conj() @ (Mz2_mat @ psi0)).item())

# --- ÉVALUATION VMC FINALE ---
vmc_final = {"h": [], "Energy": [], "Mz2": []}
for pars in tqdm(jnp.linspace(0.8, 1.2, 40).reshape(-1, 1), desc="Évaluation VMC"):
    _ha = create_operator(pars)
    _vs = vs.get_state(pars)
    _vs.sample()
    vmc_final["h"].append(pars[0].item())
    vmc_final["Energy"].append(_vs.expect(_ha).Mean.real)
    vmc_final["Mz2"].append(_vs.expect(Mz @ Mz).Mean.real)


# 1. Energie et Mz2
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(exact_data["h"], exact_data["Energy"], label="Exact")
ax[0].plot(vmc_final["h"], vmc_final["Energy"], "x", label="VMC")
ax[0].set_xlabel("h")
ax[0].set_ylabel("Energy")
ax[0].legend()

ax[1].plot(exact_data["h"], exact_data["Mz2"], label="Exact")
ax[1].plot(vmc_final["h"], vmc_final["Mz2"], "x", label="VMC")
ax[1].set_xlabel("h")
ax[1].set_ylabel("Mz2")
ax[1].legend()
plt.savefig(output_dir / "physics_results.pdf")


print(f"\n✅ Simulation terminée. Résultats sauvegardés dans le dossier : {output_dir}")