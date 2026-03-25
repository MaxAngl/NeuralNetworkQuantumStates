import os
import json
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import netket as nk
import netket_foundational as nkf
# Assure-toi d'importer ta classe ViTFNQS correctement selon l'arborescence
from netket_foundational._src.model.vit import ViTFNQS

# --- CONFIGURATION ---
RUN_DIR = Path("/users/eleves-a/2024/rami.chagnaud/Documents/NeuralNetworkQuantumStates-1/Foundational/rami_perso/2D_FNQS/Run_2D_L4_FNQS")  # Adapte le chemin
CHECKPOINT_PATH = RUN_DIR / "state_390.nk"  # Adapte le nom
META_PATH = RUN_DIR / "meta.json"
DELTA_H = 1e-3


# --- CHARGEMENT META ---
with open(META_PATH, 'r') as f:
    meta = json.load(f)

L = meta["L"]
IS_2D = meta["n_dim"] == 2
n_spins = meta["nb_spins"]
J_val = meta["hamiltonian"]["J"]
vit_params = meta["vit_config"]

# --- INITIALISATION SYSTÈME ---
hi = nk.hilbert.Spin(0.5, n_spins)
ps = nkf.ParameterSpace(N=n_spins, min=0, max=10)

ma = ViTFNQS(
    num_layers=vit_params["num_layers"],
    d_model=vit_params["d_model"],
    heads=vit_params["heads"],
    b=vit_params["b"],
    L_eff=vit_params["L_eff"], 
    n_coups=ps.size, 
    complex=True, 
    disorder=True, 
    transl_invariant=False, 
    two_dimensional=IS_2D, 
)

sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=1)

import flax
import zipfile

if not zipfile.is_zipfile(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, 'rb') as f:
        state_dict = flax.serialization.msgpack_restore(f.read())
else:
    with zipfile.ZipFile(CHECKPOINT_PATH, 'r') as zf:
        file_list = zf.namelist()
        # Sélectionne le fichier msgpack le plus profond/long (souvent le bon dans l'arborescence)
        candidates = [f for f in file_list if f.endswith('.msgpack')]
        target_file = sorted(candidates, key=len)[-1]
        with zf.open(target_file) as f:
            state_dict = flax.serialization.msgpack_restore(f.read())

vars_dict = state_dict.get('variables', state_dict.get('model', {}).get('variables', state_dict))
vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)

# --- DEFINITION HAMILTONIEN ---
def get_hamiltonian_op(h_array):
    ha_X = sum(h_array[i] * nk.operator.spin.sigmax(hi, i) for i in range(n_spins))
    
    if IS_2D:
        ha_ZZ = sum(nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, (i % L + 1) % L + (i // L) * L) for i in range(n_spins))
        ha_ZZ += sum(nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, (i + L) % n_spins) for i in range(n_spins))
    else:
        ha_ZZ = sum(nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, (i + 1) % n_spins) for i in range(n_spins))
        
    return -ha_X - J_val * ha_ZZ

# --- EVALUATION DESORDONNEE ---
h0_list = np.linspace(0.5, 4.5, 40)
N_REP = 20 # Nombre de réalisations du désordre par point
sigma = meta["hamiltonian"]["sigma"]
rng = np.random.default_rng(42)

qfi_vmc_mean = []
qfi_vmc_std = []
qfi_exact_mean = []

for h0 in tqdm(h0_list, desc="Calcul QFI moyennée"):
    qfi_vmc_reps = []
    qfi_exact_reps = []
    
    for _ in range(N_REP):
        # 1. Tirage du désordre
        epsilon = rng.normal(loc=0.0, scale=sigma, size=n_spins)
        pars_base = jnp.array(h0 + epsilon)
        pars_shift = jnp.array(h0 + DELTA_H + epsilon) # On décale globalement de DELTA_H
        
        # --- VMC ---
        vs_base = vs.get_state(pars_base)
        vs_shift = vs.get_state(pars_shift)
        
        psi_vmc_base = nk.vqs.FullSumState(hi, vs_base.model, variables=vs_base.variables).to_array()
        psi_vmc_shift = nk.vqs.FullSumState(hi, vs_shift.model, variables=vs_shift.variables).to_array()
        
        psi_vmc_base = psi_vmc_base / np.linalg.norm(psi_vmc_base)
        psi_vmc_shift = psi_vmc_shift / np.linalg.norm(psi_vmc_shift)
        
        F_vmc = np.abs(np.vdot(psi_vmc_base, psi_vmc_shift))**2
        qfi_vmc_reps.append((1 - F_vmc) / (DELTA_H**2))

        # --- EXACT ---
        H_base = get_hamiltonian_op(pars_base)
        H_shift = get_hamiltonian_op(pars_shift)
        
        _, psi_ex_base = nk.exact.lanczos_ed(H_base, k=1, compute_eigenvectors=True)
        _, psi_ex_shift = nk.exact.lanczos_ed(H_shift, k=1, compute_eigenvectors=True)
        
        F_ex = np.abs(np.vdot(psi_ex_base[:, 0], psi_ex_shift[:, 0]))**2
        qfi_exact_reps.append((1 - F_ex) / (DELTA_H**2))
        
    qfi_vmc_mean.append(np.mean(qfi_vmc_reps))
    qfi_vmc_std.append(np.std(qfi_vmc_reps))
    qfi_exact_mean.append(np.mean(qfi_exact_reps))

# --- PLOT ---
qfi_vmc_mean = np.array(qfi_vmc_mean)
qfi_vmc_std = np.array(qfi_vmc_std)

plt.figure(figsize=(8, 5))
plt.plot(h0_list, qfi_exact_mean, label="Exact Lanczos (Moyenne)", color='blue', zorder=1)
plt.scatter(h0_list, qfi_vmc_mean, label="VMC FNQS (Moyenne)", color='red', marker='o', zorder=2)
plt.fill_between(h0_list, qfi_vmc_mean - qfi_vmc_std, qfi_vmc_mean + qfi_vmc_std, color='red', alpha=0.2)

plt.xlabel(r"Champ transverse moyen $h_0$")
plt.ylabel(r"Susceptibilité de fidélité moyennée $\overline{\chi_F}$")
plt.title(f"QFI désordonnée ($\sigma={sigma}$) en fonction de $h_0$ (Grille {L}x{L})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RUN_DIR / "QFI_disorder_plot.pdf")