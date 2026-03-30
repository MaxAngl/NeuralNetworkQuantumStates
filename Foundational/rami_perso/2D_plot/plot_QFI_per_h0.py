import os
import json
import zipfile
import flax
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS

# --- CONFIGURATION ---
RUN_DIR = Path("/users/eleves-a/2024/rami.chagnaud/Documents/NeuralNetworkQuantumStates-1/Foundational/rami_perso/2D_FNQS/Run_2D_L6_FNQS")  # Adapte le chemin
CHECKPOINT_PATH = RUN_DIR  / "state_390.nk"  # Adapte le nom
META_PATH = RUN_DIR / "meta.json"
DELTA_H = 1e-3
N_SAMPLES = 4096  # Augmenté pour réduire la variance MC

# --- CHARGEMENT META ---
with open(META_PATH, 'r') as f:
    meta = json.load(f)

L = meta["L"]
IS_2D = meta["n_dim"] == 2
n_spins = meta["nb_spins"]
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
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=N_SAMPLES)

# --- CHARGEMENT DES POIDS ---
if not zipfile.is_zipfile(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, 'rb') as f:
        state_dict = flax.serialization.msgpack_restore(f.read())
else:
    with zipfile.ZipFile(CHECKPOINT_PATH, 'r') as zf:
        candidates = [f for f in zf.namelist() if f.endswith('.msgpack')]
        target_file = sorted(candidates, key=len)[-1]
        with zf.open(target_file) as f:
            state_dict = flax.serialization.msgpack_restore(f.read())

vars_dict = state_dict.get('variables', state_dict.get('model', {}).get('variables', state_dict))
vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)

# --- FONCTION D'OVERLAP VMC ---
@jax.jit
def compute_overlap_mc(vars_1, vars_2, samples_1, samples_2):
    # Log-amplitudes
    log_psi1_s1 = ma.apply(vars_1, samples_1)
    log_psi2_s1 = ma.apply(vars_2, samples_1)
    
    log_psi1_s2 = ma.apply(vars_1, samples_2)
    log_psi2_s2 = ma.apply(vars_2, samples_2)
    
    # Ratios
    ratio_12 = jnp.exp(log_psi2_s1 - log_psi1_s1)
    ratio_21 = jnp.exp(log_psi1_s2 - log_psi2_s2)
    
    O_12 = jnp.mean(ratio_12)
    O_21 = jnp.mean(ratio_21)
    
    return jnp.real(O_12 * O_21)

# --- EVALUATION VMC PURE ---
h0_list = np.linspace(0.5, 4.5, 40)
N_REP = 10 
sigma = meta["hamiltonian"]["sigma"]
rng = np.random.default_rng(42)

qfi_vmc_mean = []
qfi_vmc_std = []

for h0 in tqdm(h0_list, desc="Calcul QFI (VMC uniquement)"):
    qfi_reps = []
    
    for _ in range(N_REP):
        epsilon = rng.normal(loc=0.0, scale=sigma, size=n_spins)
        pars_base = jnp.array(h0 + epsilon)
        pars_shift = jnp.array(h0 + DELTA_H + epsilon)
        
        vs_base = vs.get_state(pars_base)
        vs_shift = vs.get_state(pars_shift)
        
        # Tirage Monte Carlo
        samples_base = vs_base.sample()
        samples_shift = vs_shift.sample()
        
        # Aplatir les chaînes
        s_base_flat = samples_base.reshape(-1, n_spins)
        s_shift_flat = samples_shift.reshape(-1, n_spins)
        
        F_vmc = compute_overlap_mc(vs_base.variables, vs_shift.variables, s_base_flat, s_shift_flat)
        
        qfi_reps.append(float((1 - F_vmc) / (DELTA_H**2)))
        
    qfi_vmc_mean.append(np.mean(qfi_reps))
    qfi_vmc_std.append(np.std(qfi_reps))

# --- PLOT ---
qfi_vmc_mean = np.array(qfi_vmc_mean)
qfi_vmc_std = np.array(qfi_vmc_std)

plt.figure(figsize=(8, 5))
plt.errorbar(h0_list, qfi_vmc_mean, yerr=qfi_vmc_std, fmt='-o', color='red', label="VMC FNQS", capsize=3)

plt.xlabel(r"Champ transverse moyen $h_0$")
plt.ylabel(r"Susceptibilité de fidélité moyennée $\overline{\chi_F}$")
plt.title(f"QFI par Monte Carlo ($\sigma={sigma}$) (Grille {L}x{L})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RUN_DIR / "QFI_disorder_MC_plot.pdf")
print(f"Graphique sauvegardé dans {RUN_DIR / 'QFI_disorder_MC_plot.pdf'}")