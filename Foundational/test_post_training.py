import os
import json
import netket as nk
import netket_foundational as nkf
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from netket_foundational._src.model.vit import ViTFNQS

# --- 1. CHARGEMENT ET RÉCUPÉRATION DU CONTEXTE ---
run_dir = Path("/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/logs/run_2026-01-31_22-19-41")
checkpoint_path = run_dir / "state_810.nk"
meta_path = run_dir / "meta.json"

with open(meta_path, 'r') as f:
    meta = json.load(f)

L = meta["L"]
J_val = meta["hamiltonian"]["J"]
sigma_disorder = meta["hamiltonian"]["sigma"]
h0_train_list = meta["hamiltonian"]["h0_train_list"]
n_replicas_train = meta["n_replicas_per_h0"]
seed = meta["seed"]

# --- 2. RECONSTRUCTION DE L'ÉTAT ---
hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10*max(h0_train_list))

ma = ViTFNQS(
    num_layers=meta["vit_config"]["num_layers"],
    d_model=meta["vit_config"]["d_model"],
    heads=meta["vit_config"]["heads"],
    b=meta["vit_config"]["b"],
    L_eff=meta["vit_config"]["L_eff"],
    n_coups=ps.size,
    complex=True, disorder=True
)

sa = nk.sampler.MetropolisLocal(hi, n_chains=meta["sampler"]["n_chains"])
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=meta["total_configs_train"], n_samples=meta["sampler"]["n_samples"])
vs.load(str(checkpoint_path))
print(f"✅ État chargé : {checkpoint_path}")

# --- 3. OPÉRATEURS ET FONCTIONS ---
Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))
Mz2_op = Mz @ Mz
Mz2_sparse = Mz2_op.to_sparse()

def create_operator(params):
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size) for i in range(hi.size))
    return -ha_X - J_val * ha_ZZ

def generate_disorder(h0_list, n_reps, size, sigma, seed_val):
    rng = np.random.default_rng(seed_val)
    all_c = []
    for h in h0_list:
        all_c.append(rng.normal(h, sigma, (n_reps, size)))
    return np.vstack(all_c)

# --- 4. PRÉPARATION DES POINTS (TRAIN ET TEST) ---
params_train = generate_disorder(h0_train_list, n_replicas_train, L, sigma_disorder, seed)

h0_test_list = [0.5, 0.85, 1.05, 1.3, 1.5, 3.0]
N_test_per_h0 = 20
params_test = generate_disorder(h0_test_list, N_test_per_h0, L, sigma_disorder, seed + 100)

# --- 5. ÉVALUATION VMC (Logique calquée sur ton script exemple) ---
def evaluate_foundational(p_list, desc):
    res = {"Energy": [], "Mz2": [], "V_score": []}
    for pars in tqdm(p_list, desc=desc):
        _ha = create_operator(pars)
        # On récupère l'état spécifique pour ces paramètres
        _vs = vs.get_state(pars) 
        _vs.sample()
        
        e_stats = _vs.expect(_ha)
        m_stats = _vs.expect(Mz2_op)
        
        res["Energy"].append(e_stats.Mean.real.item())
        res["Mz2"].append(m_stats.Mean.real.item())
        res["V_score"].append(e_stats.Variance / (e_stats.Mean.real**2 + 1e-12))
    return res

vmc_train = evaluate_foundational(params_train, "Eval Train")
vmc_test = evaluate_foundational(params_test, "Eval Test")

# --- 6. CALCULS EXACTS ---
def evaluate_exact(p_list, desc):
    res = {"Energy": [], "Mz2": []}
    for pars in tqdm(p_list, desc=desc):
        _ha = create_operator(pars)
        E0, psi0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=True)
        res["Energy"].append(E0.item())
        res["Mz2"].append((psi0.T.conj() @ (Mz2_sparse @ psi0.reshape(-1))).item().real)
    return res

exact_train = evaluate_exact(params_train, "Exact Train")
exact_test = evaluate_exact(params_test, "Exact Test")

# --- 7. CALCUL DES ERREURS ET TRACÉ ---
def get_err(vmc, exact):
    return np.abs((np.array(vmc) - np.array(exact)) / np.array(exact))

err_e_train = get_err(vmc_train["Energy"], exact_train["Energy"])
err_e_test  = get_err(vmc_test["Energy"], exact_test["Energy"])
err_m_train = get_err(vmc_train["Mz2"], exact_train["Mz2"])
err_m_test  = get_err(vmc_test["Mz2"], exact_test["Mz2"])

# Histogrammes (car on a du désordre, pas une ligne simple)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# V-Score
ax[0].hist(vmc_test["V_score"], bins=np.logspace(-6, 0, 25), alpha=0.5, label='Test', color='orange')
ax[0].hist(vmc_train["V_score"], bins=np.logspace(-6, 0, 25), alpha=0.8, label='Train', color='red')
ax[0].set_xscale('log')
ax[0].set_title("Distribution du V-score")

# Erreur Mz2
ax[1].hist(err_m_test, bins=np.logspace(-5, 0, 25), alpha=0.5, label='Test', color='blue')
ax[1].hist(err_m_train, bins=np.logspace(-5, 0, 25), alpha=0.8, label='Train', color='cyan')
ax[1].set_xscale('log')
ax[1].set_title("Erreur Relative $M_z^2$")

for a in ax: a.legend(); a.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(run_dir / "recovered_analysis_fixed.pdf")
print(f"✅ Analyse terminée. Graphique sauvegardé dans {run_dir}")