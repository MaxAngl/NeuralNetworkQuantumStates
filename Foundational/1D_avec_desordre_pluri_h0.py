import os
# Indispensable pour distribuer les réplicas efficacement
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import netket as nk
import netket_foundational as nkf
from nqs_psc.utils import save_run 

import time
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import optax
from scipy.stats import gaussian_kde
from netket.utils import struct
import matplotlib.pyplot as plt
from netket_foundational._src.model.vit import ViTFNQS
from advanced_drivers._src.callbacks.base import AbstractCallback
import netket_pro.distributed as nkpd

# ==========================================
# 1. HYPERPARAMÈTRES ET CONFIGURATION
# ==========================================
seed = 1
k = jax.random.key(seed)
L = 4              # Augmente ici pour tes tests (ex: 20, 25)
h0_train_list = [0, 0.4, 0.8, 0.9, 1.0, 1.1, 1.2, 2.0, 5.0]
sigma_disorder = 0.1 
J_val = 1.0/np.e    
n_replicas = 10    
total_configs_train = len(h0_train_list) * n_replicas
chains_per_replica = 4      
samples_per_chain = 64      
n_chains = total_configs_train * chains_per_replica 
n_samples = n_chains * samples_per_chain             
n_iter = 400       
lr_init = 0.03
lr_end = 0.005
diag_shift = 1e-4
logs_path = "logs"  

vit_params = {
    "num_layers": 4,
    "d_model": 32,
    "heads": 8,
    "b": 1,
    "L_eff": L,
}

# ==========================================
# 2. DEFINITION DU SYSTEME
# ==========================================
hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10*max(h0_train_list))

def generate_multi_h0_disorder(h0_list, n_reps, system_size, sigma, rng=None):
    if rng is None: rng = np.random.default_rng()
    all_configs = []
    for h_m in h0_list:
        configs = rng.normal(loc=h_m, scale=sigma, size=(n_reps, system_size))
        all_configs.append(configs)
    return np.vstack(all_configs)

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
    two_dimensional=False, 
)

sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=total_configs_train, n_samples=n_samples, seed=seed)

params_list = generate_multi_h0_disorder(h0_train_list, n_replicas, hi.size, sigma=sigma_disorder)
print(f"Forme des paramètres de désordre : {params_list.shape}")
vs.parameter_array = params_list

Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))

def create_operator(params):
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size) for i in range(hi.size))
    return -ha_X - J_val * ha_ZZ

ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
mz_p = nkf.operator.ParametrizedOperator(hi, ps, lambda _: Mz)

# ==========================================
# 3. LOGGING ET OPTIMISATION
# ==========================================
class SaveState(AbstractCallback, mutable=True):
    _path: str = struct.field(pytree_node=False, serialize=False)
    _prefix: str = struct.field(pytree_node=False, serialize=False)
    _save_every: int = struct.field(pytree_node=False, serialize=False)
    def __init__(self, path: str, save_every: int, prefix: str = "state"):
        self._path = path
        self._prefix = prefix
        self._save_every = save_every
    def on_run_start(self, step, driver, callbacks):
        if nkpd.is_master_process() and not os.path.exists(self._path):
            os.makedirs(self._path)
    def on_step_end(self, step, log_data, driver):
        if step % self._save_every == 0:
            path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
            driver.state.save(path)

learning_rate = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=300) 
optimizer = optax.sgd(learning_rate)
gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift)

log = nk.logging.JsonLog("log_data", save_params=False) 

meta = {
    "L": L,
    "graph": "Hypercube 1D",
    "n_dim": 1,
    "pbc": True,
    "hamiltonian": {"type": "Ising Disorder", "J": J_val, "h0_train_list": h0_train_list, "sigma": sigma_disorder},
    "model": "ViTFNQS",
    "vit_config": vit_params,
    "sampler": {"type": "MetropolisLocal", "n_chains": n_chains, "n_samples": n_samples},
    "optimizer": {"type": "SGD", "lr_init": lr_init, "lr_end": lr_end, "diag_shift": diag_shift},
    "n_iter": n_iter,
    "n_replicas_per_h0": n_replicas,
    "total_configs_train": total_configs_train,
}

try:
    run_dir = save_run(log, meta, create_only=True, base_dir=logs_path)
except Exception as e:
    run_dir = "checkpoints"

start_time = time.time()
gs.run(n_iter, out=log, obs={"ham": ha_p, "mz": mz_p}, callback=SaveState(run_dir, 10))
duration = time.time() - start_time
print(f"⏱️ Temps total d'entraînement : {duration:.2f} secondes")

meta["execution_time_seconds"] = duration
import json
with open(os.path.join(run_dir, "meta.json"), 'w') as f:
    json.dump(meta, f, indent=4)

# ==========================================
# 4. ANALYSE ET PLOTS (CONVERGENCE)
# ==========================================
print('Plotting convergence curves...')
conv_data = []
for i, pars in tqdm(enumerate(vs.parameter_array)):
    _ha = create_operator(pars)
    ed = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=False).item()
    if hasattr(log.data["ham"], "__getitem__") and len(log.data["ham"]) > i:
        ham_log = log.data["ham"][i]
        conv_data.append({"iters": ham_log.iters, "e0": ham_log.Mean, "err_val": ham_log.Mean - ed})

plt.figure()
for _data in conv_data:
    plt.plot(_data["iters"], np.abs(_data["err_val"] / _data["e0"]), alpha=0.3)
plt.xlabel("Iteration"); plt.ylabel("Rel Error"); plt.xscale("log"); plt.yscale("log")
plt.savefig(os.path.join(run_dir, "convergence.pdf")); plt.clf()

# ==========================================
# 5. TEST SUR NOUVEL ENSEMBLE (SANS FULLSUM)
# ==========================================
h0_test_list = [0.5, 0.85, 1.05, 1.3, 1.5, 3.0] 
N_test_per_h0 = 20
params_list_test = generate_multi_h0_disorder(h0_test_list, N_test_per_h0, hi.size, sigma_disorder)
N_test_total = params_list_test.shape[0]

vmc_vals = {"Energy": [], "Mz2": [], "V_score": []}
print(f'Computing NQS predictions on test set ({N_test_total} samples)...')

for i in tqdm(range(0, N_test_total, total_configs_train)):
    batch_params = params_list_test[i : i + total_configs_train]
    if len(batch_params) < total_configs_train: break
    vs.parameter_array = batch_params
    
    # On utilise directement vs.expect car c'est un FoundationalQuantumState
    # qui gère déjà l'échantillonnage pour chaque réplica en interne.
    e_obs = vs.expect(ha_p)
    mz2_obs = vs.expect(mz_p @ mz_p)
    
    for r in range(total_configs_train):
        # On extrait les moyennes et variances pour chaque réplica r
        mean_e = e_obs.Mean[r]
        var_e = e_obs.Variance[r]
        vmc_vals["Energy"].append(mean_e)
        vmc_vals["Mz2"].append(mz2_obs.Mean[r])
        vmc_vals["V_score"].append(var_e / (mean_e.real**2 + 1e-12))

print('Computing exact values on test set...')
exact_vals = {"Energy": [], "Mz2": []}
Mz2_op = Mz @ Mz
Mz2_mat = Mz2_op.to_sparse()
for pars in tqdm(params_list_test):
    _ha = create_operator(pars)
    E0, psi0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=True)
    exact_vals["Energy"].append(E0.item())
    exact_vals["Mz2"].append((psi0.T.conj() @ (Mz2_mat @ psi0.reshape(-1))).item().real)

vmc_final = {
    "Energy": np.array([np.real(e) for e in vmc_vals["Energy"]]),
    "Mz2": np.array([np.real(m) for m in vmc_vals["Mz2"]]),
    "V_score": np.array(vmc_vals["V_score"])
}
ex_mz2 = np.array(exact_vals["Mz2"])
err_test = np.abs(vmc_final['Mz2'] - ex_mz2) / (np.abs(ex_mz2) + 1e-12)

h_mean_test_full = []
for h_val in h0_test_list: h_mean_test_full.extend([h_val] * N_test_per_h0)

df_results = pd.DataFrame({
    "h_mean": h_mean_test_full,
    "exact_energy": exact_vals["Energy"],
    "vmc_energy": vmc_final["Energy"],
    "exact_mz2": exact_vals["Mz2"],
    "vmc_mz2": vmc_final["Mz2"],
    "v_score": vmc_final["V_score"]
})
df_results.to_csv(os.path.join(run_dir, "test_results.csv"), index=False)

# ==========================================
# 6. ANALYSE COMPARATIVE (SANS FULLSUM)
# ==========================================
print("Ré-évaluation des points de Train...")
vs.parameter_array = params_list  
e_train_obs = vs.expect(ha_p)
mz2_train_obs = vs.expect(mz_p @ mz_p)

train_results = {"V_score": [], "Mz2": [], "Ex_Mz2": []}
for r in range(total_configs_train):
    pars = params_list[r]
    E0, psi0 = nk.exact.lanczos_ed(create_operator(pars), k=1, compute_eigenvectors=True)
    ex_mz2_val = (psi0.reshape(-1).T.conj() @ (Mz2_mat @ psi0.reshape(-1))).item().real
    
    mean_e = e_train_obs.Mean[r]
    train_results["V_score"].append(e_train_obs.Variance[r] / (mean_e.real**2 + 1e-12))
    train_results["Mz2"].append(mz2_train_obs.Mean[r].real)
    train_results["Ex_Mz2"].append(ex_mz2_val)

v_train = np.array(train_results["V_score"])
err_train = np.abs(np.array(train_results["Mz2"]) - np.array(train_results["Ex_Mz2"])) / (np.abs(np.array(train_results["Ex_Mz2"])) + 1e-12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
all_v = np.concatenate([v_train, v_test])
bins_v = np.logspace(np.log10(all_v.min() + 1e-18), np.log10(all_v.max() + 1e-2), 25)
ax1.hist(vmc_final['V_score'], bins=bins_v, alpha=0.5, label='Test', color='orange', edgecolor='darkorange')
ax1.hist(v_train, bins=bins_v, alpha=0.8, label='Train', color='red', edgecolor='black')
ax1.set_xscale('log'); ax1.set_title("Distribution du V-score"); ax1.legend()

all_err = np.concatenate([err_train, err_test])
bins_e = np.logspace(np.log10(all_err.min() + 1e-18), np.log10(all_err.max() + 1e-1), 25)
ax2.hist(err_test, bins=bins_e, alpha=0.5, label='Test', color='blue', edgecolor='darkblue')
ax2.hist(err_train, bins=bins_e, alpha=0.8, label='Train', color='cyan', edgecolor='black')
ax2.set_xscale('log'); ax2.set_title("Distribution de l'Erreur Relative $M_z^2$"); ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "comparative_analysis.pdf"))
print(f"✅ Analyse terminée dans : {run_dir}")