import os
#Décommenter cette ligne pour L supérieur à 16 ou 20
#os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import netket as nk
import netket_foundational as nkf
from nqs_psc.utils import save_run # Assure-toi que ce module est accessible

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
# On définit tout ici pour que le 'meta' soit cohérent
seed = 1
k = jax.random.key(seed)
L = 4              # Taille du système
h0_train_list = [0, 0.4, 0.8, 0.9, 1.0, 1.1, 1.2, 2.0 , 5.0]          # Champ moyen
sigma_disorder = 0.1 # Désordre
J_val = 1.0/np.e    # Couplage Ising (défini dans create_operator)
n_replicas = 10    # Nombre de réalisations de désordre
total_configs_train = len(h0_train_list) * n_replicas
chains_per_replica = 4      # <--- ICI : Chaque réplica aura 4 chaînes indépendantes
samples_per_chain = 32      # Nombre de points récoltés par chaque chaîne
n_chains = total_configs_train * chains_per_replica 
n_samples = n_chains * samples_per_chain             
n_iter = 500       # Nombre d'étapes d'optimisation
lr_init = 0.03
lr_end = 0.005
diag_shift = 1e-4
logs_path = "logs"  # Dossier racine pour les logs

# Paramètres du modèle ViT
vit_params = {
    "num_layers": 1,
    "d_model": 16,
    "heads": 2,
    "b": 1,
    "L_eff": L,
}

# ==========================================
# 2. DEFINITION DU SYSTEME
# ==========================================

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10*max(h0_train_list))

def generate_multi_h0_disorder(h0_list, n_reps, system_size, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    all_configs = []
    for h_m in h0_list:
        configs = rng.normal(loc=h_m, scale=sigma, size=(n_reps, system_size))
        all_configs.append(configs)
    return np.vstack(all_configs) # Shape: (len(h0_list)*n_reps, system_size)

# Modèle
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

# Sampler & État Variationnel
sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=total_configs_train, n_samples=n_samples, seed=seed)

# Initialisation des paramètres (désordre)
params_list = generate_multi_h0_disorder(h0_train_list, n_replicas, hi.size, sigma=sigma_disorder)
print(f"Forme des paramètres de désordre : {params_list.shape}")
vs.parameter_array = params_list

# Opérateurs
Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))

def create_operator(params):
    assert params.shape == (hi.size,)
    # Transverse field term: sum_i h_i sigma^x_i
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    # Ising interaction
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size) for i in range(hi.size))
    return -ha_X - J_val * ha_ZZ

ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
mz_p = nkf.operator.ParametrizedOperator(hi, ps, lambda _: Mz)

# ==========================================
# 3. LOGGING ET OPTIMISATION
# ==========================================

# Callback de sauvegarde
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
        # Sauvegarde initiale optionnelle
        # path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
        # driver.state.save(path)

    def on_step_end(self, step, log_data, driver):
        if step % self._save_every == 0:
            path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
            driver.state.save(path)

# Optimiseur
learning_rate = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=300) 
optimizer = optax.sgd(learning_rate)
gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift)

# Logger
# On initialise le logger avec un dossier temporaire ou final
log = nk.logging.JsonLog("log_data", save_params=False) 

# Dictionnaire META propre
meta = {
    "L": L,
    "graph": "Hypercube 1D",
    "n_dim": 1,
    "pbc": True,
    "hamiltonian": {
        "type": "Ising Disorder", 
        "J": J_val, 
        "h0_train_list": h0_train_list, # On enregistre la liste complète
        "sigma": sigma_disorder
    },
    "model": "ViTFNQS",
    "vit_config": vit_params,
    "sampler": {
        "type": "MetropolisLocal", 
        "n_chains": n_chains, 
        "n_samples": n_samples
    },
    "optimizer": {
        "type": "SGD", 
        "lr_init": lr_init, 
        "lr_end": lr_end, 
        "diag_shift": diag_shift
    },
    "n_iter": n_iter,
    "n_replicas_per_h0": n_replicas,
    "total_configs_train": total_configs_train,
    "seed": seed,
}

# Initialisation du logger et création du dossier
log = nk.logging.JsonLog("log_data", save_params=False)

# Création de la structure de dossier via ta fonction utilitaire
# Note: Assure-toi que save_run renvoie bien le chemin créé si tu veux l'utiliser pour SaveState
try:
    run_dir = save_run(log, meta, create_only=True, base_dir=logs_path)
except Exception as e:
    print(f"Warning: save_run issue ({e}), using default path.")
    run_dir = "checkpoints"

start_time = time.time()

# Lancement du run
gs.run(
    n_iter,
    out=log,
    obs={"ham": ha_p, "mz": mz_p},
    callback=SaveState(run_dir, 10), # Sauvegarde dans le dossier créé
)

duration = time.time() - start_time
print(f"⏱️ Temps total d'entraînement : {duration:.2f} secondes")

# Mise à jour du meta.json avec le temps d'exécution final
meta["execution_time_seconds"] = duration
import json
with open(os.path.join(run_dir, "meta.json"), 'w') as f:
    json.dump(meta, f, indent=4)

# ==========================================
# 4. ANALYSE ET PLOTS (CONVERGENCE)
# ==========================================
print('Plotting convergence curves...')
conv_data = []

# Attention: log.data["ham"] est une liste de loggers (un par replica) si nkf gère le logging shardé ainsi.
# Sinon, l'accès peut varier selon la version de nkf/nk.
# On suppose ici que log.data["ham"] est accessible par index [i].

for i, pars in tqdm(enumerate(vs.parameter_array)):
    _ha = create_operator(pars)
    # Calcul exact (Lanczos)
    ed = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=False).item()

    # Récupération des données du log
    # Note: Il faut s'assurer que log.data["ham"] contient bien une liste correspondant aux replicas
    # Si nkf agrège tout, cette boucle doit être adaptée. 
    # Supposons que log.data["ham"][i] existe :
    
    if hasattr(log.data["ham"], "__getitem__") and len(log.data["ham"]) > i:
        ham_log = log.data["ham"][i]
        err_val = ham_log.Mean - ed
        conv_data.append({
            "iters": ham_log.iters,
            "e0": ham_log.Mean,
            "err_val": err_val
        })

plt.figure()
for _data in conv_data:
    plt.plot(
        _data["iters"],
        np.abs(_data["err_val"] / _data["e0"]),
        alpha=0.3
    )

plt.xlabel("Iteration")
plt.ylabel("Rel Error")
plt.xscale("log")
plt.yscale("log")
plt.savefig(os.path.join(run_dir, "convergence.pdf"))
plt.clf()

# ==========================================
# 5. TEST SUR NOUVEL ENSEMBLE (CORRIGÉ)
# ==========================================

h0_test_list = [0.5, 0.85, 1.05, 1.3, 1.5, 3.0] # Valeurs d'interpolation et d'extrapolation
N_test_per_h0 = 20
params_list_test = generate_multi_h0_disorder(h0_test_list, N_test_per_h0, hi.size, sigma_disorder)
N_test_total = params_list_test.shape[0]

vmc_vals = {"Energy": [], "Mz2": [], "V_score": []}

print(f'Computing NQS predictions on test set ({N_test_total} samples)...')

# On avance par paquets de n_replicas (ici 10)
for i in tqdm(range(0, N_test_total, total_configs_train)):
    # 1. On prend un lot de 10 paramètres
    batch_params = params_list_test[i : i + total_configs_train]
    
    # Si le dernier lot est plus petit que n_replicas, on le complète avec des zéros (ou on ignore)
    if len(batch_params) < total_configs_train: break

    # 2. On injecte le lot dans le vstate
    vs.parameter_array = batch_params
    
    # 3. On évalue chaque réplica du lot individuellement
    for r in range(total_configs_train):
        pars = batch_params[r]
        _vs = vs.get_state(pars) # Récupère l'état spécifique au paramètre r
        
        # Passage en FullSum pour la précision (puisque L=4)
        vs_fs = nk.vqs.FullSumState(hilbert=hi, model=_vs.model, variables=_vs.variables)
        
        _ha = create_operator(pars)
        _e = vs_fs.expect(_ha)
        _o = vs_fs.expect(Mz @ Mz)
        
        # Calcul du V-score
        v_score = _e.variance / (_e.Mean.real**2 + 1e-12)
        
        vmc_vals["Energy"].append(_e.Mean)
        vmc_vals["Mz2"].append(_o.Mean)
        vmc_vals["V_score"].append(v_score)

# --- Calcul Exact de Test (Indispensable pour comparer) ---
exact_vals = {"Energy": [], "Mz2": []}
print('Computing exact values on test set...')
Mz2_op = Mz @ Mz
Mz2_mat = Mz2_op.to_sparse()

for pars in tqdm(params_list_test):
    _ha = create_operator(pars)
    E0, psi0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=True)
    exact_vals["Energy"].append(E0.item())
    exact_vals["Mz2"].append((psi0.T.conj() @ (Mz2_mat @ psi0.reshape(-1))).item().real)

# --- Conversion des résultats VMC en numpy arrays ---
vmc_final = {
    "Energy": np.array([np.real(e) for e in vmc_vals["Energy"]]),
    "Mz2": np.array([np.real(m) for m in vmc_vals["Mz2"]]),
    "V_score": np.array(vmc_vals["V_score"])
}

# --- Conversion des résultats EXACTS en numpy arrays (INDISPENSABLE) ---
ex_energy = np.array(exact_vals["Energy"])
ex_mz2 = np.array(exact_vals["Mz2"])

# Maintenant le calcul d'erreur ne plantera plus
err_test = np.abs(vmc_final['Mz2'] - ex_mz2) / (np.abs(ex_mz2) + 1e-12)

# ==========================================
# 5bis. SAUVEGARDE DES RESULTATS (CSV)
# ==========================================
# On crée une liste des h_mean correspondant à chaque point de test
h_mean_test_full = []
for h_val in h0_test_list:
    h_mean_test_full.extend([h_val] * N_test_per_h0)

df_results = pd.DataFrame({
    "h_mean": h_mean_test_full, # Utilise la liste étendue
    "exact_energy": exact_vals["Energy"],
    "vmc_energy": vmc_final["Energy"],
    "exact_mz2": exact_vals["Mz2"],
    "vmc_mz2": vmc_final["Mz2"],
    "v_score": vmc_final["V_score"]
})

csv_path = os.path.join(run_dir, "test_results.csv")
df_results.to_csv(csv_path, index=False)
print(f"✅ Données de test sauvegardées dans : {csv_path}")

# ==========================================
# 6. ANALYSE COMPARATIVE : TRAIN VS TEST
# ==========================================

print("Ré-évaluation des points de Train pour comparaison...")
vs.parameter_array = params_list  # On remet les 10 points originaux

train_results = {"V_score": [], "Mz2": [], "Ex_Mz2": []}

for r in range(total_configs_train):
    pars = params_list[r]
    _vs = vs.get_state(pars)
    vs_fs = nk.vqs.FullSumState(hilbert=hi, model=_vs.model, variables=_vs.variables)
    
    # VMC
    _e = vs_fs.expect(create_operator(pars))
    _o = vs_fs.expect(Mz @ Mz)
    
    # Exact
    E0, psi0 = nk.exact.lanczos_ed(create_operator(pars), k=1, compute_eigenvectors=True)
    ex_mz2_val = (psi0.reshape(-1).T.conj() @ (Mz2_mat @ psi0.reshape(-1))).item().real
    
    train_results["V_score"].append(_e.variance / (_e.Mean.real**2 + 1e-12))
    train_results["Mz2"].append(_o.Mean.real)
    train_results["Ex_Mz2"].append(ex_mz2_val)

# --- CONVERSION EN ARRAYS POUR CALCULS ---
v_train = np.array(train_results["V_score"])
err_train = np.abs(np.array(train_results["Mz2"]) - np.array(train_results["Ex_Mz2"])) / (np.abs(np.array(train_results["Ex_Mz2"])) + 1e-12)

v_test = vmc_final['V_score']
# err_test a déjà été calculé plus haut avec ex_mz2 converti

# --- TRACÉ DES HISTOGRAMMES ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Histogramme V-scores
all_v = np.concatenate([v_train, v_test])
bins_v = np.logspace(np.log10(all_v.min() + 1e-18), np.log10(all_v.max() + 1e-2), 25)
ax1.hist(v_test, bins=bins_v, alpha=0.5, label='Test', color='orange', edgecolor='darkorange')
ax1.hist(v_train, bins=bins_v, alpha=0.8, label='Train', color='red', edgecolor='black')
ax1.set_xscale('log')
ax1.set_title("Distribution du V-score")
ax1.legend()

# Histogramme Erreurs Relatives
all_err = np.concatenate([err_train, err_test])
bins_e = np.logspace(np.log10(all_err.min() + 1e-18), np.log10(all_err.max() + 1e-1), 25)
ax2.hist(err_test, bins=bins_e, alpha=0.5, label='Test', color='blue', edgecolor='darkblue')
ax2.hist(err_train, bins=bins_e, alpha=0.8, label='Train', color='cyan', edgecolor='black')
ax2.set_xscale('log')
ax2.set_title("Distribution de l'Erreur Relative $M_z^2$")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "comparative_analysis.pdf"))