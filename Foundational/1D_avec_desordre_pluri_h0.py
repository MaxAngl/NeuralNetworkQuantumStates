import os
import sys
# Ajouter le répertoire racine du projet au chemin Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

#Décommenter cette ligne pour L supérieur à 16 ou 20
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
# Gestionnaire de mémoire plus efficace pour éviter la fragmentation
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 
# Permet à JAX de ne pas allouer 90% de la VRAM au démarrage
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import netket as nk
import netket_foundational as nkf

from src.nqs_psc.utils import save_run # Assure-toi que ce module est accessible

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
rng = np.random.default_rng(seed)
k = jax.random.key(seed)

# --- PARAMÈTRES PHYSIQUES ---
L = 16                                      # Taille du système
h0_train_list = [ 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 2, 3.5, 5.0 ]
sigma_disorder = 0.1 
J_val = 1.0/np.e    
n_replicas = 10                             # Nombre de réalisations de désordre

# --- PARAMÈTRES MONTE CARLO ---
total_configs_train = len(h0_train_list) * (n_replicas + 1)
chains_per_replica = 4      
samples_per_chain = 2       
n_chains = total_configs_train * chains_per_replica 
n_samples = n_chains * samples_per_chain    

# --- CALCUL AUTOMATIQUE ET SYSTÉMATIQUE DU CHUNK_SIZE ---
# Cible : On veut traiter environ 512 à 1024 samples à la fois pour éviter le OOM
# tout en gardant une bonne vectorisation.
TARGET_CHUNK = 64 

if n_samples <= TARGET_CHUNK:
    chunk_size = n_samples
else:
    # On cherche le plus grand diviseur de n_samples qui est <= TARGET_CHUNK
    chunk_size = 1
    for i in range(TARGET_CHUNK, 0, -1):
        if n_samples % i == 0:
            chunk_size = i
            break

print(f"🔹 Configuration : {n_samples} samples total.")
print(f"🔹 Chunk size auto-calculé : {chunk_size} (Diviseur optimal <= {TARGET_CHUNK})")

# --- PARAMÈTRES D'OPTIMISATION ---
n_iter = 500       
lr_init = 0.03
lr_end = 0.005
diag_shift = 2e-4
logs_path = "logs"


# Paramètres du modèle ViT
vit_params = {
    "num_layers": 2,
    "d_model": 16,
    "heads": 4,
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
        # 1. Génération des 'n_reps' configurations désordonnées (Aléatoire)
        random_configs = rng.normal(loc=h_m, scale=sigma, size=(n_reps, system_size))
        
        # 2. Création de la configuration homogène (h_m, h_m, ..., h_m) (Exacte)
        # Shape (1, system_size)
        homogeneous_config = np.full((1, system_size), h_m)
        
        # 3. On empile les deux : on obtient (n_reps + 1) configurations pour ce h_m
        # L'homogène est ajoutée à la fin du bloc de ce h_m
        batch_configs = np.vstack([random_configs, homogeneous_config])
        
        all_configs.append(batch_configs)
        
    return np.vstack(all_configs)

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
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=total_configs_train, n_samples=n_samples, seed=seed, chunk_size=chunk_size)

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

    def on_step_end(self, step, log_data, driver):
        if step % self._save_every == 0:
            path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
            driver.state.save(path)

# Optimiseur
learning_rate = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=300) 
optimizer = optax.sgd(learning_rate)
solver = nk.optimizer.solver.cholesky
gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift, solver=solver)

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


# Création de la structure de dossier via ta fonction utilitaire
try:
    run_dir = save_run(log, meta, create_only=True, base_dir=logs_path)
except Exception as e:
    print(f"Warning: save_run issue ({e}), using default path.")
    run_dir = "checkpoints"

# Initialisation du logger et création du dossier
log = nk.logging.JsonLog(os.path.join(run_dir, "log_data.json"), save_params=False)

# AJOUT : SAUVEGARDE DES CONFIGURATIONS DE DÉSORDRE
disorder_path = os.path.join(run_dir, "disorder_configs.npy")
np.save(disorder_path, params_list)
print(f"Configurations de désordre sauvegardées dans : {disorder_path}")

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

# Sans calcul exact, on plotte juste l'énergie VMC
for i, pars in tqdm(enumerate(vs.parameter_array)):
    # Récupération des données du log
    if hasattr(log.data["ham"], "__getitem__") and len(log.data["ham"]) > i:
        ham_log = log.data["ham"][i]
        
        # On ne calcule plus l'erreur relative car pas d'ed
        conv_data.append({
            "iters": ham_log.iters,
            "e0": ham_log.Mean,
        })

plt.figure()
for _data in conv_data:
    plt.plot(
        _data["iters"],
        _data["e0"], # On affiche l'énergie brute au lieu de l'erreur relative
        alpha=0.3
    )

plt.xlabel("Iteration")
plt.ylabel("Energy (VMC)")
# plt.xscale("log") # Pas forcément pertinent pour l'énergie brute
# plt.yscale("log") 
plt.savefig(os.path.join(run_dir, f"Found_disordered_pluri_h0_L={L}_convergence.pdf"))
plt.clf()


# --- SAUVEGARDE DES DONNÉES DE TRAIN ---
# On récupère les h_0 correspondant à chaque réplica de train
h_mean_train_full = []
for h_val in h0_train_list:
    # Rappel : on a (n_replicas + 1) configs par h_val dans le train (random + homogene)
    # total_configs_train est calculé plus haut sur cette base, mais ici il faut être précis
    configs_per_h_train = n_replicas + 1 
    h_mean_train_full.extend([h_val] * configs_per_h_train)

df_train = pd.DataFrame({
    "h_mean": h_mean_train_full,
    "v_score": v_train,
})
df_train.to_csv(os.path.join(run_dir, "train_results.csv"), index=False)
print(f"✅ Données de train sauvegardées dans : {run_dir}/train_results.csv")