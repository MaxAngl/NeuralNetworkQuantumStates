import os
import sys
# Ajouter le répertoire racine du projet au chemin Python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, foundational_dir)
sys.path.insert(0, project_root)

# Décommenter cette ligne pour L supérieur à 16 ou 20
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
# Gestionnaire de mémoire plus efficace pour éviter la fragmentation
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 
# Permet à JAX de ne pas allouer 90% de la VRAM au démarrage
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import netket as nk
import netket_foundational as nkf

from src.nqs_psc.utils import save_run 

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
from netket.sampler import rules
from flax import struct
from flip_rules import GlobalFlipRule

# ==========================================
# 1. HYPERPARAMÈTRES ET CONFIGURATION
# ==========================================
seed = 1
rng = np.random.default_rng(seed)
k = jax.random.key(seed)

# --- MODIFICATION 2D : Définition de la grille ---
L_side = 5  # Taille du côté de la grille (5x5 = 25 spins)
if len(sys.argv) > 1:
    L_side = int(sys.argv[1])
L_total = L_side**2  # --- MODIFICATION 2D : Nombre total de spins ---

h0_train_list = [ 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 2, 3.5, 5.0 ]
sigma_disorder = 0.1 
J_val = 1.0    
n_replicas = 10 

# --- PARAMÈTRES MONTE CARLO ---
total_configs_train = len(h0_train_list) * (n_replicas + 1)
chains_per_replica = 4      
samples_per_chain = 2       
n_chains = total_configs_train * chains_per_replica
n_samples = n_chains * samples_per_chain
prob_global_flip = 0.03  

# --- PARAMÈTRES D'OPTIMISATION ---
n_iter = 300       
lr_init = 0.03
lr_end = 0.005
diag_shift = 2e-4
logs_path = os.path.join(foundational_dir, "logs")

# --- CALCUL AUTOMATIQUE DU CHUNK_SIZE ---
TARGET_CHUNK = 16 
if n_samples <= TARGET_CHUNK:
    chunk_size = n_samples
else:
    chunk_size = 1
    for i in range(TARGET_CHUNK, 0, -1):
        if n_samples % i == 0:
            chunk_size = i
            break

chunk_size_bwd=4

print(f"🔹 Configuration 2D : {L_side}x{L_side} = {L_total} spins.")
print(f"🔹 Chunk size auto-calculé : {chunk_size}")

# --- MODIFICATION 2D : Paramètres du modèle ViT ---
vit_params = {
    "num_layers": 2,
    "d_model": 16,
    "heads": 4,
    "b": 1,
    "L_eff": L_side, # --- MODIFICATION 2D : L_eff devient la dimension linéaire ---
}

# ==========================================
# 2. DEFINITION DU SYSTEME
# ==========================================

# --- MODIFICATION 2D : Création d'un graphe Grid ---
graph = nk.graph.Grid(shape=[L_side, L_side], pbc=True) 
hi = nk.hilbert.Spin(0.5, graph.n_nodes)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10*max(h0_train_list))

def generate_multi_h0_disorder(h0_list, n_reps, system_size, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    all_configs = []
    for h_m in h0_list:
        raw_configs = rng.normal(loc=h_m, scale=sigma, size=(n_reps, system_size))
        random_configs = np.abs(raw_configs)
        homogeneous_config = np.full((1, system_size), h_m)
        batch_configs = np.vstack([random_configs, homogeneous_config])
        all_configs.append(batch_configs)
    return np.vstack(all_configs)

# --- MODIFICATION 2D : Activation du flag two_dimensional ---
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
    two_dimensional=True, # --- MODIFICATION 2D ---
)        

sa = nk.sampler.MetropolisSampler(
    hi,
    rule=GlobalFlipRule(prob_global_flip),
    n_chains=n_chains
)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=total_configs_train, n_samples=n_samples, seed=seed, chunk_size=chunk_size)

# Initialisation des spins
sigma_orig = vs.sampler_state.σ
flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
half = flat_sigma.shape[0] // 2

# --- MODIFICATION 2D : Utilisation de L_total au lieu de L ---
flat_sigma = flat_sigma.at[:half, :L_total].set(1)
flat_sigma = flat_sigma.at[half:, :L_total].set(-1)

sigma_new = flat_sigma.reshape(sigma_orig.shape)
vs.sampler_state = vs.sampler_state.replace(σ=sigma_new)

params_list = generate_multi_h0_disorder(h0_train_list, n_replicas, hi.size, sigma=sigma_disorder)
vs.parameter_array = params_list

# --- MODIFICATION 2D : Opérateurs sur la grille ---
Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))

def create_operator(params):
    assert params.shape == (hi.size,)
    # Champ transverse
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    # --- MODIFICATION 2D : Somme sur les arêtes (edges) du graphe Grid ---
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, j) for i, j in graph.edges())
    return -ha_X - J_val * ha_ZZ

ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
mz_p = nkf.operator.ParametrizedOperator(hi, ps, lambda _: Mz)

# ==========================================
# 3. LOGGING ET OPTIMISATION
# ==========================================

class SaveState(AbstractCallback):
    _path: str = struct.field(pytree_node=False)
    _prefix: str = struct.field(pytree_node=False)
    _save_every: int = struct.field(pytree_node=False)

    def __init__(self, path: str, save_every: int, prefix: str = "state"):
        self._path = path
        self._prefix = prefix
        self._save_every = save_every
        if nkpd.is_master_process():
            os.makedirs(self._path, exist_ok=True)

    def on_step_end(self, step, log_data, driver):
        if step % self._save_every == 0:
            if nkpd.is_master_process() and not os.path.exists(self._path):
                os.makedirs(self._path, exist_ok=True)
            path = os.path.join(self._path, f"{self._prefix}_{driver.step_count}.nk")
            driver.state.save(path)

learning_rate = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=300)
optimizer = optax.sgd(learning_rate)
def cg_solver(A, b):
    return jax.scipy.sparse.linalg.cg(A, b, tol=1e-4)[0]

gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift, linear_solver_fn=cg_solver, chunk_size_bwd=chunk_size_bwd, use_ntk=True)

log = nk.logging.JsonLog("log_data", save_params=False) 

# --- MODIFICATION 2D : Mise à jour des Meta-données ---
meta = {
    "L_side": L_side,
    "L_total": L_total,
    "graph": "Grid 2D", # --- MODIFICATION 2D ---
    "n_dim": 2,          # --- MODIFICATION 2D ---
    "pbc": True,
    "hamiltonian": {
        "type": "Ising 2D Disorder", 
        "J": J_val, 
        "h0_train_list": h0_train_list,
        "sigma": sigma_disorder
    },
    "model": "ViT