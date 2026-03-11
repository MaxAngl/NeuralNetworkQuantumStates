import os
import sys
# Ajouter le rÃ©pertoire racine du projet au chemin Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

#DÃ©commenter cette ligne pour L supÃ©rieur Ã  16 ou 20
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
# Gestionnaire de mÃ©moire plus efficace pour Ã©viter la fragmentation
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 
# Permet Ã  JAX de ne pas allouer 90% de la VRAM au dÃ©marrage
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import optax
from netket.utils import struct
import matplotlib.pyplot as plt

import netket as nk
import netket_foundational as nkf
from netket.sampler import rules
import netket_pro.distributed as nkpd
from flax import struct

from src.nqs_psc.utils import save_run
from netket_foundational._src.model.vit import ViTFNQS
from advanced_drivers._src.callbacks.base import AbstractCallback
from flip_rules import GlobalFlipRule
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# FIX NQXPACK: Sérialisation absolue (JAX Arrays + PT Sampler)
# ==========================================
try:
    from nqxpack._src.lib_v1 import register_serialization
    from netket.sampler import ParallelTemperingSampler
    
    def _serialize_jax_array(arr):
        return np.array(arr).tolist()
        
    try:
        register_serialization(jax.Array, _serialize_jax_array)
    except Exception:
        pass 

    def _to_builtin(x):
        if x is None:
            return None
        return np.array(x).tolist()

    def _serialize_pt_sampler(sampler):
        return {
            "n_replicas": _to_builtin(getattr(sampler, "n_replicas", 1)),
            "n_chains": _to_builtin(getattr(sampler, "n_chains", 0)),
            "machine_pow": _to_builtin(getattr(sampler, "machine_pow", 2.0)),
        }

    register_serialization(ParallelTemperingSampler, _serialize_pt_sampler)
    print("âœ… FIX NQXPACK: SÃ©rialisation (PT + JAX Arrays) enregistrÃ©e et sÃ©curisÃ©e !")
    
except ImportError:
    print("â„¹ï¸ nqxpack n'est pas dÃ©tectÃ©, on ignore la sÃ©rialisation custom.")
except Exception as e:
    print(f"âš ï¸ Erreur lors de l'enregistrement de la sÃ©rialisation : {e}")

# ==========================================
# 1. HYPERPARAMÃˆTRES ET CONFIGURATION
# ==========================================
seed = 1
rng = np.random.default_rng(seed)
k = jax.random.key(seed)

# --- PARAMAMETRES PHYSIQUES ---
L = 8                                      # Côté de la grille
n_spins = L**2                             # Nombre total de spins
b = 2                                      # Taille du patch
h0_train_list = [ 0.2, 0.6, 1.0, 1.5, 2, 2.8, 2.9, 3.0,3.2, 3.4, 3.6, 4.0, 5.0]
sigma_disorder = 0.1 
J_val = 1.0    
n_replicas = 10                             

# --- PARAMÃˆTRES MONTE CARLO & PT ---
total_configs_train = len(h0_train_list) * (n_replicas + 1)
chains_per_replica = 4      
samples_per_chain = 2       
n_chains = total_configs_train * chains_per_replica
n_samples = n_chains * samples_per_chain

# >>>> HYPERPARAMÃˆTRES PT <<<<
n_pt_temperatures = 8      
prob_global_flip = 0.01    

# --- PARAMÃˆTRES D'OPTIMISATION ---
n_iter = 400       
lr_init = 0.03
lr_end = 0.005
diag_shift = 2e-4
logs_path = "logs/2D_FNQS"

# --- CALCUL DU CHUNK_SIZE ---
TARGET_CHUNK = 64 

if n_samples <= TARGET_CHUNK:
    chunk_size = n_samples
else:
    chunk_size = 1
    for i in range(TARGET_CHUNK, 0, -1):
        if n_samples % i == 0:
            chunk_size = i
            break

print(f"ðŸ”¹ Configuration : {n_samples} samples total.")
print(f"ðŸ”¹ Chunk size auto-calculÃ© : {chunk_size} (Diviseur optimal <= {TARGET_CHUNK})")

# ParamÃ¨tres du modÃ¨le ViT
vit_params = {
    "num_layers": 2,
    "d_model": 16,
    "heads": 4,
    "b": b,
    "L_eff": (L // b)**2,
}

# ==========================================
# 2. DEFINITION DU SYSTEME
# ==========================================

hi = nk.hilbert.Spin(0.5, n_spins)
ps = nkf.ParameterSpace(N=n_spins, min=0, max=10*max(h0_train_list))

def generate_multi_h0_disorder(h0_list, n_reps, system_size, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    all_configs = []
    
    for h_m in h0_list:
        random_configs = rng.normal(loc=h_m, scale=sigma, size=(n_reps, system_size))
        homogeneous_config = np.full((1, system_size), h_m)
        batch_configs = np.vstack([random_configs, homogeneous_config])
        all_configs.append(batch_configs)
        
    return np.vstack(all_configs)

# ModÃ¨le
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
    two_dimensional=True, 
)        
 
# Sampler PT & Ã‰tat Variationnel
sa = nk.sampler.ParallelTemperingSampler(
    hi,
    rule=GlobalFlipRule(prob_global_flip),
    n_replicas=n_pt_temperatures,
    n_chains=n_chains
)

vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=total_configs_train, n_samples=n_samples, seed=seed, chunk_size=chunk_size)

# Initialisation des paramÃ¨tres (dÃ©sordre)
params_list = generate_multi_h0_disorder(h0_train_list, n_replicas, n_spins, sigma=sigma_disorder)
print(f"Forme des paramÃ¨tres de dÃ©sordre : {params_list.shape}")
vs.parameter_array = params_list

# OpÃ©rateurs
Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(n_spins)) * (1 / float(n_spins))

def create_operator(params):
    assert params.shape == (n_spins,)
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(n_spins))
    
    # Interactions 2D avec PBC
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i % L + 1) % L + (i // L) * L) for i in range(n_spins))
    ha_ZZ += sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + L) % n_spins) for i in range(n_spins))
    
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

gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift, linear_solver_fn=cg_solver)

log = nk.logging.JsonLog("log_data", save_params=False) 

meta = {
    "L": L,
    "nb_spins": n_spins,
    "graph": "Hypercube 2D",
    "n_dim": 2,
    "pbc": True,
    "hamiltonian": {
        "type": "Ising Disorder", 
        "J": J_val, 
        "h0_train_list": h0_train_list, 
        "sigma": sigma_disorder
    },
    "model": "ViTFNQS",
    "vit_config": vit_params,
    "sampler": {
        "type": "ParallelTemperingSampler", 
        "n_pt_temperatures": n_pt_temperatures,
        "n_chains_per_temp": n_chains, 
        "n_samples": n_samples,
        "rule": "GlobalFlipRule",
        "prob_global_flip": prob_global_flip
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

try:
    run_dir = save_run(log, meta, create_only=True, base_dir=logs_path)
except Exception as e:
    print(f"Warning: save_run issue ({e}), using default path.")
    run_dir = "checkpoints"

log = nk.logging.JsonLog(os.path.join(run_dir, "log_data"), save_params=False)

disorder_path = os.path.join(run_dir, "disorder_configs.npy")
np.save(disorder_path, params_list)
print(f"Configurations de dÃ©sordre sauvegardÃ©es dans : {disorder_path}")

start_time = time.time()

gs.run(
    n_iter,
    out=log,
    obs={"ham": ha_p, "mz": mz_p},
    callback=SaveState(run_dir, 10),
)

duration = time.time() - start_time
print(f"â±ï¸ Temps total d'entrainement : {duration:.2f} secondes")

meta["execution_time_seconds"] = duration
import json
with open(os.path.join(run_dir, "meta.json"), 'w') as f:
    json.dump(meta, f, indent=4)

# ==========================================
# 4. PLOTS ET ANALYSE FINALE
# ==========================================
print('Analyse et sauvegarde...')

conv_data = []
for i, pars in tqdm(enumerate(vs.parameter_array)):
    if hasattr(log.data["ham"], "__getitem__") and len(log.data["ham"]) > i:
        ham_log = log.data["ham"][i]
        conv_data.append({"iters": ham_log.iters, "e0": np.real(ham_log.Mean)})

plt.figure()
for _data in conv_data: 
    plt.plot(_data["iters"], _data["e0"], alpha=0.3)
plt.xlabel("Iterations")
plt.ylabel("Energy (Real)")
plt.savefig(os.path.join(run_dir, "convergence.pdf"))
plt.clf()

print("Calcul des V-scores finaux sur le Train...")
train_results = {"v_score": []}

for r in tqdm(range(total_configs_train)):
    pars = params_list[r]
    _vs = vs.get_state(pars)
    
    vs_mc = nk.vqs.MCState(
        sampler=nk.sampler.MetropolisLocal(hi, n_chains=16), 
        model=_vs.model, 
        variables=_vs.variables, 
        n_samples=1024, 
        chunk_size=64
    )
    
    _e = vs_mc.expect(create_operator(pars))
    val = _e.variance / (_e.Mean.real**2 + 1e-12)
    train_results["v_score"].append(val)

v_train = np.array(train_results["v_score"])

h_mean_train_full = []
for h_val in h0_train_list: 
    h_mean_train_full.extend([h_val] * (n_replicas + 1))

min_len = min(len(h_mean_train_full), len(v_train))

df_train = pd.DataFrame({
    "h_mean": h_mean_train_full[:min_len], 
    "v_score": v_train[:min_len]
})

output_csv = os.path.join(run_dir, "train_results.csv")
df_train.to_csv(output_csv, index=False)
print(f"âœ… Terminé ! Résultats sauvegardées dans : {output_csv}")