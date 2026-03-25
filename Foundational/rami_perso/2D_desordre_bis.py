# --- VERSION ULTRA-ROBUSTE DES CHEMINS ---
import os
import sys

# 1. Chemin absolu du script actuel
script_path = os.path.abspath(__file__)
foundational_dir = os.path.dirname(os.path.dirname(script_path))
project_root = os.path.dirname(foundational_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, foundational_dir)

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 
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
    
except ImportError:
    pass
except Exception:
    pass

# ==========================================
# 1. HYPERPARAMÈTRES ET CONFIGURATION
# ==========================================
seed = 1
rng = np.random.default_rng(seed)
k = jax.random.key(seed)

# --- PARAMÈTRES PHYSIQUES ---
L = 7
n_spins = L**2
b = 1
h0_train_list = [ 0.2, 0.6, 1.0, 1.5, 2, 2.8, 2.9, 3.0, 3.2, 3.4, 3.6, 4.0, 5.0 ]
sigma_disorder = 0.1 
J_val = 1.0    
n_replicas = 10                             

# --- PARAMÈTRES MONTE CARLO ---
total_configs_train = len(h0_train_list) * (n_replicas + 1)
chains_per_replica = 4      
samples_per_chain = 2       
n_chains = total_configs_train * chains_per_replica
n_samples = n_chains * samples_per_chain
prob_global_flip = 0.2    

# --- PARAMÈTRES D'OPTIMISATION ---
n_iter = 400       
lr_init = 0.03
lr_end = 0.005
diag_shift = 2e-4
logs_path = os.path.join(foundational_dir, "logs")

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

chunk_size_bwd = 4

# Paramètres du modèle ViT
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
        raw_configs = rng.normal(loc=h_m, scale=sigma, size=(n_reps, system_size))
        random_configs = np.abs(raw_configs)
        homogeneous_config = np.full((1, system_size), h_m)
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
    two_dimensional=True, 
)        
 
# Sampler Metropolis & État Variationnel (adapté du premier script)
sa = nk.sampler.MetropolisSampler(
    hi,
    rule=GlobalFlipRule(prob_global_flip),
    n_chains=n_chains
)

vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=total_configs_train, n_samples=n_samples, seed=seed, chunk_size=chunk_size)

# Modification déterministe de l'état du sampler
sigma_orig = vs.sampler_state.σ
flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
half = flat_sigma.shape[0] // 2

flat_sigma = flat_sigma.at[:half, :n_spins].set(1)
flat_sigma = flat_sigma.at[half:, :n_spins].set(-1)

sigma_new = flat_sigma.reshape(sigma_orig.shape)
vs.sampler_state = vs.sampler_state.replace(σ=sigma_new)

# Initialisation des paramètres (désordre)
params_list = generate_multi_h0_disorder(h0_train_list, n_replicas, n_spins, sigma=sigma_disorder)
vs.parameter_array = params_list

# Opérateurs
Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(n_spins)) * (1 / float(n_spins))

def create_operator(params):
    assert params.shape == (n_spins,)
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(n_spins))
    
    # Interactions 2D avec PBC exactes
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i % L + 1) % L + (i // L) * L) for i in range(n_spins))
    ha_ZZ += sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + L) % n_spins) for i in range(n_spins))
    
    return -ha_X - J_val * ha_ZZ

ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
mz_p = nkf.operator.ParametrizedOperator(hi, ps, lambda _: Mz)

# Callback Mémoire pour logs séquentiels
class ReplicaLogger(AbstractCallback):
    params_list: np.ndarray = struct.field(pytree_node=False)
    n_spins: int = struct.field(pytree_node=False)
    eval_every: int = struct.field(pytree_node=False)
    
    def __init__(self, params_list, n_spins, eval_every=10):
        self.params_list = params_list
        self.n_spins = n_spins
        self.eval_every = eval_every
        
    def __call__(self, step, log_data, driver):
        if step % self.eval_every != 0:
            return True
            
        vs = driver.state
        hi = vs.hilbert
        sa_eval = nk.sampler.MetropolisLocal(hi, n_chains=4)
        ham_dict = {}
        
        for i, pars in enumerate(self.params_list):
            _vs = vs.get_state(pars)
            
            mc_vs = nk.vqs.MCState(
                sampler=sa_eval, 
                model=_vs.model, 
                variables=_vs.variables, 
                n_samples=256,
                chunk_size=16
            )
            mc_vs.reset()
            
            sigma_orig = np.array(mc_vs.sampler_state.σ)
            flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
            half = flat_sigma.shape[0] // 2
            flat_sigma[:half, :self.n_spins] = 1
            flat_sigma[half:, :self.n_spins] = -1
            sigma_new = jnp.array(flat_sigma.reshape(sigma_orig.shape))
            mc_vs.sampler_state = mc_vs.sampler_state.replace(**{'σ': sigma_new})
            
            H_op = create_operator(pars)
            stats = mc_vs.expect(H_op)
            
            ham_dict[str(i)] = {
                "Mean": float(np.real(stats.Mean)),
                "Variance": float(stats.variance)
            }
            
        log_data["ham"] = ham_dict
        return True

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

meta = {
    "L": L,
    "nb_spins": n_spins,
    "graph": "Square Grid 2D",
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
        "type": "MetropolisSampler", 
        "n_chains": n_chains, 
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
except Exception:
    run_dir = "checkpoints"

log = nk.logging.JsonLog(os.path.join(run_dir, "log_data.json"), save_params=False)

disorder_path = os.path.join(run_dir, "disorder_configs.npy")
np.save(disorder_path, params_list)

vs.chunk_size = chunk_size

start_time = time.time()

gs.run(
    n_iter,
    out=log,
    callback=[
        SaveState(run_dir, 10), 
        ReplicaLogger(params_list, n_spins, eval_every=10)
    ]
)

if "Energy" in log.data:
    log.data["ham"] = log.data["Energy"]

duration = time.time() - start_time

meta["execution_time_seconds"] = duration
import json
with open(os.path.join(run_dir, "meta.json"), 'w') as f:
    json.dump(meta, f, indent=4)

# ==========================================
# 4. PLOTS ET ANALYSE FINALE
# ==========================================

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